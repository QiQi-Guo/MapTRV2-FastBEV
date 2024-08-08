# -*- coding: utf-8 -*-
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmcv.runner import force_fp32
from mmdet.models.backbones.resnet import BasicBlock
from mmcv.cnn import build_conv_layer
from torch.cuda.amp.autocast_mode import autocast

from mmdet.models import DETECTORS, build_backbone, build_head, build_neck
from mmseg.models import build_head as build_seg_head
from mmdet.models.detectors import BaseDetector
from mmdet3d.core import bbox3d2result
from mmseg.ops import resize
from mmcv.runner import get_dist_info, auto_fp16
import numpy as np

import copy

import cv2

from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER_SEQUENCE

class Mlp(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.ReLU,
                 drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class SELayer(nn.Module):

    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.conv_reduce = nn.Conv2d(channels, channels, 1, bias=True)
        self.act1 = act_layer()
        self.conv_expand = nn.Conv2d(channels, channels, 1, bias=True)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)

class DepthNet(nn.Module):

    def __init__(self,
                 in_channels,
                 mid_channels,
                 context_channels,
                 depth_channels,
                 use_dcn=True,
                 use_aspp=True,
                 with_cp=False,
                 aspp_mid_channels=-1,
                 only_depth=False):
        super(DepthNet, self).__init__()
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(
                in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.only_depth = only_depth or context_channels == 0
        if not self.only_depth:
            self.context_conv = nn.Conv2d(
                mid_channels, context_channels, kernel_size=1, stride=1, padding=0)
            self.context_mlp = Mlp(22, mid_channels, mid_channels)
            self.context_se = SELayer(mid_channels)  # NOTE: add camera-aware
        self.bn = nn.BatchNorm1d(22)
        self.depth_mlp = Mlp(22, mid_channels, mid_channels)
        self.depth_se = SELayer(mid_channels)  # NOTE: add camera-aware

        depth_conv_list = [
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
        ]
        if use_aspp:
            if aspp_mid_channels<0:
                aspp_mid_channels = mid_channels
            depth_conv_list.append(ASPP(mid_channels, aspp_mid_channels))
        if use_dcn:
            depth_conv_list.append(
                build_conv_layer(
                    cfg=dict(
                        type='DCN',
                        in_channels=mid_channels,
                        out_channels=mid_channels,
                        kernel_size=3,
                        padding=1,
                        groups=4,
                        im2col_step=128,
                    )))
        depth_conv_list.append(
            nn.Conv2d(
                mid_channels,
                depth_channels,
                kernel_size=1,
                stride=1,
                padding=0))
        self.depth_conv = nn.Sequential(*depth_conv_list)
        self.with_cp = with_cp

    def forward(self, x, mlp_input):
        mlp_input = self.bn(mlp_input.reshape(-1, mlp_input.shape[-1]))
        x = self.reduce_conv(x)
        if not self.only_depth:
            context_se = self.context_mlp(mlp_input)[..., None, None]
            context = self.context_se(x, context_se)
            context = self.context_conv(context)
        depth_se = self.depth_mlp(mlp_input)[..., None, None]
        depth = self.depth_se(x, depth_se)
        if self.with_cp:
            depth = checkpoint(self.depth_conv, depth)
        else:
            depth = self.depth_conv(depth)
        if not self.only_depth:
            return torch.cat([depth, context], dim=1)
        else:
            return depth

@TRANSFORMER_LAYER_SEQUENCE.register_module()
class FastBEV(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        feat_down_sample,
        pc_range,
        voxel_size,
        origin,
        dbound,
        downsample,
        loss_depth_weight,
        depthnet_cfg,
        grid_config,
        bev_layers,

        extrinsic_noise=0,
        multi_scale_id=[0],
        multi_scale_3d_scaler=None,
        with_cp=False,
        backproject='inplace',
    ):
        super().__init__()

        self.multi_scale_id = multi_scale_id
        self.multi_scale_3d_scaler = multi_scale_3d_scaler
        self.feat_down_sample = feat_down_sample
        self.loss_depth_weight = loss_depth_weight
        self.grid_config = grid_config
        self.extrinsic_noise = extrinsic_noise

        self.voxel_size = [voxel_size]
        self.n_voxels = [[(pc_range[3] - pc_range[0]) / voxel_size[0],
                         (pc_range[4] - pc_range[1]) / voxel_size[1],
                         (pc_range[5] - pc_range[2]) / voxel_size[2],],]
        self.origin = origin

        self.C = out_channels
        self.D = int((dbound[1] - dbound[0]) / dbound[2])

        # detach adj feature
        self.backproject = backproject
        # checkpoint
        self.with_cp = with_cp
        # self.depth_net = DepthNet(in_channels, in_channels,
        #                           self.C, self.D, **depthnet_cfg)

        self.bev_layers = bev_layers
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels * self.bev_layers, out_channels, 3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),

            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

        if downsample > 1:
            assert downsample == 2, downsample
            self.downsample = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    3,
                    stride=downsample,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )
        else:
            self.downsample = nn.Identity()

    def extract_feat(self,
                     img_feats,
                     img_metas,
                     rots,
                     trans,
                     intrins,
                     post_rots,
                     post_trans,
                     lidar2ego_rots,
                     lidar2ego_trans,
                     ):
        batch_size = img_feats.shape[0]

        mlvl_feats = [img_feats]

        mlvl_volumes = []
        for lvl, mlvl_feat in enumerate(mlvl_feats):

            stride_i = math.ceil(img_metas[0]['img_shape'][0][1] / mlvl_feat.shape[-1])  # P4 880 / 32 = 27.5

            mlvl_feat_split = torch.split(mlvl_feat, 6, dim=1)

            volume_list = []
            for seq_id in range(len(mlvl_feat_split)):
                volumes = []
                for batch_id, seq_img_meta in enumerate(img_metas):
                    feat_i = mlvl_feat_split[seq_id][batch_id]  # [nv, c, h, w]
                    img_meta = copy.deepcopy(seq_img_meta)
                    img_meta["lidar2cam"] = img_meta["lidar2cam"][seq_id*6:(seq_id+1)*6]
                    if isinstance(img_meta["img_shape"], list):
                        img_meta["img_shape"] = img_meta["img_shape"][seq_id*6:(seq_id+1)*6]
                        img_meta["img_shape"] = img_meta["img_shape"][0]
                    height = math.ceil(img_meta["img_shape"][0] / stride_i)
                    width = math.ceil(img_meta["img_shape"][1] / stride_i)


                    n_voxels, voxel_size = self.n_voxels[0], self.voxel_size[0]

                    points = get_points(  # [3, vx, vy, vz]
                        n_voxels=torch.tensor(n_voxels),
                        voxel_size=torch.tensor(voxel_size),
                        origin=torch.tensor(self.origin),
                    ).to(feat_i.device)

                    volume = backproject_inplace(
                        feat_i[:, :, :height, :width],
                        points,
                        img_meta,
                        stride_i,
                        noise=self.extrinsic_noise
                    )  # [c, vx, vy, vz]

                    volumes.append(volume)
                volume_list.append(torch.stack(volumes))  # list([bs, c, vx, vy, vz])
    
            mlvl_volumes.append(torch.cat(volume_list, dim=1))  # list([bs, seq*c, vx, vy, vz])

        mlvl_volumes = torch.cat(mlvl_volumes, dim=1)  # [bs, lvl*seq*c, vx, vy, vz]

        B, C, W, L, H = mlvl_volumes.shape
        x = mlvl_volumes.permute(0, 1, 4, 2, 3).contiguous().view(B, C*H, W, L)

        x = x.squeeze(dim=-1).permute(0, 1, 3, 2).contiguous()

        return x

    def get_cam_feats(self, x, mlp_input):
        B, N, C, fH, fW = x.shape

        x = x.view(B * N, C, fH, fW)

        x = self.depth_net(x, mlp_input)
        depth = x[:, : self.D].softmax(dim=1)  # 6,68,15,25  得到每个像素的gailv 深度概率
        x = depth.unsqueeze(1) * x[:, self.D : (self.D + self.C)].unsqueeze(2)

        x = x.view(B, N, self.C, self.D, fH, fW)
        x = x.permute(0, 1, 3, 4, 5, 2)
        depth = depth.view(B, N, self.D, fH, fW)
        return x, depth

    @force_fp32()
    def get_depth_loss(self, depth_labels, depth_preds):
        # import pdb;pdb.set_trace()
        if depth_preds is None:
            return 0

        depth_labels = self.get_downsampled_gt_depth(depth_labels)
        depth_preds = depth_preds.permute(0, 1, 3, 4, 2).contiguous().view(-1, self.D)
        # fg_mask = torch.max(depth_labels, dim=1).values > 0.0 # 只计算有深度的前景的深度loss
        # import pdb;pdb.set_trace()
        fg_mask = depth_labels > 0.0  # 只计算有深度的前景的深度loss
        depth_labels = depth_labels[fg_mask]
        depth_preds = depth_preds[fg_mask]
        with autocast(enabled=False):
            depth_loss = F.binary_cross_entropy(
                depth_preds,
                depth_labels,
                reduction='none',
            ).sum() / max(1.0, fg_mask.sum())
        # if depth_loss <= 0.:
        #     import pdb;pdb.set_trace()
        return self.loss_depth_weight * depth_loss

    def get_downsampled_gt_depth(self, gt_depths):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(B * N, H // self.feat_down_sample,
                                   self.feat_down_sample, W // self.feat_down_sample,
                                   self.feat_down_sample, 1)
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(-1, self.feat_down_sample * self.feat_down_sample)
        # 把gt_depth做feat_down_sample倍数的采样
        gt_depths_tmp = torch.where(gt_depths == 0.0,
                                    1e5 * torch.ones_like(gt_depths),
                                    gt_depths)
        # 因为深度很稀疏，大部分的点都是0，所以把0变成10000，下一步取-1维度上的最小就是深度的值
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(B * N, H // self.feat_down_sample,
                                   W // self.feat_down_sample)

        gt_depths = (
            gt_depths -
            (self.grid_config['depth'][0] -
             self.grid_config['depth'][2])) / self.grid_config['depth'][2]
        gt_depths = torch.where((gt_depths < self.D + 1) & (gt_depths >= 0.0),
                                gt_depths, torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(
            gt_depths.long(), num_classes=self.D + 1).view(-1, self.D + 1)[:,
                                                                           1:]
        return gt_depths.float()

    def get_mlp_input(self, sensor2ego, intrin, post_rot, post_tran):
        B, N, _, _ = sensor2ego.shape
        mlp_input = torch.stack([
            intrin[:, :, 0, 0],
            intrin[:, :, 1, 1],
            intrin[:, :, 0, 2],
            intrin[:, :, 1, 2],
            post_rot[:, :, 0, 0],
            post_rot[:, :, 0, 1],
            post_tran[:, :, 0],
            post_rot[:, :, 1, 0],
            post_rot[:, :, 1, 1],
            post_tran[:, :, 1],
        ], dim=-1)
        sensor2ego = sensor2ego[:, :, :3, :].reshape(B, N, -1)
        mlp_input = torch.cat([mlp_input, sensor2ego], dim=-1)
        return mlp_input

    # @auto_fp16(apply_to=('img', ))
    def forward(self, img_feats, img_metas):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        # B, N, C, fH, fW = img_feats.shape  # 1,6,256,15,25
        lidar2img = []
        camera2ego = []
        camera_intrinsics = []
        img_aug_matrix = []
        lidar2ego = []

        for img_meta in img_metas:
            lidar2img.append(img_meta['lidar2img'])
            camera2ego.append(img_meta['camera2ego'])
            camera_intrinsics.append(img_meta['camera_intrinsics'])
            img_aug_matrix.append(img_meta['img_aug_matrix'])
            lidar2ego.append(img_meta['lidar2ego'])
        lidar2img = np.asarray(lidar2img)  # 1,6,4,4
        lidar2img = img_feats.new_tensor(lidar2img)  # (B, N, 4, 4)
        camera2ego = np.asarray(camera2ego)  # 1,6,4,4
        camera2ego = img_feats.new_tensor(camera2ego)  # (B, N, 4, 4)
        camera_intrinsics = np.asarray(camera_intrinsics)
        camera_intrinsics = img_feats.new_tensor(camera_intrinsics)  # (B, N, 4, 4)
        img_aug_matrix = np.asarray(img_aug_matrix)
        img_aug_matrix = img_feats.new_tensor(img_aug_matrix)  # (B, N, 4, 4)
        lidar2ego = np.asarray(lidar2ego)
        lidar2ego = img_feats.new_tensor(lidar2ego)  # (B, N, 4, 4)

        rots = camera2ego[..., :3, :3]  # 1,6,3,3
        trans = camera2ego[..., :3, 3]  # 1,6,3
        intrins = camera_intrinsics[..., :3, :3]  # 1,6,3,3
        post_rots = img_aug_matrix[..., :3, :3]  # 1,6,3,3
        post_trans = img_aug_matrix[..., :3, 3]  # 1,6,3
        lidar2ego_rots = lidar2ego[..., :3, :3]  # 1,3,3
        lidar2ego_trans = lidar2ego[..., :3, 3]  # 1,3


        # mlp_input = self.get_mlp_input(camera2ego, camera_intrinsics, post_rots, post_trans)  # 1,6,22
        # _, depth = self.get_cam_feats(img_feats, mlp_input)

        x = self.extract_feat(
            img_feats,
            img_metas,
            rots,
            trans,
            intrins,
            post_rots,
            post_trans,
            lidar2ego_rots,
            lidar2ego_trans,
        )
        # x = x.permute(0, 1, 3, 2).contiguous()

        x = self.fuse(x)

        x = self.downsample(x)
        ret_dict = dict(
            bev=x,
            # depth=depth,
            depth=None,
        )

        return ret_dict

@torch.no_grad()
def get_points(n_voxels, voxel_size, origin):
    # 3,200,200,4
    points = torch.stack(
        torch.meshgrid(
            [
                torch.arange(n_voxels[0]),
                torch.arange(n_voxels[1]),
                torch.arange(n_voxels[2]),
            ]
        )
    )
    new_origin = origin - n_voxels / 2.0 * voxel_size
    points = points * voxel_size.view(3, 1, 1, 1) + new_origin.view(3, 1, 1, 1)
    points = points[..., :3]
    return points

def backproject_inplace(features, points, img_meta, stride, noise):
    '''
    function: 2d feature + predefined point cloud -> 3d volume
    input:
        features: [6, 64, 225, 400]
        points: [3, 200, 200, 12]
        projection: [6, 3, 4]
    output:
        volume: [64, 200, 200, 12]
    '''
    n_images, n_channels, height, width = features.shape
    n_x_voxels, n_y_voxels, n_z_voxels = points.shape[-3:]
    # [3, 200, 200, 12] -> [1, 3, 480000] -> [6, 3, 480000]
    points = points.view(1, 3, -1).expand(n_images, 3, -1)
    # [6, 3, 480000] -> [6, 4, 480000]
    # points = torch.cat((points, torch.ones_like(points[:, :1])), dim=1)
    # ego_to_cam
    # [6, 3, 4] * [6, 4, 480000] -> [6, 3, 480000]

    # lidar2cam = features.new_tensor(img_meta['lidar2cam']).to(features.device)
    lidar2ego_rot = torch.tensor(img_meta['lidar2ego'])[:3, :3].to(features.device)
    lidar2ego_tran = torch.tensor(img_meta['lidar2ego'])[:3, 3].to(features.device)
    points = lidar2ego_rot.view(1, 3, 3).matmul(points)
    points += lidar2ego_tran.view(1, 3, 1)

    camera2ego_rots = torch.tensor(img_meta['camera2ego'])[:, :3, :3].to(features.device)
    camera2ego_trans = torch.tensor(img_meta['camera2ego'])[:, :3, 3].to(features.device).view(6, 1, 3)
    points -= camera2ego_trans.view(6, 3, 1)
    points = torch.inverse(camera2ego_rots).matmul(points)

    intrinsics = torch.tensor(img_meta['camera_intrinsics'])[:, :3, :3].to(features.device)
    intrinsics[:, :2] /= stride
    points = intrinsics.matmul(points)

    img_aug_matrix_rots = features.new_tensor(img_meta['img_aug_matrix'])[:, :3, :3].to(features.device)
    img_aug_matrix_trans = features.new_tensor(img_meta['img_aug_matrix'])[:, :3, 3].to(features.device)
    points = img_aug_matrix_rots.matmul(points)
    points += img_aug_matrix_trans.view(6, 3, 1)
    points_2d_3 = points

    # lidar2ego = torch.tensor(img_meta['lidar2ego']).to(features.device)
    # camera2ego = torch.tensor(img_meta['camera2ego']).to(features.device)
    # lidar2cam = lidar2ego.matmul(torch.inverse(camera2ego))
    # lidar2img = intrinsics.matmul(lidar2cam)

    # img_aug_matrix = features.new_tensor(img_meta['img_aug_matrix']).to(features.device)

    # projection = img_aug_matrix.matmul(lidar2img)
    # projection = lidar2img

    # points_2d_3 = projection.matmul(points)

    x = (points_2d_3[:, 0] / points_2d_3[:, 2]).round().long()  # [6, 480000]
    y = (points_2d_3[:, 1] / points_2d_3[:, 2]).round().long()  # [6, 480000]
    z = points_2d_3[:, 2]  # [6, 480000]
    valid = (x >= 0) & (y >= 0) & (x < width) & (y < height) & (z > 0)  # [6, 480000]
    # valid = (x >= 0) & (y >= 0) & (x < 800) & (y < 480) & (z > 0)  # [6, 480000]

    # #----------------------------------------------------projection test-----------------------------------------------#
    # for img_id, img in enumerate(n_img):
    #     img[0] = img[0] * 58.395 + 123.675
    #     img[1] = img[1] * 57.12 + 116.28
    #     img[2] = img[2] * 57.375 + 103.53
    #     img = img.to(torch.int16)
    #     img = img.cpu().numpy().transpose((1, 2, 0))
    #
    #     coor_x = 799-x[img_id].unsqueeze(0)
    #     coor_y = 479-y[img_id].unsqueeze(0)
    #     coor = torch.cat((coor_x,coor_y), dim=0)
    #     coor = coor * valid[img_id].view((1, -1))
    #     feat_zeros = torch.zeros((480,800,3))
    #     for coor_id in range(coor.shape[1]):
    #         if valid[img_id, coor_id]:
    #             # feat_zeros[coor[1, coor_id], coor[0, coor_id]] = torch.tensor((128, 0, 0))
    #             img[coor[1, coor_id], coor[0, coor_id]] = torch.tensor((255, 255, 255))
    #     # feat_zeros = feat_zeros.cpu().numpy()
    #
    #     # feat_zeros = cv2.resize(feat_zeros,(800, 480))
    #     # img = img + feat_zeros
    #     cv2.imwrite(f'./img/{img_id}.png', img)
    # #----------------------------------------------------projection test-----------------------------------------------#

    # method2：特征填充，只填充有效特征，重复特征直接覆盖
    volume = torch.zeros(
        (n_channels, points.shape[-1]), device=features.device
    ).type_as(features)
    for i in range(n_images):
        volume[:, valid[i]] = features[i, :, y[i, valid[i]], x[i, valid[i]]]

    volume = volume.view(n_channels, n_x_voxels, n_y_voxels, n_z_voxels)
    return volume

class _ASPPModule(nn.Module):

    def __init__(self, inplanes, planes, kernel_size, padding, dilation,
                 BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):

    def __init__(self, inplanes, mid_channels=256, BatchNorm=nn.BatchNorm2d):
        super(ASPP, self).__init__()

        dilations = [1, 6, 12, 18]

        self.aspp1 = _ASPPModule(
            inplanes,
            mid_channels,
            1,
            padding=0,
            dilation=dilations[0],
            BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(
            inplanes,
            mid_channels,
            3,
            padding=dilations[1],
            dilation=dilations[1],
            BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(
            inplanes,
            mid_channels,
            3,
            padding=dilations[2],
            dilation=dilations[2],
            BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(
            inplanes,
            mid_channels,
            3,
            padding=dilations[3],
            dilation=dilations[3],
            BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, mid_channels, 1, stride=1, bias=False),
            BatchNorm(mid_channels),
            nn.ReLU(),
        )
        self.conv1 = nn.Conv2d(
            int(mid_channels * 5), inplanes, 1, bias=False)
        self.bn1 = BatchNorm(inplanes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(
            x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
