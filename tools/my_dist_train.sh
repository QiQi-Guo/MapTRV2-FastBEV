#!/usr/bin/env bash

#CONFIG=$1
GPUS=$1
PORT=${PORT:-28509}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py \
     /home/adas4gpu1/Documents/guoqi/MapTR-maptrv2/projects/configs/maptrv2/maptrv2_nusc_r50_24ep_loss_seg_5.py \
    --launcher pytorch ${@:3} --deterministic \
    --work-dir $(dirname "$0")/../work_dirs/without_centerline/lr_5e-4_loss_seg_5

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py \
    /home/adas4gpu1/Documents/guoqi/MapTR-maptrv2/projects/configs/maptrv2/maptrv2_nusc_r50_24ep_loss_seg_7.py \
    --launcher pytorch ${@:3} --deterministic \
    --work-dir $(dirname "$0")/../work_dirs/without_centerline/lr_5e-4_loss_seg_7

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py \
    /home/adas4gpu1/Documents/guoqi/MapTR-maptrv2/projects/configs/maptrv2/maptrv2_nusc_r50_24ep_loss_seg_9.py \
    --launcher pytorch ${@:3} --deterministic \
    --work-dir $(dirname "$0")/../work_dirs/without_centerline/lr_5e-4_loss_seg_9

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py \
    /home/adas4gpu1/Documents/guoqi/MapTR-maptrv2/projects/configs/maptrv2/maptrv2_nusc_r50_24ep_loss_seg_11.py \
    --launcher pytorch ${@:3} --deterministic \
    --work-dir $(dirname "$0")/../work_dirs/without_centerline/lr_5e-4_loss_seg_11