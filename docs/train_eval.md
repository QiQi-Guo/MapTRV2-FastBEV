# Prerequisites

**Please ensure you have prepared the environment and the nuScenes dataset.**

# Train and Test

Train MapTR with 4 GPUs 
```
./tools/dist_train.sh ./projects/configs/maptrv2/maptrv2_nusc_r50_24ep_w_centerline_fastbev/7_lvl_depth_seg_3.py 4
```

Eval MapTR with 8 GPUs
```
./tools/dist_test_map.sh ./projects/configs/maptrv2/maptrv2_nusc_r50_24ep_w_centerline_fastbev/7_lvl_depth_seg_3.py ./ckpts/7_lvl_depth_seg_3.pth 4
```




# Visualization 

we provide tools for visualization and benchmark under `path/to/MapTR/tools/maptr`
