### DeepIM: Deep Iterative Matching for 6D Pose Estimation
Yi Li, Gu Wang, Xiangyang Ji, Yu Xiang and Dieter Fox.
In ECCV, 2018.
[arXiv](https://arxiv.org/abs/1804.00175), [project page](https://rse-lab.cs.washington.edu/projects/deepim/)

This is an official MXNet implementation mainly developed and maintained by [Yi Li](https://github.com/liyi14) and [Gu Wang](https://github.com/wangg12).

**News** (2020-12-04): A PyTorch implementation of DeepIM by [Yu Xiang](https://github.com/yuxng) has been released ([here](https://github.com/NVlabs/DeepIM-PyTorch))!

### Citing DeepIM
If you find DeepIM useful in your research, please consider citing:
```
@inproceedings{li2018deepim,
title     = {DeepIM: Deep Iterative Matching for 6D Pose Estimation},
author    = {Yi Li and Gu Wang and Xiangyang Ji and Yu Xiang and Dieter Fox},
booktitle = {European Conference on Computer Vision (ECCV)},
year      = {2018}
}
```
## Overall Framework
![intro](https://github.com/user-attachments/assets/ad97ce78-2449-4f4a-b8a8-c9635a3079ef)

### Network Structure
![net_structure](https://github.com/user-attachments/assets/e78e96ef-1bd8-4758-b77b-ac57b1505b2d)

### Zoom In Operation
![zoom_in](https://github.com/user-attachments/assets/2c2c204f-4042-412e-bbe3-ca5543897bd5)

## Main Results

### LINEMOD
![LM6d_table](https://github.com/user-attachments/assets/7e3a5f91-2656-4598-b8fa-21161e1af136)

### Occlusion LINEMOD
![LM6d_Occ_results](https://github.com/user-attachments/assets/57b10958-12e3-4bd7-a31b-bd583f31c55a)
![LM6d_Occ_results_pictures](https://github.com/user-attachments/assets/6b1ad035-dc7a-41b6-9fc3-a273ded72e05)

### Unseen Objects from ModelNet
![Unseen](https://github.com/user-attachments/assets/bfcb57b0-d814-426f-961a-3e72ada5be7a)

The red and green lines
represent the edges of 3D model projected from the initial poses and our refined poses
respectively.
