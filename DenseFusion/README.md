# DenseFusion

<p align="center">
	<img src ="assets/pullfig.png" width="1000" />
</p>

## News
We have released the code and arXiv preprint for our new project [6-PACK](https://sites.google.com/view/6packtracking) which is based on this work and used for category-level 6D pose tracking.

## Table of Content
- [Overview](#overview)
- [Results](#results)
- [Trained Checkpoints](#trained-checkpoints)
- [Tips for your own dataset](#tips-for-your-own-dataset)
- [Citations](#citations)
- [License](#license)

## Overview

This repository is the implementation code of the paper "DenseFusion: 6D Object Pose Estimation by Iterative Dense Fusion"([arXiv](https://arxiv.org/abs/1901.04780), [Project](https://sites.google.com/view/densefusion), [Video](https://www.youtube.com/watch?v=SsE5-FuK5jo)) by Wang et al. at [Stanford Vision and Learning Lab](http://svl.stanford.edu/) and [Stanford People, AI & Robots Group](http://pair.stanford.edu/). The model takes an RGB-D image as input and predicts the 6D pose of the each object in the frame. This network is implemented using [PyTorch](https://pytorch.org/) and the rest of the framework is in Python. Since this project focuses on the 6D pose estimation process, we do not specifically limit the choice of the segmentation models. You can choose your preferred semantic-segmentation/instance-segmentation methods according to your needs. In this repo, we provide our full implementation code of the DenseFusion model, Iterative Refinement model and a vanilla SegNet semantic-segmentation model used in our real-robot grasping experiment. The ROS code of the real robot grasping experiment is not included.

## Results

* YCB_Video Dataset:

Quantitative evaluation result with ADD-S metric compared to other RGB-D methods. `Ours(per-pixel)` is the result of the DenseFusion model without refinement and `Ours(iterative)` is the result with iterative refinement.

<p align="center">
	<img src ="assets/result_ycb.png" width="600" />
</p>

**Important!** Before you use these numbers to compare with your methods, please make sure one important issus: One difficulty for testing on the YCB_Video Dataset is how to let the network to tell the difference between the object `051_large_clamp` and `052_extra_large_clamp`. The result of all the approaches in this table uses the same segmentation masks released by PoseCNN without any detection priors, so all of them suffer a performance drop on these two objects because of the poor detection result and this drop is also added to the final overall score. If you have added detection priors to your detector to distinguish these two objects, please clarify or do not copy the overall score for comparsion experiments.

* LineMOD Dataset:

Quantitative evaluation result with ADD metric for non-symmetry objects and ADD-S for symmetry objects(eggbox, glue) compared to other RGB-D methods. High performance RGB methods are also listed for reference.

<p align="center">
	<img src ="assets/result_linemod.png" width="500" />
</p>

The qualitative result on the YCB_Video dataset.

<p align="center">
	<img src ="assets/compare.png" width="600" />
</p>

## Trained Checkpoints
You can download the trained DenseFusion and Iterative Refinement checkpoints of both datasets from [Link](https://drive.google.com/drive/folders/19ivHpaKm9dOrr12fzC8IDFczWRPFxho7).

## Tips for your own dataset
As you can see in this repo, the network code and the hyperparameters (lr and w) remain the same for both datasets. Which means you might not need to adjust too much on the network structure and hyperparameters when you use this repo on your own dataset. Please make sure that the distance metric in your dataset should be converted to meter, otherwise the hyperparameter w need to be adjusted. Several useful tools including [LabelFusion](https://github.com/RobotLocomotion/LabelFusion) and [sixd_toolkit](https://github.com/thodan/sixd_toolkit) has been tested to work well. (Please make sure to turn on the depth image collection in LabelFusion when you use it.)


## Citations
Please cite [DenseFusion](https://sites.google.com/view/densefusion) if you use this repository in your publications:
```
@article{wang2019densefusion,
  title={DenseFusion: 6D Object Pose Estimation by Iterative Dense Fusion},
  author={Wang, Chen and Xu, Danfei and Zhu, Yuke and Mart{\'\i}n-Mart{\'\i}n, Roberto and Lu, Cewu and Fei-Fei, Li and Savarese, Silvio},
  booktitle={Computer Vision and Pattern Recognition (CVPR)},
  year={2019}
}
```

## License
Licensed under the [MIT License](LICENSE)
