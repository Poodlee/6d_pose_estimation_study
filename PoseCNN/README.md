# PoseCNN-PyTorch: A PyTorch Implementation of the PoseCNN Framework for 6D Object Pose Estimation

### Introduction

We implement PoseCNN in PyTorch in this project.

PoseCNN is an end-to-end Convolutional Neural Network for 6D object pose estimation. PoseCNN estimates the 3D translation of an object by localizing its center in the image and predicting its distance from the camera. The 3D rotation of the object is estimated by regressing to a quaternion representation. [arXiv](https://arxiv.org/abs/1711.00199), [Project](https://rse-lab.cs.washington.edu/projects/posecnn/)

Rotation regression in PoseCNN cannot handle symmetric objects very well. Check [PoseRBPF](https://github.com/NVlabs/PoseRBPF) for a better solution for symmetric objects.

The code also supports pose refinement by matching segmented 3D point cloud of an object to its SDF.

<div align="center">
  <img src="https://github.com/user-attachments/assets/0a2b1c36-06fa-4682-8cdb-fd941ec990c5" alt="intro" width="640" height="320">
</div>

### License

PoseCNN-PyTorch is released under the NVIDIA Source Code License (refer to the LICENSE file for details).

### Citation

If you find the package is useful in your research, please consider citing:

    @inproceedings{xiang2018posecnn,
        Author = {Yu Xiang and Tanner Schmidt and Venkatraman Narayanan and Dieter Fox},
        Title = {{PoseCNN}: A Convolutional Neural Network for {6D} Object Pose Estimation in Cluttered Scenes},
        booktitle = {Robotics: Science and Systems (RSS)},
        Year = {2018}
    }

### Required environment

- Ubuntu 16.04 or above
- PyTorch 0.4.1 or above
- CUDA 9.1 or above

