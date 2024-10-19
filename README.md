# 6D Pose Estimation Study

![6dposebackground](https://github.com/user-attachments/assets/41d1f23d-f878-4563-b33d-88a8c93d29bd)

<br>

<div align="center">

![Python](https://img.shields.io/badge/python-v3.8+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-v1.8+-red)
![torchvision](https://img.shields.io/badge/torchvision-v0.9+-orange)
![numpy](https://img.shields.io/badge/numpy-v1.19+-lightgrey)
![OpenCV](https://img.shields.io/badge/OpenCV-v4.5+-yellowgreen)
![License](https://img.shields.io/badge/license-Apache%202.0-green)
![Issues](https://img.shields.io/github/issues/your-repo/your-project)


</div>

<br>

## Introduction

6D pose estimation is crucial in robotics and computer vision because it allows robots to understand the position and orientation (pose) of objects in a 3D space, which is essential for tasks like manipulation, navigation, and interaction with the physical world. Accurate 6D pose estimation enables robots to perform complex operations such as grasping objects, assembling parts, and navigating dynamic environments.

This study aims to first explore and understand the foundational research papers that serve as the basis for 6D pose estimation techniques. Following that, we will focus on creating a custom dataset tailored to specific use cases, and ultimately apply state-of-the-art 6D pose estimation methods on this dataset.

The overall goal is to solidify the theoretical understanding and then gain hands-on experience by implementing and optimizing 6D pose estimation for real-world applications.

<br>

## Team Members

| ![동준](https://github.com/user-attachments/assets/2dbbc136-5342-4a26-8e95-56c8e8076d27) | ![태욱](https://github.com/user-attachments/assets/1b4d1a10-2600-400a-a57c-bb057227cf57) | ![정우](https://github.com/user-attachments/assets/91efbc28-3219-49eb-bebe-c71f329a88cf) | ![윤서](https://github.com/user-attachments/assets/55d9c740-f067-4cc3-b1ab-43a9466029aa) | ![채리](https://github.com/user-attachments/assets/5e491b00-daaa-461b-8ba6-67e06e3caf39) |
|:----------------------------------------------------------:|:----------------------------------------------------------:|:----------------------------------------------------------:|:----------------------------------------------------------:|:----------------------------------------------------------:|
| **동준**                                                     | **태욱**                                                         | **정우**                                                         | **윤서**                                                         | **채리**                                                         |




<br>

## 📅 Study Timeline

```mermaid
gantt
    title Timeline
    dateFormat  YYYY-MM-DD
    excludes    weekends
    axisFormat  %Y-%m-%d
    section Read Papers
    PoseCNN        :active,  des1, 2024-10-19, 2024-11-02
    SSD-6D         :         des2, 2024-11-03, 2024-11-16
    PVNet          :         des3, 2024-11-17, 2024-11-30
    DenseFusion    :         des4, 2024-12-01, 2024-12-14
    PointFusion    :         des5, 2024-12-15, 2024-12-28
    DeepIM         :         des6, 2024-12-29, 2025-01-11

```


<br>

---

## 📚 Study Topics

<details>
<summary><b>PoseCNN</b></summary>
  
- **Title**: PoseCNN: A Convolutional Neural Network for 6D Object Pose Estimation in Cluttered Scenes  
- **Key Focus**: Feature extraction from RGB-D data and 6D pose prediction using CNN-based architecture.  
- **Notes**: This study will focus on the integration of depth information for improving pose accuracy.

</details>

<details>
<summary><b>SSD-6D</b></summary>
  
- **Title**: SSD-6D: Making RGB-Based 3D Object Pose Estimation Efficient  
- **Key Focus**: Fast and efficient 6D pose estimation using a single-stage detection model.  
- **Notes**: Key study area is optimizing real-time performance while maintaining accuracy.

</details>

<details>
<summary><b>PVNet</b></summary>
  
- **Title**: PVNet: Pixel-wise Voting Network for 6DoF Pose Estimation  
- **Key Focus**: A novel approach that utilizes pixel-wise voting for predicting the 6D pose of objects.  
- **Notes**: Special attention will be paid to the voting mechanism and its impact on pose accuracy.

</details>

<details>
<summary><b>DenseFusion</b></summary>
  
- **Title**: DenseFusion: 6D Object Pose Estimation by Iterative Dense Fusion  
- **Key Focus**: Combining RGB and depth information via an iterative fusion process to refine pose estimation.  
- **Notes**: A deep dive into how DenseFusion integrates data at different stages for 6D pose refinement.

</details>

<details>
<summary><b>PointFusion</b></summary>
  
- **Title**: PointFusion: Deep Sensor Fusion for 3D Bounding Box Estimation  
- **Key Focus**: Fusing RGB and point cloud data for more accurate 3D object pose predictions.  
- **Notes**: This study will explore how PointFusion bridges sensor fusion techniques for pose estimation.

</details>

<details>
<summary><b>DeepIM</b></summary>
  
- **Title**: DeepIM: Deep Iterative Matching for 6D Pose Estimation  
- **Key Focus**: Utilizing an iterative matching framework to enhance object pose estimation over multiple steps.  
- **Notes**: Exploration of how this iterative process boosts pose accuracy over initial estimates.

</details>
