import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms

import PointNet
import ResNet

class PointFusion(nn.Module):
    def __init__(self, pnt_cnt=100):
        super(PointFusion, self).__init__()
        self.image_embedding = ResNet.ResNetFeatures()
        self.pcl_embedding = PointNet.PointNetEncoder(channel=3)

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 16, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 24)
        self.fc5 = nn.Linear(128, 1)

        self.fusion_dropout = nn.Dropout2d(p=0.4)
        self.relu = torch.nn.ReLU()
        self.softmax = nn.Softmax(dim=0)

        # Camera intrinsic parameters
        fx, fy = 572.41140, 573.57043
        cx, cy = 325.26110, 242.04899
        self.intrinsic_matrix = torch.tensor(
            [[fx, 0, cx],
             [0, fy, cy],
             [0,  0,  1]],
            dtype=torch.float32
        )

        # Camera position (assume origin)
        self.camera_position = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)

    def forward(self, img, cloud):
        B, N, D = cloud.shape    
        # Step 1: Extract cropped and aligned point cloud
        aligned_cloud, _ = self.project_image_to_3d(img, cloud)

        # Step 2: Extract RGB features from image
        img_feats = self.image_embedding(img)

        # Step 3: Extract point-wise and global features from aligned point cloud
        point_feats, global_feats = self.pcl_embedding(aligned_cloud)

        # Step 4: Duplicate features for each point in the aligned cloud
        img_feats = img_feats.repeat(1, aligned_cloud.size(1), 1)
        global_feats = global_feats.repeat(1, aligned_cloud.size(1), 1)

        # Step 5: Concatenate features and pass through MLP layers
        dense_feats = torch.cat([img_feats, point_feats, global_feats], 2)
        x = self.fc1(dense_feats)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)

        # Step 6: Compute corner offsets and scores
        corner_offsets = self.fc4(x).view(B, aligned_cloud.size(1), 8, 3)
        scores = self.softmax(self.fc5(x)).view(B, aligned_cloud.size(1))

        return corner_offsets, scores
    
    def project_image_to_3d(self, cropped_img, cloud):
        """
        Align the 3D points corresponding to the cropped image's central region with the z-axis.
        Args:
            cropped_img: (B, 3, H, W) Cropped image.
            cloud: (B, N, 3) Point cloud data.
        Returns:
            aligned_cloud: (B, M, 3) Points within the cropped region in 3D space, aligned to the z-axis.
        """
        B, _, H, W = cropped_img.size()
        device = cropped_img.device

        # 2D center pixel coordinates
        img_center_2d = torch.tensor(
            [W / 2.0, H / 2.0, 1.0], device=device
        ).unsqueeze(0).expand(B, -1)  # (B, 3)

        # Compute ray direction for 3D projection
        ray = torch.bmm(
            img_center_2d.unsqueeze(1), torch.inverse(self.intrinsic_matrix).unsqueeze(0).expand(B, -1, -1)
        ).squeeze(1)  # (B, 3)

        # Assume depth z from the point cloud mean z-value
        mean_z = cloud[:, :, 2].mean(dim=1, keepdim=True)  # (B, 1)
        bbox_center_3d = ray * mean_z  # (B, 3)

        # Compute distance of all points to bbox_center_3d
        distances = torch.norm(cloud - bbox_center_3d.unsqueeze(1), dim=2)  # (B, N)

        # Define a radius to select points around bbox_center_3d
        radius = 0.2  # Define the radius in 3D space (adjust as necessary)
        mask = distances < radius  # (B, N)

        # Filter points within the radius
        cropped_cloud = torch.stack([cloud[b, mask[b]] for b in range(B)], dim=0)  # (B, M, 3)

        if cropped_cloud.size(1) == 0:
            raise ValueError("No points found in the cropped region. Check input data or parameters.")

        # Compute Rc to align bbox_center_3d with the z-axis
        Rc = self.compute_rc(bbox_center_3d)

        # Translate and rotate cropped_cloud
        aligned_cloud = torch.bmm(cropped_cloud - bbox_center_3d.unsqueeze(1), Rc.transpose(2, 1))  # (B, M, 3)

        return aligned_cloud, bbox_center_3d

    def compute_rc(self, bbox_center_3d):
        """
        Compute the canonical rotation matrix Rc to align bbox_center_3d with the z-axis.
        Args:
            bbox_center_3d: (B, 3) 3D coordinates of the bounding box center.
        Returns:
            Rc: (B, 3, 3) Canonical rotation matrix for each batch.
        """
        B = bbox_center_3d.size(0)
        device = bbox_center_3d.device

        # Compute ray from camera to bbox_center_3d (camera at origin)
        ray = bbox_center_3d  # (B, 3) Since camera_position = (0, 0, 0)
        ray_norm = ray.norm(dim=1, keepdim=True)  # (B, 1)

        # Handle cases where the center is too close to the origin
        if torch.any(ray_norm < 1e-6):
            raise ValueError("bbox_center_3d is too close to the camera position (origin).")

        ray = ray / ray_norm  # Normalize to unit vector (B, 3)

        # Define the canonical z-axis
        z_axis = torch.tensor([0.0, 0.0, 1.0], device=device).unsqueeze(0).expand(B, -1)  # (B, 3)

        # Compute cross product and angles
        cross_product = torch.cross(ray, z_axis, dim=1)  # (B, 3)
        sin_theta = cross_product.norm(dim=1, keepdim=True)  # (B, 1)
        cos_theta = (ray * z_axis).sum(dim=1, keepdim=True)  # (B, 1)

        # Handle cases where ray is parallel to z-axis
        near_parallel = sin_theta < 1e-6
        Rc = torch.eye(3, device=device).unsqueeze(0).expand(B, -1, -1)  # Identity matrix for parallel cases

        if torch.any(~near_parallel):
            # Skew-symmetric matrix for cross product
            K = torch.zeros(B, 3, 3, device=device)
            K[:, 0, 1] = -cross_product[:, 2]
            K[:, 0, 2] =  cross_product[:, 1]
            K[:, 1, 0] =  cross_product[:, 2]
            K[:, 1, 2] = -cross_product[:, 0]
            K[:, 2, 0] = -cross_product[:, 1]
            K[:, 2, 1] =  cross_product[:, 0]

            # Rodrigues' formula: R = I + sinθ * K + (1 - cosθ) * K^2
            I = torch.eye(3, device=device).unsqueeze(0).expand(B, -1, -1)
            Rc[~near_parallel] = (
                I[~near_parallel]
                + sin_theta[~near_parallel].unsqueeze(2) * K[~near_parallel]
                + (1 - cos_theta[~near_parallel]).unsqueeze(2) * torch.bmm(K[~near_parallel], K[~near_parallel])
            )

        return Rc
