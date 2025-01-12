import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNetEncoder(nn.Module):
    """
    PointNet Encoder with Rc-based normalization:
    - BatchNorm 제거.
    - STN 대신 Rc를 사용하여 입력 정규화.
    """
    def __init__(self, channel=3):
        super(PointNetEncoder, self).__init__()
        
        # Conv1D layers
        self.conv1 = nn.Conv1d(channel, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

    def forward(self, x):
        """
        Args:
            x: (B, N, D) Input point cloud (D=3 typically for x, y, z coordinates)
            Rc: (B, 3, 3) Canonical rotation matrix for input normalization
        Returns:
            global_feat: (B, 1024) Global feature vector
            point_feat: (B, N, 64) Local point features
        """
        if Rc is None:
            raise ValueError("Rc (canonical rotation matrix) is required but not provided.")

        B, N, D = x.size()
        if D != 3:
            raise ValueError(f"Input x must have 3 dimensions (x, y, z coordinates), but got {D} dimensions.")

        # Transpose to (B, D, N) for Conv1D
        x = x.transpose(2, 1)  # (B, D, N)

        # Extract features
        x = F.relu(self.conv1(x))  # (B, 64, N)
        point_feat = x  # Local point features
        x = F.relu(self.conv2(x))  # (B, 128, N)
        x = self.conv3(x)  # (B, 1024, N)

        # Global feature (max pooling)
        x = torch.max(x, 2, keepdim=False)[0]  # (B, 1024)
        global_feat = x

        return global_feat, point_feat