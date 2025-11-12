"""
PointNet encoder for robomimic.

Repo (clone this): https://github.com/yanx27/Pointnet_Pointnet2_pytorch

Usage in config:
    {
        "observation": {
            "modalities": {
                "obs": {
                    "low_dim": ["joint_positions", "gripper_position"],
                    "scan": ["pointcloud"]
                }
            },
            "encoder": {
                "scan": {
                    "core_class": "PointNetEncoder",
                    "core_kwargs": {
                        "feature_dimension": 256
                    }
                }
            }
        }
    }
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np

from robomimic.models.obs_core import EncoderCore

# CHANGE THIS PATH
POINTNET_PATH = "/home/raymond/projects/Pointnet_Pointnet2_pytorch/models"
if POINTNET_PATH not in sys.path:
    sys.path.insert(0, POINTNET_PATH)


class PointNetEncoder(EncoderCore):


    def __init__(
        self,
        input_shape,
        feature_dimension=256,
        use_normals=False,
    ):
        """
        Args:
            input_shape (tuple): Expected shape (N, 3) or (3, N) for point clouds
            feature_dimension (int): Output feature dimension
            use_normals (bool): Whether input includes normals (6D instead of 3D)
        """
        super().__init__(input_shape=input_shape)

        self.feature_dimension = feature_dimension
        self.use_normals = use_normals

        expected_channel = 6 if use_normals else 3
        if input_shape[0] == expected_channel:
            self.channel_first = True
            self.num_points = input_shape[1]
        elif input_shape[-1] == expected_channel:
            self.channel_first = False
            self.num_points = input_shape[0]
        else:
            raise ValueError(
                f"Expected input_shape with {expected_channel} channels, got {input_shape}. "
                f"Point clouds should be (N, {expected_channel}) or ({expected_channel}, N)"
            )

        try:
            from pointnet_utils import PointNetEncoder as PointNetFeat
        except ImportError as e:
            raise ImportError(
                "Could not import PointNet. Please clone the repository, check above\n"
            ) from e

        # PointNetFeat outputs 1024-dim global features
        channel = 6 if use_normals else 3
        self.pointnet_feat = PointNetFeat(
            global_feat=True,
            feature_transform=True,
            channel=channel
        )

        # Project from 1024-dim to desired feature dimension
        self.feature_projection = nn.Linear(1024, feature_dimension)

    def output_shape(self, input_shape=None):
        """
        Returns the output shape of this encoder.

        Returns:
            list: [feature_dimension]
        """
        return [self.feature_dimension]

    def forward(self, inputs):
        """
        Forward pass through PointNet.

        Args:
            inputs (torch.Tensor): Point cloud batch
                - If channel_first: (batch, 3, N) or (batch, 6, N)
                - If channel_last: (batch, N, 3) or (batch, N, 6)

        Returns:
            torch.Tensor: Encoded features of shape (batch, feature_dimension)
        """
        # Convert to channel-first format: (batch, 3/6, N)
        if not self.channel_first:
            x = inputs.transpose(1, 2).contiguous()
        else:
            x = inputs

        expected_channel = 6 if self.use_normals else 3
        if x.shape[1] != expected_channel:
            raise ValueError(
                f"Expected {expected_channel} channels, got {x.shape[1]}. "
                f"Input shape: {x.shape}, use_normals={self.use_normals}"
            )

        # PointNet feature extraction
        # Returns (global_features, trans_matrix, trans_feat_matrix)
        global_feat, trans, trans_feat = self.pointnet_feat(x)

        # Project 1024-dim features to desired dimension
        features = self.feature_projection(global_feat)

        return features


class PointNetPlusPlusEncoder(EncoderCore):
    """
    PointNet++ encoder for 3D point clouds.
    """

    def __init__(
        self,
        input_shape,
        feature_dimension=256,
        use_normals=False,
    ):
        """
        Args:
            input_shape (tuple): Expected shape (N, 3) or (3, N) for point clouds
            feature_dimension (int): Output feature dimension
            use_normals (bool): Whether input includes normals (6D instead of 3D)
        """
        super().__init__(input_shape=input_shape)

        self.feature_dimension = feature_dimension
        self.use_normals = use_normals

        # Determine input format
        expected_channel = 6 if use_normals else 3
        if input_shape[0] == expected_channel:
            self.channel_first = True
            self.num_points = input_shape[1]
        elif input_shape[-1] == expected_channel:
            self.channel_first = False
            self.num_points = input_shape[0]
        else:
            raise ValueError(
                f"Expected input_shape with {expected_channel} channels, got {input_shape}"
            )

        # Import PointNet++ modules
        try:
            from pointnet2_utils import PointNetSetAbstraction
        except ImportError as e:
            raise ImportError(
                "Could not import PointNet++. Please clone the repository:\n"
                "  cd /home/raymond/projects\n"
                "  git clone https://github.com/yanx27/Pointnet_Pointnet2_pytorch.git\n"
                f"Expected path: {POINTNET_PATH}\n"
                f"Error: {e}"
            ) from e

        # Build PointNet++ feature extractor (same as classification model)
        in_channel = 6 if use_normals else 3
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)

        # PointNet++ outputs 1024-dim features
        self.feature_projection = nn.Linear(1024, feature_dimension)

    def output_shape(self, input_shape=None):
        """Returns output shape."""
        return [self.feature_dimension]

    def forward(self, inputs):
        """
        Forward pass through PointNet++.

        Args:
            inputs (torch.Tensor): Point cloud batch
                - If channel_first: (batch, 3, N) or (batch, 6, N)
                - If channel_last: (batch, N, 3) or (batch, N, 6)

        Returns:
            torch.Tensor: Encoded features of shape (batch, feature_dimension)
        """
        # Convert to channel-first: (batch, 3/6, N)
        if not self.channel_first:
            x = inputs.transpose(1, 2).contiguous()
        else:
            x = inputs

        # Verify shape
        expected_channel = 6 if self.use_normals else 3
        if x.shape[1] != expected_channel:
            raise ValueError(
                f"Expected {expected_channel} channels, got {x.shape[1]}"
            )

        batch_size = x.shape[0]

        # Extract XYZ and normals
        xyz = x[:, :3, :].transpose(1, 2).contiguous()  # (batch, N, 3)
        if self.use_normals:
            norm = x[:, 3:, :]  # (batch, 3, N)
        else:
            norm = None

        # Forward through PointNet++ set abstraction layers
        l1_xyz, l1_points = self.sa1(xyz.transpose(1, 2).contiguous(), norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        # Extract global features (1024-dim)
        global_feat = l3_points.view(batch_size, 1024)

        # Project to desired dimension
        features = self.feature_projection(global_feat)

        return features
