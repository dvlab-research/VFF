import torch
import torch.nn as nn
import torch.nn.functional as F

from pcdet.models.model_utils.basic_block_2d import BasicBlock2D


class Conv2DCollapse(nn.Module):

    def __init__(self, model_cfg, grid_size):
        """
        Initializes 2D convolution collapse module
        Args:
            model_cfg: EasyDict, Model configuration
            grid_size: (X, Y, Z) Voxel grid size
        """
        super().__init__()
        self.model_cfg = model_cfg
        self.num_heights = grid_size[-1]
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.point_only = self.model_cfg.get('POINT_ONLY', False)
        self.block = BasicBlock2D(in_channels=self.num_bev_features * self.num_heights,
                                  out_channels=self.num_bev_features,
                                  **self.model_cfg.ARGS)

    def forward(self, batch_dict):
        """
        Collapses voxel features to BEV via concatenation and channel reduction
        Args:
            batch_dict:
                voxel_features: (B, C, Z, Y, X), Voxel feature representation
        Returns:
            batch_dict:
                spatial_features: (B, C, Y, X), BEV feature representation
        """
        if self.point_only:
            voxel_features = batch_dict['encoded_spconv_tensor'].dense()
            voxel_features = F.interpolate(voxel_features, size=(self.num_heights, *voxel_features.shape[-2:]), 
                                           mode='trilinear') # (B, C, Z', Y, X) -> (B, C, Z, Y, X)
        else:
            voxel_features = batch_dict["voxel_features"]
        bev_features = voxel_features.flatten(start_dim=1, end_dim=2)  # (B, C, Z, Y, X) -> (B, C*Z, Y, X)
        bev_features = self.block(bev_features)  # (B, C*Z, Y, X) -> (B, C, Y, X)
        batch_dict["spatial_features"] = bev_features
        return batch_dict
