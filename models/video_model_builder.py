# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.


"""Video models."""

import math
import os
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
#import slowfast.utils.logging as logging
from models.weight_init_helper import init_weights
from models.attention import MultiScaleBlock
from models.batchnorm_helper import get_norm
from models.common import TwoStreamFusion
from models.reversible_mvit import ReversibleMViT
from models.uniformerv2 import Uniformerv2
from models.video_swin import SwinTransformer3D
from models.vit import vit_base_patch16_224
from models.utils import (
    calc_mvit_feature_geometry,
    get_3d_sincos_pos_embed,
    round_width,
    validate_checkpoint_wrapper_import,
)

import models.head_helper as head_helper 
import models.operators as operators 
import models.resnet_helper as resnet_helper 
import models.stem_helper as stem_helper  # noqa
from models.build import MODEL_REGISTRY

try:
    from fairscale.nn.checkpoint import checkpoint_wrapper
except ImportError:
    checkpoint_wrapper = None


#logger = logging.get_logger(__name__)

# Number of blocks for different stages given the model depth.
_MODEL_STAGE_DEPTH = {18: (2, 2, 2, 2), 50: (3, 4, 6, 3), 101: (3, 4, 23, 3)}

# Basis of temporal kernel sizes for each of the stage.
_TEMPORAL_KERNEL_BASIS = {
    "2d": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[1]],  # res4 temporal kernel.
        [[1]],  # res5 temporal kernel.
    ],
    "c2d": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[1]],  # res4 temporal kernel.
        [[1]],  # res5 temporal kernel.
    ],
    "slow_c2d": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[1]],  # res4 temporal kernel.
        [[1]],  # res5 temporal kernel.
    ],
    "i3d": [
        [[5]],  # conv1 temporal kernel.
        [[3]],  # res2 temporal kernel.
        [[3, 1]],  # res3 temporal kernel.
        [[3, 1]],  # res4 temporal kernel.
        [[1, 3]],  # res5 temporal kernel.
    ],
    "slow_i3d": [
        [[5]],  # conv1 temporal kernel.
        [[3]],  # res2 temporal kernel.
        [[3, 1]],  # res3 temporal kernel.
        [[3, 1]],  # res4 temporal kernel.
        [[1, 3]],  # res5 temporal kernel.
    ],
    "slow": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[3]],  # res4 temporal kernel.
        [[3]],  # res5 temporal kernel.
    ],
    "fast": [
        [[5]],  # conv1 temporal kernel.
        [[3]],  # res2 temporal kernel.
        [[3]],  # res3 temporal kernel.
        [[3]],  # res4 temporal kernel.
        [[3]],  # res5 temporal kernel.
    ],
    "slowfast": [
        [[1], [5]],  # conv1 temporal kernel for slow and fast pathway.
        [[1], [3]],  # res2 temporal kernel for slow and fast pathway.
        [[1], [3]],  # res3 temporal kernel for slow and fast pathway.
        [[3], [3]],  # res4 temporal kernel for slow and fast pathway.
        [[3], [3]],  # res5 temporal kernel for slow and fast pathway.
    ],
    "x3d": [
        [[5]],  # conv1 temporal kernels.
        [[3]],  # res2 temporal kernels.
        [[3]],  # res3 temporal kernels.
        [[3]],  # res4 temporal kernels.
        [[3]],  # res5 temporal kernels.
    ],
}

_POOL1 = {
    "2d": [[1, 1, 1]],
    "c2d": [[2, 1, 1]],
    "slow_c2d": [[1, 1, 1]],
    "i3d": [[2, 1, 1]],
    "slow_i3d": [[1, 1, 1]],
    "slow": [[1, 1, 1]],
    "fast": [[1, 1, 1]],
    "slowfast": [[1, 1, 1], [1, 1, 1]],
    "x3d": [[1, 1, 1]],
}


class FuseFastToSlow(nn.Module):
    """
    Fuses the information from the Fast pathway to the Slow pathway. Given the
    tensors from Slow pathway and Fast pathway, fuse information from Fast to
    Slow, then return the fused tensors from Slow and Fast pathway in order.
    """

    def __init__(
        self,
        dim_in,
        fusion_conv_channel_ratio,
        fusion_kernel,
        alpha,
        eps=1e-5,
        bn_mmt=0.1,
        inplace_relu=True,
        norm_module=nn.BatchNorm3d,
    ):
        """
        Args:
            dim_in (int): the channel dimension of the input.
            fusion_conv_channel_ratio (int): channel ratio for the convolution
                used to fuse from Fast pathway to Slow pathway.
            fusion_kernel (int): kernel size of the convolution used to fuse
                from Fast pathway to Slow pathway.
            alpha (int): the frame rate ratio between the Fast and Slow pathway.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            inplace_relu (bool): if True, calculate the relu on the original
                input without allocating new memory.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
        """
        super(FuseFastToSlow, self).__init__()
        self.conv_f2s = nn.Conv3d(
            dim_in,
            dim_in * fusion_conv_channel_ratio,
            kernel_size=[fusion_kernel, 1, 1],
            stride=[alpha, 1, 1],
            padding=[fusion_kernel // 2, 0, 0],
            bias=False,
        )
        self.bn = norm_module(
            num_features=dim_in * fusion_conv_channel_ratio,
            eps=eps,
            momentum=bn_mmt,
        )
        self.relu = nn.ReLU(inplace_relu)

    def forward(self, x):
        x_s = x[0]
        x_f = x[1]
        fuse = self.conv_f2s(x_f)
        fuse = self.bn(fuse)
        fuse = self.relu(fuse)
        x_s_fuse = torch.cat([x_s, fuse], 1)
        return [x_s_fuse, x_f]


@MODEL_REGISTRY.register()
class SlowFast(nn.Module):
    """
    SlowFast model builder for SlowFast network.

    Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He.
    "SlowFast networks for video recognition."
    https://arxiv.org/pdf/1812.03982.pdf
    """

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(SlowFast, self).__init__()
        self.norm_module = get_norm(cfg)
        self.cfg = cfg
        self.enable_detection = cfg.DETECTION.ENABLE
        self.num_pathways = 2
        self._construct_network(cfg)
        init_weights(
            self,
            cfg.MODEL.FC_INIT_STD,
            cfg.RESNET.ZERO_INIT_FINAL_BN,
            cfg.RESNET.ZERO_INIT_FINAL_CONV,
        )

    def _construct_network(self, cfg):
        """
        Builds a SlowFast model. The first pathway is the Slow pathway and the
            second pathway is the Fast pathway.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        assert cfg.MODEL.ARCH in _POOL1.keys()
        pool_size = _POOL1[cfg.MODEL.ARCH]
        assert len({len(pool_size), self.num_pathways}) == 1
        assert cfg.RESNET.DEPTH in _MODEL_STAGE_DEPTH.keys()

        (d2, d3, d4, d5) = _MODEL_STAGE_DEPTH[cfg.RESNET.DEPTH]

        num_groups = cfg.RESNET.NUM_GROUPS
        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        dim_inner = num_groups * width_per_group
        out_dim_ratio = (
            cfg.SLOWFAST.BETA_INV // cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO
        )

        temp_kernel = _TEMPORAL_KERNEL_BASIS[cfg.MODEL.ARCH]

        self.s1 = stem_helper.VideoModelStem(
            dim_in=cfg.DATA.INPUT_CHANNEL_NUM,
            dim_out=[width_per_group, width_per_group // cfg.SLOWFAST.BETA_INV],
            kernel=[temp_kernel[0][0] + [7, 7], temp_kernel[0][1] + [7, 7]],
            stride=[[1, 2, 2]] * 2,
            padding=[
                [temp_kernel[0][0][0] // 2, 3, 3],
                [temp_kernel[0][1][0] // 2, 3, 3],
            ],
            norm_module=self.norm_module,
        )
        self.s1_fuse = FuseFastToSlow(
            width_per_group // cfg.SLOWFAST.BETA_INV,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
            norm_module=self.norm_module,
        )

        self.s2 = resnet_helper.ResStage(
            dim_in=[
                width_per_group + width_per_group // out_dim_ratio,
                width_per_group // cfg.SLOWFAST.BETA_INV,
            ],
            dim_out=[
                width_per_group * 4,
                width_per_group * 4 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_inner=[dim_inner, dim_inner // cfg.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[1],
            stride=cfg.RESNET.SPATIAL_STRIDES[0],
            num_blocks=[d2] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[0],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[0],
            nonlocal_group=cfg.NONLOCAL.GROUP[0],
            nonlocal_pool=cfg.NONLOCAL.POOL[0],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[0],
            norm_module=self.norm_module,
        )
        self.s2_fuse = FuseFastToSlow(
            width_per_group * 4 // cfg.SLOWFAST.BETA_INV,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
            norm_module=self.norm_module,
        )

        for pathway in range(self.num_pathways):
            pool = nn.MaxPool3d(
                kernel_size=pool_size[pathway],
                stride=pool_size[pathway],
                padding=[0, 0, 0],
            )
            self.add_module("pathway{}_pool".format(pathway), pool)

        self.s3 = resnet_helper.ResStage(
            dim_in=[
                width_per_group * 4 + width_per_group * 4 // out_dim_ratio,
                width_per_group * 4 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_out=[
                width_per_group * 8,
                width_per_group * 8 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_inner=[dim_inner * 2, dim_inner * 2 // cfg.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[2],
            stride=cfg.RESNET.SPATIAL_STRIDES[1],
            num_blocks=[d3] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[1],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[1],
            nonlocal_group=cfg.NONLOCAL.GROUP[1],
            nonlocal_pool=cfg.NONLOCAL.POOL[1],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[1],
            norm_module=self.norm_module,
        )
        self.s3_fuse = FuseFastToSlow(
            width_per_group * 8 // cfg.SLOWFAST.BETA_INV,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
            norm_module=self.norm_module,
        )

        self.s4 = resnet_helper.ResStage(
            dim_in=[
                width_per_group * 8 + width_per_group * 8 // out_dim_ratio,
                width_per_group * 8 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_out=[
                width_per_group * 16,
                width_per_group * 16 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_inner=[dim_inner * 4, dim_inner * 4 // cfg.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[3],
            stride=cfg.RESNET.SPATIAL_STRIDES[2],
            num_blocks=[d4] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[2],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[2],
            nonlocal_group=cfg.NONLOCAL.GROUP[2],
            nonlocal_pool=cfg.NONLOCAL.POOL[2],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[2],
            norm_module=self.norm_module,
        )
        self.s4_fuse = FuseFastToSlow(
            width_per_group * 16 // cfg.SLOWFAST.BETA_INV,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
            norm_module=self.norm_module,
        )

        self.s5 = resnet_helper.ResStage(
            dim_in=[
                width_per_group * 16 + width_per_group * 16 // out_dim_ratio,
                width_per_group * 16 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_out=[
                width_per_group * 32,
                width_per_group * 32 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_inner=[dim_inner * 8, dim_inner * 8 // cfg.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[4],
            stride=cfg.RESNET.SPATIAL_STRIDES[3],
            num_blocks=[d5] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[3],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[3],
            nonlocal_group=cfg.NONLOCAL.GROUP[3],
            nonlocal_pool=cfg.NONLOCAL.POOL[3],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[3],
            norm_module=self.norm_module,
        )

        if cfg.DETECTION.ENABLE:
            self.head = head_helper.ResNetRoIHead(
                dim_in=[
                    width_per_group * 32,
                    width_per_group * 32 // cfg.SLOWFAST.BETA_INV,
                ],
                num_classes=cfg.MODEL.NUM_CLASSES,
                pool_size=[
                    [
                        cfg.DATA.NUM_FRAMES
                        // cfg.SLOWFAST.ALPHA
                        // pool_size[0][0],
                        1,
                        1,
                    ],
                    [cfg.DATA.NUM_FRAMES // pool_size[1][0], 1, 1],
                ],
                resolution=[[cfg.DETECTION.ROI_XFORM_RESOLUTION] * 2] * 2,
                scale_factor=[cfg.DETECTION.SPATIAL_SCALE_FACTOR] * 2,
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
                aligned=cfg.DETECTION.ALIGNED,
                detach_final_fc=cfg.MODEL.DETACH_FINAL_FC,
            )
        else:
            self.head = head_helper.ResNetBasicHead(
                dim_in=[
                    width_per_group * 32,
                    width_per_group * 32 // cfg.SLOWFAST.BETA_INV,
                ],
                num_classes=cfg.MODEL.NUM_CLASSES,
                pool_size=[None, None]
                if cfg.MULTIGRID.SHORT_CYCLE
                or cfg.MODEL.MODEL_NAME == "ContrastiveModel"
                else [
                    [
                        cfg.DATA.NUM_FRAMES
                        // cfg.SLOWFAST.ALPHA
                        // pool_size[0][0],
                        cfg.DATA.TRAIN_CROP_SIZE // 32 // pool_size[0][1],
                        cfg.DATA.TRAIN_CROP_SIZE // 32 // pool_size[0][2],
                    ],
                    [
                        cfg.DATA.NUM_FRAMES // pool_size[1][0],
                        cfg.DATA.TRAIN_CROP_SIZE // 32 // pool_size[1][1],
                        cfg.DATA.TRAIN_CROP_SIZE // 32 // pool_size[1][2],
                    ],
                ],  # None for AdaptiveAvgPool3d((1, 1, 1))
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
                detach_final_fc=cfg.MODEL.DETACH_FINAL_FC,
                cfg=cfg,
            )

    def forward(self, x, bboxes=None):
        x = x[:]  # avoid pass by reference
        x = self.s1(x)
        x = self.s1_fuse(x)
        x = self.s2(x)
        x = self.s2_fuse(x)
        for pathway in range(self.num_pathways):
            pool = getattr(self, "pathway{}_pool".format(pathway))
            x[pathway] = pool(x[pathway])
        x = self.s3(x)
        x = self.s3_fuse(x)
        x = self.s4(x)
        x = self.s4_fuse(x)
        x = self.s5(x)
        if self.enable_detection:
            x = self.head(x, bboxes)
        else:
            x = self.head(x)
        return x


@MODEL_REGISTRY.register()
class ResNet(nn.Module):
    """
    ResNet model builder. It builds a ResNet like network backbone without
    lateral connection (C2D, I3D, Slow).

    Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He.
    "SlowFast networks for video recognition."
    https://arxiv.org/pdf/1812.03982.pdf

    Xiaolong Wang, Ross Girshick, Abhinav Gupta, and Kaiming He.
    "Non-local neural networks."
    https://arxiv.org/pdf/1711.07971.pdf
    """

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(ResNet, self).__init__()
        self.norm_module = get_norm(cfg)
        self.enable_detection = cfg.DETECTION.ENABLE
        self.num_pathways = 1
        self._construct_network(cfg)
        init_weights(
            self,
            cfg.MODEL.FC_INIT_STD,
            cfg.RESNET.ZERO_INIT_FINAL_BN,
            cfg.RESNET.ZERO_INIT_FINAL_CONV,
        )

    def _construct_network(self, cfg):
        """
        Builds a single pathway ResNet model.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        assert cfg.MODEL.ARCH in _POOL1.keys()
        pool_size = _POOL1[cfg.MODEL.ARCH]
        assert len({len(pool_size), self.num_pathways}) == 1
        assert cfg.RESNET.DEPTH in _MODEL_STAGE_DEPTH.keys()
        self.cfg = cfg

        (d2, d3, d4, d5) = _MODEL_STAGE_DEPTH[cfg.RESNET.DEPTH]

        num_groups = cfg.RESNET.NUM_GROUPS
        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        dim_inner = num_groups * width_per_group

        temp_kernel = _TEMPORAL_KERNEL_BASIS[cfg.MODEL.ARCH]

        s1 = stem_helper.VideoModelStem(
            dim_in=cfg.DATA.INPUT_CHANNEL_NUM,
            dim_out=[width_per_group],
            kernel=[temp_kernel[0][0] + [7, 7]],
            stride=[[1, 2, 2]],
            padding=[[temp_kernel[0][0][0] // 2, 3, 3]],
            norm_module=self.norm_module,
        )

        s2 = resnet_helper.ResStage(
            dim_in=[width_per_group],
            dim_out=[width_per_group * 4],
            dim_inner=[dim_inner],
            temp_kernel_sizes=temp_kernel[1],
            stride=cfg.RESNET.SPATIAL_STRIDES[0],
            num_blocks=[d2],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[0],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[0],
            nonlocal_group=cfg.NONLOCAL.GROUP[0],
            nonlocal_pool=cfg.NONLOCAL.POOL[0],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[0],
            norm_module=self.norm_module,
        )

        # Based on profiling data of activation size, s1 and s2 have the activation sizes
        # that are 4X larger than the second largest. Therefore, checkpointing them gives
        # best memory savings. Further tuning is possible for better memory saving and tradeoffs
        # with recomputing FLOPs.
        if cfg.MODEL.ACT_CHECKPOINT:
            validate_checkpoint_wrapper_import(checkpoint_wrapper)
            self.s1 = checkpoint_wrapper(s1)
            self.s2 = checkpoint_wrapper(s2)
        else:
            self.s1 = s1
            self.s2 = s2

        for pathway in range(self.num_pathways):
            pool = nn.MaxPool3d(
                kernel_size=pool_size[pathway],
                stride=pool_size[pathway],
                padding=[0, 0, 0],
            )
            self.add_module("pathway{}_pool".format(pathway), pool)

        self.s3 = resnet_helper.ResStage(
            dim_in=[width_per_group * 4],
            dim_out=[width_per_group * 8],
            dim_inner=[dim_inner * 2],
            temp_kernel_sizes=temp_kernel[2],
            stride=cfg.RESNET.SPATIAL_STRIDES[1],
            num_blocks=[d3],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[1],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[1],
            nonlocal_group=cfg.NONLOCAL.GROUP[1],
            nonlocal_pool=cfg.NONLOCAL.POOL[1],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[1],
            norm_module=self.norm_module,
        )

        self.s4 = resnet_helper.ResStage(
            dim_in=[width_per_group * 8],
            dim_out=[width_per_group * 16],
            dim_inner=[dim_inner * 4],
            temp_kernel_sizes=temp_kernel[3],
            stride=cfg.RESNET.SPATIAL_STRIDES[2],
            num_blocks=[d4],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[2],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[2],
            nonlocal_group=cfg.NONLOCAL.GROUP[2],
            nonlocal_pool=cfg.NONLOCAL.POOL[2],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[2],
            norm_module=self.norm_module,
        )

        self.s5 = resnet_helper.ResStage(
            dim_in=[width_per_group * 16],
            dim_out=[width_per_group * 32],
            dim_inner=[dim_inner * 8],
            temp_kernel_sizes=temp_kernel[4],
            stride=cfg.RESNET.SPATIAL_STRIDES[3],
            num_blocks=[d5],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[3],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[3],
            nonlocal_group=cfg.NONLOCAL.GROUP[3],
            nonlocal_pool=cfg.NONLOCAL.POOL[3],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[3],
            norm_module=self.norm_module,
        )

        if self.enable_detection:
            self.head = head_helper.ResNetRoIHead(
                dim_in=[width_per_group * 32],
                num_classes=cfg.MODEL.NUM_CLASSES,
                pool_size=[[cfg.DATA.NUM_FRAMES // pool_size[0][0], 1, 1]],
                resolution=[[cfg.DETECTION.ROI_XFORM_RESOLUTION] * 2],
                scale_factor=[cfg.DETECTION.SPATIAL_SCALE_FACTOR],
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
                aligned=cfg.DETECTION.ALIGNED,
                detach_final_fc=cfg.MODEL.DETACH_FINAL_FC,
            )
        else:
            self.head = head_helper.ResNetBasicHead(
                dim_in=[width_per_group * 32],
                num_classes=cfg.MODEL.NUM_CLASSES,
                pool_size=[None]
                if cfg.MULTIGRID.SHORT_CYCLE
                or cfg.MODEL.MODEL_NAME == "ContrastiveModel"
                else [
                    [
                        cfg.DATA.NUM_FRAMES // pool_size[0][0],
                        cfg.DATA.TRAIN_CROP_SIZE // 32 // pool_size[0][1],
                        cfg.DATA.TRAIN_CROP_SIZE // 32 // pool_size[0][2],
                    ]
                ],  # None for AdaptiveAvgPool3d((1, 1, 1))
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
                detach_final_fc=cfg.MODEL.DETACH_FINAL_FC,
                cfg=cfg,
            )

    def forward(self, x, bboxes=None):
        x = x[:]  # avoid pass by reference
        x = self.s1(x)
        x = self.s2(x)
        y = []  # Don't modify x list in place due to activation checkpoint.
        for pathway in range(self.num_pathways):
            pool = getattr(self, "pathway{}_pool".format(pathway))
            y.append(pool(x[pathway]))
        x = self.s3(y)
        x = self.s4(x)
        x = self.s5(x)
        if self.enable_detection:
            x = self.head(x, bboxes)
        else:
            x = self.head(x)
        return x


@MODEL_REGISTRY.register()
class X3D(nn.Module):
    """
    X3D model builder. It builds a X3D network backbone, which is a ResNet.

    Christoph Feichtenhofer.
    "X3D: Expanding Architectures for Efficient Video Recognition."
    https://arxiv.org/abs/2004.04730
    """

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(X3D, self).__init__()
        self.norm_module = get_norm(cfg)
        self.enable_detection = cfg.DETECTION.ENABLE
        self.num_pathways = 1

        exp_stage = 2.0
        self.dim_c1 = cfg.X3D.DIM_C1

        self.dim_res2 = (
            round_width(self.dim_c1, exp_stage, divisor=8)
            if cfg.X3D.SCALE_RES2
            else self.dim_c1
        )
        self.dim_res3 = round_width(self.dim_res2, exp_stage, divisor=8)
        self.dim_res4 = round_width(self.dim_res3, exp_stage, divisor=8)
        self.dim_res5 = round_width(self.dim_res4, exp_stage, divisor=8)

        self.block_basis = [
            # blocks, c, stride
            [1, self.dim_res2, 2],
            [2, self.dim_res3, 2],
            [5, self.dim_res4, 2],
            [3, self.dim_res5, 2],
        ]
        self._construct_network(cfg)
        init_weights(
            self, cfg.MODEL.FC_INIT_STD, cfg.RESNET.ZERO_INIT_FINAL_BN
        )

    def _round_repeats(self, repeats, multiplier):
        """Round number of layers based on depth multiplier."""
        multiplier = multiplier
        if not multiplier:
            return repeats
        return int(math.ceil(multiplier * repeats))

    def _construct_network(self, cfg):
        """
        Builds a single pathway X3D model.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        assert cfg.MODEL.ARCH in _POOL1.keys()
        assert cfg.RESNET.DEPTH in _MODEL_STAGE_DEPTH.keys()

        (d2, d3, d4, d5) = _MODEL_STAGE_DEPTH[cfg.RESNET.DEPTH]

        num_groups = cfg.RESNET.NUM_GROUPS
        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        dim_inner = num_groups * width_per_group

        w_mul = cfg.X3D.WIDTH_FACTOR
        d_mul = cfg.X3D.DEPTH_FACTOR
        dim_res1 = round_width(self.dim_c1, w_mul)

        temp_kernel = _TEMPORAL_KERNEL_BASIS[cfg.MODEL.ARCH]

        self.s1 = stem_helper.VideoModelStem(
            dim_in=cfg.DATA.INPUT_CHANNEL_NUM,
            dim_out=[dim_res1],
            kernel=[temp_kernel[0][0] + [3, 3]],
            stride=[[1, 2, 2]],
            padding=[[temp_kernel[0][0][0] // 2, 1, 1]],
            norm_module=self.norm_module,
            stem_func_name="x3d_stem",
        )

        # blob_in = s1
        dim_in = dim_res1
        for stage, block in enumerate(self.block_basis):
            dim_out = round_width(block[1], w_mul)
            dim_inner = int(cfg.X3D.BOTTLENECK_FACTOR * dim_out)

            n_rep = self._round_repeats(block[0], d_mul)
            prefix = "s{}".format(
                stage + 2
            )  # start w res2 to follow convention

            s = resnet_helper.ResStage(
                dim_in=[dim_in],
                dim_out=[dim_out],
                dim_inner=[dim_inner],
                temp_kernel_sizes=temp_kernel[1],
                stride=[block[2]],
                num_blocks=[n_rep],
                num_groups=[dim_inner]
                if cfg.X3D.CHANNELWISE_3x3x3
                else [num_groups],
                num_block_temp_kernel=[n_rep],
                nonlocal_inds=cfg.NONLOCAL.LOCATION[0],
                nonlocal_group=cfg.NONLOCAL.GROUP[0],
                nonlocal_pool=cfg.NONLOCAL.POOL[0],
                instantiation=cfg.NONLOCAL.INSTANTIATION,
                trans_func_name=cfg.RESNET.TRANS_FUNC,
                stride_1x1=cfg.RESNET.STRIDE_1X1,
                norm_module=self.norm_module,
                dilation=cfg.RESNET.SPATIAL_DILATIONS[stage],
                drop_connect_rate=cfg.MODEL.DROPCONNECT_RATE
                * (stage + 2)
                / (len(self.block_basis) + 1),
            )
            dim_in = dim_out
            self.add_module(prefix, s)

        if self.enable_detection:
            NotImplementedError
        else:
            spat_sz = int(math.ceil(cfg.DATA.TRAIN_CROP_SIZE / 32.0))
            self.head = head_helper.X3DHead(
                dim_in=dim_out,
                dim_inner=dim_inner,
                dim_out=cfg.X3D.DIM_C5,
                num_classes=cfg.MODEL.NUM_CLASSES,
                pool_size=[cfg.DATA.NUM_FRAMES, spat_sz, spat_sz],
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
                bn_lin5_on=cfg.X3D.BN_LIN5,
            )

    def forward(self, x, bboxes=None):
        for module in self.children():
            x = module(x)
        return x



@MODEL_REGISTRY.register()
class MViT(nn.Module):
    """
    Model builder for MViTv1 and MViTv2.

    "MViTv2: Improved Multiscale Vision Transformers for Classification and Detection"
    Yanghao Li, Chao-Yuan Wu, Haoqi Fan, Karttikeya Mangalam, Bo Xiong, Jitendra Malik, Christoph Feichtenhofer
    https://arxiv.org/abs/2112.01526
    "Multiscale Vision Transformers"
    Haoqi Fan, Bo Xiong, Karttikeya Mangalam, Yanghao Li, Zhicheng Yan, Jitendra Malik, Christoph Feichtenhofer
    https://arxiv.org/abs/2104.11227
    """

    def __init__(self, cfg):
        super().__init__()
        # Get parameters.
        assert cfg.DATA.TRAIN_CROP_SIZE == cfg.DATA.TEST_CROP_SIZE
        self.cfg = cfg
        pool_first = cfg.MVIT.POOL_FIRST
        # Prepare input.
        spatial_size = cfg.DATA.TRAIN_CROP_SIZE
        temporal_size = cfg.DATA.NUM_FRAMES
        in_chans = cfg.DATA.INPUT_CHANNEL_NUM[0]
        self.use_2d_patch = cfg.MVIT.PATCH_2D
        self.enable_detection = cfg.DETECTION.ENABLE
        self.enable_rev = cfg.MVIT.REV.ENABLE
        self.patch_stride = cfg.MVIT.PATCH_STRIDE
        if self.use_2d_patch:
            self.patch_stride = [1] + self.patch_stride
        self.T = cfg.DATA.NUM_FRAMES // self.patch_stride[0]
        self.H = cfg.DATA.TRAIN_CROP_SIZE // self.patch_stride[1]
        self.W = cfg.DATA.TRAIN_CROP_SIZE // self.patch_stride[2]
        # Prepare output.
        num_classes = cfg.MODEL.NUM_CLASSES
        embed_dim = cfg.MVIT.EMBED_DIM
        # Prepare backbone
        num_heads = cfg.MVIT.NUM_HEADS
        mlp_ratio = cfg.MVIT.MLP_RATIO
        qkv_bias = cfg.MVIT.QKV_BIAS
        self.drop_rate = cfg.MVIT.DROPOUT_RATE
        depth = cfg.MVIT.DEPTH
        drop_path_rate = cfg.MVIT.DROPPATH_RATE
        layer_scale_init_value = cfg.MVIT.LAYER_SCALE_INIT_VALUE
        head_init_scale = cfg.MVIT.HEAD_INIT_SCALE
        mode = cfg.MVIT.MODE
        self.cls_embed_on = cfg.MVIT.CLS_EMBED_ON
        self.use_mean_pooling = cfg.MVIT.USE_MEAN_POOLING
        # Params for positional embedding
        self.use_abs_pos = cfg.MVIT.USE_ABS_POS
        self.use_fixed_sincos_pos = cfg.MVIT.USE_FIXED_SINCOS_POS
        self.sep_pos_embed = cfg.MVIT.SEP_POS_EMBED
        self.rel_pos_spatial = cfg.MVIT.REL_POS_SPATIAL
        self.rel_pos_temporal = cfg.MVIT.REL_POS_TEMPORAL
        if cfg.MVIT.NORM == "layernorm":
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        else:
            raise NotImplementedError("Only supports layernorm.")
        self.num_classes = num_classes
        self.patch_embed = stem_helper.PatchEmbed(
            dim_in=in_chans,
            dim_out=embed_dim,
            kernel=cfg.MVIT.PATCH_KERNEL,
            stride=cfg.MVIT.PATCH_STRIDE,
            padding=cfg.MVIT.PATCH_PADDING,
            conv_2d=self.use_2d_patch,
        )

        if cfg.MODEL.ACT_CHECKPOINT:
            self.patch_embed = checkpoint_wrapper(self.patch_embed)
        self.input_dims = [temporal_size, spatial_size, spatial_size]
        assert self.input_dims[1] == self.input_dims[2]
        self.patch_dims = [
            self.input_dims[i] // self.patch_stride[i]
            for i in range(len(self.input_dims))
        ]
        num_patches = math.prod(self.patch_dims)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule

        if self.cls_embed_on:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            pos_embed_dim = num_patches + 1
        else:
            pos_embed_dim = num_patches

        if self.use_abs_pos:
            if self.sep_pos_embed:
                self.pos_embed_spatial = nn.Parameter(
                    torch.zeros(
                        1, self.patch_dims[1] * self.patch_dims[2], embed_dim
                    )
                )
                self.pos_embed_temporal = nn.Parameter(
                    torch.zeros(1, self.patch_dims[0], embed_dim)
                )
                if self.cls_embed_on:
                    self.pos_embed_class = nn.Parameter(
                        torch.zeros(1, 1, embed_dim)
                    )
            else:
                self.pos_embed = nn.Parameter(
                    torch.zeros(
                        1,
                        pos_embed_dim,
                        embed_dim,
                    ),
                    requires_grad=not self.use_fixed_sincos_pos,
                )

        if self.drop_rate > 0.0:
            self.pos_drop = nn.Dropout(p=self.drop_rate)

        dim_mul, head_mul = torch.ones(depth + 1), torch.ones(depth + 1)
        for i in range(len(cfg.MVIT.DIM_MUL)):
            dim_mul[cfg.MVIT.DIM_MUL[i][0]] = cfg.MVIT.DIM_MUL[i][1]
        for i in range(len(cfg.MVIT.HEAD_MUL)):
            head_mul[cfg.MVIT.HEAD_MUL[i][0]] = cfg.MVIT.HEAD_MUL[i][1]

        pool_q = [[] for i in range(cfg.MVIT.DEPTH)]
        pool_kv = [[] for i in range(cfg.MVIT.DEPTH)]
        stride_q = [[] for i in range(cfg.MVIT.DEPTH)]
        stride_kv = [[] for i in range(cfg.MVIT.DEPTH)]

        for i in range(len(cfg.MVIT.POOL_Q_STRIDE)):
            stride_q[cfg.MVIT.POOL_Q_STRIDE[i][0]] = cfg.MVIT.POOL_Q_STRIDE[i][
                1:
            ]
            if cfg.MVIT.POOL_KVQ_KERNEL is not None:
                pool_q[cfg.MVIT.POOL_Q_STRIDE[i][0]] = cfg.MVIT.POOL_KVQ_KERNEL
            else:
                pool_q[cfg.MVIT.POOL_Q_STRIDE[i][0]] = [
                    s + 1 if s > 1 else s for s in cfg.MVIT.POOL_Q_STRIDE[i][1:]
                ]

        # If POOL_KV_STRIDE_ADAPTIVE is not None, initialize POOL_KV_STRIDE.
        if cfg.MVIT.POOL_KV_STRIDE_ADAPTIVE is not None:
            _stride_kv = cfg.MVIT.POOL_KV_STRIDE_ADAPTIVE
            cfg.MVIT.POOL_KV_STRIDE = []
            for i in range(cfg.MVIT.DEPTH):
                if len(stride_q[i]) > 0:
                    _stride_kv = [
                        max(_stride_kv[d] // stride_q[i][d], 1)
                        for d in range(len(_stride_kv))
                    ]
                cfg.MVIT.POOL_KV_STRIDE.append([i] + _stride_kv)

        for i in range(len(cfg.MVIT.POOL_KV_STRIDE)):
            stride_kv[cfg.MVIT.POOL_KV_STRIDE[i][0]] = cfg.MVIT.POOL_KV_STRIDE[
                i
            ][1:]
            if cfg.MVIT.POOL_KVQ_KERNEL is not None:
                pool_kv[
                    cfg.MVIT.POOL_KV_STRIDE[i][0]
                ] = cfg.MVIT.POOL_KVQ_KERNEL
            else:
                pool_kv[cfg.MVIT.POOL_KV_STRIDE[i][0]] = [
                    s + 1 if s > 1 else s
                    for s in cfg.MVIT.POOL_KV_STRIDE[i][1:]
                ]

        self.pool_q = pool_q
        self.pool_kv = pool_kv
        self.stride_q = stride_q
        self.stride_kv = stride_kv

        self.norm_stem = norm_layer(embed_dim) if cfg.MVIT.NORM_STEM else None

        input_size = self.patch_dims

        if self.enable_rev:

            # rev does not allow cls token
            assert not self.cls_embed_on

            self.rev_backbone = ReversibleMViT(cfg, self)

            embed_dim = round_width(
                embed_dim, dim_mul.prod(), divisor=num_heads
            )

            self.fuse = TwoStreamFusion(
                cfg.MVIT.REV.RESPATH_FUSE, dim=2 * embed_dim
            )

            if "concat" in self.cfg.MVIT.REV.RESPATH_FUSE:
                self.norm = norm_layer(2 * embed_dim)
            else:
                self.norm = norm_layer(embed_dim)

        else:

            self.blocks = nn.ModuleList()

            for i in range(depth):
                num_heads = round_width(num_heads, head_mul[i])
                if cfg.MVIT.DIM_MUL_IN_ATT:
                    dim_out = round_width(
                        embed_dim,
                        dim_mul[i],
                        divisor=round_width(num_heads, head_mul[i]),
                    )
                else:
                    dim_out = round_width(
                        embed_dim,
                        dim_mul[i + 1],
                        divisor=round_width(num_heads, head_mul[i + 1]),
                    )
                attention_block = MultiScaleBlock(
                    dim=embed_dim,
                    dim_out=dim_out,
                    num_heads=num_heads,
                    input_size=input_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop_rate=self.drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    kernel_q=pool_q[i] if len(pool_q) > i else [],
                    kernel_kv=pool_kv[i] if len(pool_kv) > i else [],
                    stride_q=stride_q[i] if len(stride_q) > i else [],
                    stride_kv=stride_kv[i] if len(stride_kv) > i else [],
                    mode=mode,
                    has_cls_embed=self.cls_embed_on,
                    pool_first=pool_first,
                    rel_pos_spatial=self.rel_pos_spatial,
                    rel_pos_temporal=self.rel_pos_temporal,
                    rel_pos_zero_init=cfg.MVIT.REL_POS_ZERO_INIT,
                    residual_pooling=cfg.MVIT.RESIDUAL_POOLING,
                    dim_mul_in_att=cfg.MVIT.DIM_MUL_IN_ATT,
                    separate_qkv=cfg.MVIT.SEPARATE_QKV,
                )

                if cfg.MODEL.ACT_CHECKPOINT:
                    attention_block = checkpoint_wrapper(attention_block)
                self.blocks.append(attention_block)
                if len(stride_q[i]) > 0:
                    input_size = [
                        size // stride
                        for size, stride in zip(input_size, stride_q[i])
                    ]

                embed_dim = dim_out

            self.norm = norm_layer(embed_dim)

        if self.enable_detection:
            self.head = head_helper.ResNetRoIHead(
                dim_in=[embed_dim],
                num_classes=num_classes,
                pool_size=[[temporal_size // self.patch_stride[0], 1, 1]],
                resolution=[[cfg.DETECTION.ROI_XFORM_RESOLUTION] * 2],
                scale_factor=[cfg.DETECTION.SPATIAL_SCALE_FACTOR],
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
                aligned=cfg.DETECTION.ALIGNED,
            )
        else:
            self.head = head_helper.TransformerBasicHead(
                2 * embed_dim
                if ("concat" in cfg.MVIT.REV.RESPATH_FUSE and self.enable_rev)
                else embed_dim,
                num_classes,
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
                cfg=cfg,
            )
        if self.use_abs_pos:
            if self.sep_pos_embed:
                trunc_normal_(self.pos_embed_spatial, std=0.02)
                trunc_normal_(self.pos_embed_temporal, std=0.02)
                if self.cls_embed_on:
                    trunc_normal_(self.pos_embed_class, std=0.02)
            else:
                trunc_normal_(self.pos_embed, std=0.02)
                if self.use_fixed_sincos_pos:
                    pos_embed = get_3d_sincos_pos_embed(
                        self.pos_embed.shape[-1],
                        self.H,
                        self.T,
                        cls_token=self.cls_embed_on,
                    )
                    self.pos_embed.data.copy_(
                        torch.from_numpy(pos_embed).float().unsqueeze(0)
                    )

        if self.cls_embed_on:
            trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

        self.head.projection.weight.data.mul_(head_init_scale)
        self.head.projection.bias.data.mul_(head_init_scale)

        self.feat_size, self.feat_stride = calc_mvit_feature_geometry(cfg)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0.02)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        names = []
        if self.cfg.MVIT.ZERO_DECAY_POS_CLS:
            if self.use_abs_pos:
                if self.sep_pos_embed:
                    names.extend(
                        [
                            "pos_embed_spatial",
                            "pos_embed_temporal",
                            "pos_embed_class",
                        ]
                    )
                else:
                    names.append("pos_embed")
            if self.rel_pos_spatial:
                names.extend(["rel_pos_h", "rel_pos_w", "rel_pos_hw"])
            if self.rel_pos_temporal:
                names.extend(["rel_pos_t"])
            if self.cls_embed_on:
                names.append("cls_token")

        return names

    def _get_pos_embed(self, pos_embed, bcthw):

        if len(bcthw) == 4:
            t, h, w = 1, bcthw[-2], bcthw[-1]
        else:
            t, h, w = bcthw[-3], bcthw[-2], bcthw[-1]
        if self.cls_embed_on:
            cls_pos_embed = pos_embed[:, 0:1, :]
            pos_embed = pos_embed[:, 1:]
        txy_num = pos_embed.shape[1]
        p_t, p_h, p_w = self.patch_dims
        assert p_t * p_h * p_w == txy_num

        if (p_t, p_h, p_w) != (t, h, w):
            new_pos_embed = F.interpolate(
                pos_embed[:, :, :]
                .reshape(1, p_t, p_h, p_w, -1)
                .permute(0, 4, 1, 2, 3),
                size=(t, h, w),
                mode="trilinear",
            )
            pos_embed = new_pos_embed.reshape(1, -1, t * h * w).permute(0, 2, 1)

        if self.cls_embed_on:
            pos_embed = torch.cat((cls_pos_embed, pos_embed), dim=1)

        return pos_embed

    def _forward_reversible(self, x):
        """
        Reversible specific code for forward computation.
        """
        # rev does not support cls token or detection
        assert not self.cls_embed_on
        assert not self.enable_detection

        x = self.rev_backbone(x)

        if self.use_mean_pooling:
            x = self.fuse(x)
            x = x.mean(1)
            x = self.norm(x)
        else:
            x = self.norm(x)
            x = self.fuse(x)
            x = x.mean(1)

        x = self.head(x)

        return x

    def forward(self, x, bboxes=None, return_attn=False):
        x = x
        x, bcthw = self.patch_embed(x)
        bcthw = list(bcthw)
        if len(bcthw) == 4:  # Fix bcthw in case of 4D tensor
            bcthw.insert(2, torch.tensor(self.T))
        T, H, W = bcthw[-3], bcthw[-2], bcthw[-1]
        assert len(bcthw) == 5 and (T, H, W) == (self.T, self.H, self.W), bcthw
        B, N, C = x.shape

        s = 1 if self.cls_embed_on else 0
        if self.use_fixed_sincos_pos:
            x += self.pos_embed[:, s:, :]  # s: on/off cls token

        if self.cls_embed_on:
            cls_tokens = self.cls_token.expand(
                B, -1, -1
            )  # stole cls_tokens impl from Phil Wang, thanks
            if self.use_fixed_sincos_pos:
                cls_tokens = cls_tokens + self.pos_embed[:, :s, :]
            
            x = torch.cat((cls_tokens, x), dim=1)

        if self.use_abs_pos:
            if self.sep_pos_embed:
                pos_embed = self.pos_embed_spatial.repeat(
                    1, self.patch_dims[0], 1
                ) + torch.repeat_interleave(
                    self.pos_embed_temporal,
                    self.patch_dims[1] * self.patch_dims[2],
                    dim=1,
                )
                if self.cls_embed_on:
                    pos_embed = torch.cat([self.pos_embed_class, pos_embed], 1)
                x += self._get_pos_embed(pos_embed, bcthw)
            else:
                x += self._get_pos_embed(self.pos_embed, bcthw)

        if self.drop_rate:
            x = self.pos_drop(x)

        if self.norm_stem:
            x = self.norm_stem(x)

        thw = [T, H, W]

        if self.enable_rev:
            x = self._forward_reversible(x)

        else:
            for blk in self.blocks:
                x, thw = blk(x, thw)

            if self.enable_detection:
                assert not self.enable_rev

                x = self.norm(x)
                if self.cls_embed_on:
                    x = x[:, 1:]

                B, _, C = x.shape
                x = x.transpose(1, 2).reshape(B, C, thw[0], thw[1], thw[2])

                x = self.head([x], bboxes)

            else:
                if self.use_mean_pooling:
                    if self.cls_embed_on:
                        x = x[:, 1:]
                    x = x.mean(1)
                    x = self.norm(x)
                elif self.cls_embed_on:
                    x = self.norm(x)
                    x = x[:, 0]
                else:  # this is default, [norm->mean]
                    x = self.norm(x)
                    x = x.mean(1)
                x = self.head(x)

        return x


def load_config(args, path_to_config=None):
    from models.configs.defaults import get_cfg
    import models.checkpoint as cu

    """
    Given the arguemnts, load and initialize the configs.
    Args:
        args (argument): arguments includes `shard_id`, `num_shards`,
            `init_method`, `cfg_file`, and `opts`.
    """
    # Setup cfg.
    cfg = get_cfg()
    # Load config from cfg.
    if path_to_config is not None:
        cfg.merge_from_file(path_to_config)
    # Load config from command line, overwrite config from opts.
    if args is not None:
        cfg.merge_from_list(args.opts)

    # Inherit parameters from args.
    if hasattr(args, "num_shards") and hasattr(args, "shard_id"):
        cfg.NUM_SHARDS = args.num_shards
        cfg.SHARD_ID = args.shard_id
    if hasattr(args, "rng_seed"):
        cfg.RNG_SEED = args.rng_seed
    if hasattr(args, "output_dir"):
        cfg.OUTPUT_DIR = args.output_dir

    # Create the checkpoint dir.
    cu.make_checkpoint_dir(cfg.OUTPUT_DIR)
    return cfg

def check_weights_url(parent_dir,url):
    import wget
    if not os.path.exists(parent_dir): # check if weights folder exists
        os.makedirs(parent_dir) # if not, create it
    name = url.split('/')[-1] # get weights filename from URL
    filepath = os.path.join(parent_dir,name) # create complete filepath
    if not os.path.isfile(filepath): # check if weights have been previously hownloaded
        print('Downloading {} and saving it to {}'.format(url, filepath))
        response = wget.download(url, filepath)
    return filepath
        
    

def x3d_xs(num_frames=None,frame_size=None):
    from models.configs.defaults import assert_and_infer_cfg
    from models.checkpoint import load_checkpoint
    weights = 'https://dl.fbaipublicfiles.com/pyslowfast/x3d_models/x3d_xs.pyth'
    path = 'checkpoints'
    weights = check_weights_url(path,weights)
    
    path_to_config = 'models/configs/kinetics/x3d_xs.yaml'
    cfg = load_config(None, path_to_config)
    if num_frames is not None:
        cfg.DATA.NUM_FRAMES = num_frames
    if frame_size is not None:
        cfg.TRAIN_CROP_SIZE= frame_size
        cfg.TEST_CROP_SIZE: frame_size
    _, net = load_checkpoint(path_to_checkpoint=weights,model=X3D(cfg),data_parallel=False)
    return cfg, net
    

def x3d_s(num_frames=None,frame_size=None):
    from models.configs.defaults import assert_and_infer_cfg
    from models.checkpoint import load_checkpoint
    weights = 'https://dl.fbaipublicfiles.com/pyslowfast/x3d_models/x3d_s.pyth'
    path = 'checkpoints'
    weights = check_weights_url(path,weights)
    
    path_to_config = 'models/configs/kinetics/x3d_s.yaml'
    cfg = load_config(None, path_to_config)
    if num_frames is not None:
        cfg.DATA.NUM_FRAMES = num_frames
    if frame_size is not None:
        cfg.TRAIN_CROP_SIZE= frame_size
        cfg.TEST_CROP_SIZE: frame_size
    _, net = load_checkpoint(path_to_checkpoint=weights,model=X3D(cfg),data_parallel=False)
    return cfg, net


def x3d_m(num_frames=None,frame_size=None):
    from models.configs.defaults import assert_and_infer_cfg
    from models.checkpoint import load_checkpoint
    weights = 'https://dl.fbaipublicfiles.com/pyslowfast/x3d_models/x3d_m.pyth'
    path = 'checkpoints'
    weights = check_weights_url(path,weights)
    
    path_to_config = 'models/configs/kinetics/x3d_m.yaml'
    cfg = load_config(None, path_to_config)
    if num_frames is not None:
        cfg.DATA.NUM_FRAMES = num_frames
    if frame_size is not None:
        cfg.TRAIN_CROP_SIZE = frame_size
        cfg.TEST_CROP_SIZE: frame_size
    _, net = load_checkpoint(path_to_checkpoint=weights,model=X3D(cfg),data_parallel=False)
    return cfg, net


def x3d_l(num_frames=None,frame_size=None):
    from models.configs.defaults import assert_and_infer_cfg
    from models.checkpoint import load_checkpoint
    weights='https://dl.fbaipublicfiles.com/pyslowfast/x3d_models/x3d_l.pyth'
    path = 'checkpoints'
    weights = check_weights_url(path,weights)
    
    path_to_config = 'models/configs/kinetics/x3d_l.yaml'
    cfg = load_config(None, path_to_config)
    if num_frames is not None:
        cfg.DATA.NUM_FRAMES = num_frames
    if frame_size is not None:
        cfg.TRAIN_CROP_SIZE= frame_size
        cfg.TEST_CROP_SIZE: frame_size
    _, net = load_checkpoint(path_to_checkpoint=weights,model=X3D(cfg),data_parallel=False)
    return cfg, net


def slow_4x16_r50(num_frames=None,frame_size=None):
    from models.configs.defaults import assert_and_infer_cfg
    from models.checkpoint import load_checkpoint
    weights='https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/kinetics400/SLOWONLY_4x16_R50.pkl'
    path = 'checkpoints'
    weights = check_weights_url(path,weights)
    path_to_config = 'models/configs/kinetics/slow_4x16_r50.yaml'
    cfg = load_config(None, path_to_config)
    if num_frames is not None:
        cfg.DATA.NUM_FRAMES = num_frames
    if frame_size is not None:
        cfg.TRAIN_CROP_SIZE= frame_size
        cfg.TEST_CROP_SIZE: frame_size
    _, net = load_checkpoint(path_to_checkpoint=weights,model=ResNet(cfg),data_parallel=False,convert_from_caffe2=True)
    return cfg, net

def slow_8x8_r50(num_frames=None,frame_size=None):
    from models.configs.defaults import assert_and_infer_cfg
    from models.checkpoint import load_checkpoint
    weights='https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/kinetics400/SLOWONLY_8x8_R50.pkl'
    path = 'checkpoints'
    weights = check_weights_url(path,weights)
    path_to_config = 'models/configs/kinetics/slow_8x8_r50.yaml'
    cfg = load_config(None, path_to_config)
    if num_frames is not None:
        cfg.DATA.NUM_FRAMES = num_frames
    if frame_size is not None:
        cfg.TRAIN_CROP_SIZE= frame_size
        cfg.TEST_CROP_SIZE: frame_size
    _, net = load_checkpoint(path_to_checkpoint=weights,model=ResNet(cfg),data_parallel=False,convert_from_caffe2=True)
    return cfg, net

def slowfast_4x16_r50(num_frames=None,frame_size=None):
    from models.configs.defaults import assert_and_infer_cfg
    from models.checkpoint import load_checkpoint
    weights='https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/kinetics400/SLOWFAST_4x16_R50.pkl'
    path = 'checkpoints'
    weights = check_weights_url(path,weights)
    path_to_config = 'models/configs/kinetics/slowfast_4x16_r50.yaml'
    cfg = load_config(None, path_to_config)
    if num_frames is not None:
        cfg.DATA.NUM_FRAMES = num_frames
    if frame_size is not None:
        cfg.TRAIN_CROP_SIZE= frame_size
        cfg.TEST_CROP_SIZE: frame_size
    _, net = load_checkpoint(path_to_checkpoint=weights,model=SlowFast(cfg),data_parallel=False,convert_from_caffe2=True)
    return cfg, net


def slowfast_8x8_r50(num_frames=None,frame_size=None):
    from models.configs.defaults import assert_and_infer_cfg
    from models.checkpoint import load_checkpoint
    weights='https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/kinetics400/SLOWFAST_8x8_R50.pkl'
    path = 'checkpoints'
    weights = check_weights_url(path,weights)
    path_to_config = 'models/configs/kinetics/slowfast_8x8_r50.yaml'
    cfg = load_config(None, path_to_config)
    if num_frames is not None:
        cfg.DATA.NUM_FRAMES = num_frames
    if frame_size is not None:
        cfg.TRAIN_CROP_SIZE= frame_size
        cfg.TEST_CROP_SIZE: frame_size
    _, net = load_checkpoint(path_to_checkpoint=weights,model=SlowFast(cfg),data_parallel=False,convert_from_caffe2=True)
    return cfg, net


def mvitv2_b_32x3(num_frames=None,frame_size=None):
    from models.configs.defaults import assert_and_infer_cfg
    from models.checkpoint import load_checkpoint
    weights='https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/mvitv2/pysf_video_models/MViTv2_B_32x3_k400_f304025456.pyth'
    path = 'checkpoints'
    weights = check_weights_url(path,weights)
    
    path_to_config = 'models/configs/kinetics/mvitv2_b_32x3.yaml'
    cfg = load_config(None, path_to_config)
    if num_frames is not None:
        cfg.DATA.NUM_FRAMES = num_frames
    if frame_size is not None:
        cfg.TRAIN_CROP_SIZE= frame_size
        cfg.TEST_CROP_SIZE: frame_size
    _, net = load_checkpoint(path_to_checkpoint=weights,model=MViT(cfg),data_parallel=False)
    return cfg, net



def mvitv2_s_16x4(num_frames=None,frame_size=None):
    from models.configs.defaults import assert_and_infer_cfg
    from models.checkpoint import load_checkpoint
    weights='https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/mvitv2/pysf_video_models/MViTv2_S_16x4_k400_f302660347.pyth'
    path = 'checkpoints'
    weights = check_weights_url(path,weights)
    
    path_to_config = 'models/configs/kinetics/mvitv2_s_16x4.yaml'
    cfg = load_config(None, path_to_config)
    if num_frames is not None:
        cfg.DATA.NUM_FRAMES = num_frames
    if frame_size is not None:
        cfg.TRAIN_CROP_SIZE= frame_size
        cfg.TEST_CROP_SIZE: frame_size
    _, net = load_checkpoint(path_to_checkpoint=weights,model=MViT(cfg),data_parallel=False)
    return cfg, net



def rev_mvit(num_frames=None,frame_size=None):
    from models.configs.defaults import assert_and_infer_cfg
    from models.checkpoint import load_checkpoint
    weights='https://dl.fbaipublicfiles.com/pyslowfast/rev/REV_MVIT_B_16x4.pyth'
    path = 'checkpoints'
    weights = check_weights_url(path,weights)
    
    path_to_config = 'models/configs/kinetics/rev_mvit_b_16x4_conv.yaml'
    cfg = load_config(None, path_to_config)
    if num_frames is not None:
        cfg.DATA.NUM_FRAMES = num_frames
    if frame_size is not None:
        cfg.TRAIN_CROP_SIZE= frame_size
        cfg.TEST_CROP_SIZE: frame_size
    _, net = load_checkpoint(path_to_checkpoint=weights,model=MViT(cfg),data_parallel=False)
    return cfg, net



def uniformerv2_b_16_8(num_frames=None,frame_size=None):
    from models.configs.defaults import assert_and_infer_cfg
    from models.checkpoint import load_checkpoint
    weights='https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/uniformerv2/k400/k400_k710_uniformerv2_b16_8x224.pyth'
    path = 'checkpoints'
    weights = check_weights_url(path,weights)
    
    path_to_config = 'models/configs/kinetics/uniformerv2-b_16_8.yaml'
    cfg = load_config(None, path_to_config)
    if num_frames is not None:
        cfg.DATA.NUM_FRAMES = num_frames
    if frame_size is not None:
        cfg.TRAIN_CROP_SIZE= frame_size
        cfg.TEST_CROP_SIZE: frame_size
    _, net = load_checkpoint(path_to_checkpoint=weights,model=Uniformerv2(cfg),data_parallel=False)
    return cfg, net


def uniformerv2_l_14_8(num_frames=None,frame_size=None):
    from models.configs.defaults import assert_and_infer_cfg
    from models.checkpoint import load_checkpoint
    weights='https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/uniformerv2/k400/k400_k710_uniformerv2_l14_8x224.pyth'
    path = 'checkpoints'
    weights = check_weights_url(path,weights)
    
    path_to_config = 'models/configs/kinetics/uniformerv2-l_14_8.yaml'
    cfg = load_config(None, path_to_config)
    if num_frames is not None:
        cfg.DATA.NUM_FRAMES = num_frames
    if frame_size is not None:
        cfg.TRAIN_CROP_SIZE= frame_size
        cfg.TEST_CROP_SIZE: frame_size
    _, net = load_checkpoint(path_to_checkpoint=weights,model=Uniformerv2(cfg),data_parallel=False)
    return cfg, net



def uniformerv2_l_14_16(num_frames=None,frame_size=None):
    from models.configs.defaults import assert_and_infer_cfg
    from models.checkpoint import load_checkpoint
    weights='https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/uniformerv2/k400/k400_k710_uniformerv2_l14_16x224.pyth'
    path = 'checkpoints'
    weights = check_weights_url(path,weights)
    
    path_to_config = 'models/configs/kinetics/uniformerv2-l_14_16.yaml'
    cfg = load_config(None, path_to_config)
    if num_frames is not None:
        cfg.DATA.NUM_FRAMES = num_frames
    if frame_size is not None:
        cfg.TRAIN_CROP_SIZE= frame_size
        cfg.TEST_CROP_SIZE: frame_size
    _, net = load_checkpoint(path_to_checkpoint=weights,model=Uniformerv2(cfg),data_parallel=False)
    return cfg, net



def swin_t(num_frames=None,frame_size=None):
    from models.configs.defaults import assert_and_infer_cfg
    from models.checkpoint import load_checkpoint
    weights='https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_tiny_patch244_window877_kinetics400_1k.pth'
    path = 'checkpoints'
    weights = check_weights_url(path,weights)
    
    path_to_config = 'models/configs/kinetics/swin_t.yaml'
    cfg = load_config(None, path_to_config)
    if num_frames is not None:
        cfg.DATA.NUM_FRAMES = num_frames
    if frame_size is not None:
        cfg.TRAIN_CROP_SIZE= frame_size
        cfg.TEST_CROP_SIZE: frame_size
    _, net = load_checkpoint(path_to_checkpoint=weights,model=SwinTransformer3D(cfg),data_parallel=False)
    return cfg, net


def swin_s(num_frames=None,frame_size=None):
    from models.configs.defaults import assert_and_infer_cfg
    from models.checkpoint import load_checkpoint
    weights='https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_small_patch244_window877_kinetics400_1k.pth'
    path = 'checkpoints'
    weights = check_weights_url(path,weights)
    
    path_to_config = 'models/configs/kinetics/swin_s.yaml'
    cfg = load_config(None, path_to_config)
    if num_frames is not None:
        cfg.DATA.NUM_FRAMES = num_frames
    if frame_size is not None:
        cfg.TRAIN_CROP_SIZE= frame_size
        cfg.TEST_CROP_SIZE: frame_size
    _, net = load_checkpoint(path_to_checkpoint=weights,model=SwinTransformer3D(cfg),data_parallel=False)
    return cfg, net


def swin_b(num_frames=None,frame_size=None):
    from models.configs.defaults import assert_and_infer_cfg
    from models.checkpoint import load_checkpoint
    weights='https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_base_patch244_window877_kinetics400_22k.pth'
    path = 'checkpoints'
    weights = check_weights_url(path,weights)
    
    path_to_config = 'models/configs/kinetics/swin_b.yaml'
    cfg = load_config(None, path_to_config)
    if num_frames is not None:
        cfg.DATA.NUM_FRAMES = num_frames
    if frame_size is not None:
        cfg.TRAIN_CROP_SIZE= frame_size
        cfg.TEST_CROP_SIZE: frame_size
    _, net = load_checkpoint(path_to_checkpoint=weights,model=SwinTransformer3D(cfg),data_parallel=False)
    return cfg, net


def timesformer(num_frames=None,frame_size=None):
    from models.configs.defaults import assert_and_infer_cfg
    from models.checkpoint import load_checkpoint
    weights='https://www.dropbox.com/s/g5t24we9gl5yk88/TimeSformer_divST_8x32_224_K400.pyth?dl=1'
    path = 'checkpoints'
    weights = check_weights_url(path,weights)
    
    path_to_config = 'models/configs/kinetics/timesformer.yaml'
    cfg = load_config(None, path_to_config)
    if num_frames is not None:
        cfg.DATA.NUM_FRAMES = num_frames
    if frame_size is not None:
        cfg.TRAIN_CROP_SIZE= frame_size
        cfg.TEST_CROP_SIZE: frame_size
    _, net = load_checkpoint(path_to_checkpoint=weights,model=vit_base_patch16_224(cfg),data_parallel=False)
    return cfg, net


def csn(num_frames=None,frame_size=None):
    import pytorchvideo.models as models
    from models.checkpoint import load_checkpoint
    weights='https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/CSN_32x2_R101.pyth'
    path = 'checkpoints'
    weights = check_weights_url(path,weights)
    _, net = load_checkpoint(path_to_checkpoint=weights,model=models.create_csn(model_depth=101),data_parallel=False)
    return None, net

def r2_plus_1d(num_frames=None,frame_size=None):
    from pytorchvideo.models.r2plus1d import create_r2plus1d
    from models.checkpoint import load_checkpoint
    weights='https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/R2PLUS1D_16x4_R50.pyth'
    path = 'checkpoints'
    weights = check_weights_url(path,weights)
    _, net = load_checkpoint(path_to_checkpoint=weights,model=create_r2plus1d(model_depth=50),data_parallel=False)
    return None, net
    

def get_models_dict():
    models = {'csn' : csn,
              'r2_plus_1d' : r2_plus_1d,
              'x3d_xs': x3d_xs,
              'x3d_s': x3d_s,
              'x3d_m': x3d_m,
              'x3d_l': x3d_l,
              'slow_4x16_r50': slow_4x16_r50,
              'slow_8x8_r50' : slow_8x8_r50,
              'mvitv2_b_32x3': mvitv2_b_32x3,
              'mvitv2_s_16x4': mvitv2_s_16x4,
              'rev_mvit' : rev_mvit,
              'uniformerv2_l_14_8' : uniformerv2_l_14_8,
              'uniformerv2_b_16_8' : uniformerv2_b_16_8,
              'uniformerv2_l_14_16' : uniformerv2_l_14_16,
              'swin_t' : swin_t,
              'swin_s' : swin_s,
              'swin_b' : swin_b,
              'timesformer' : timesformer
              }

    return models  
    



if __name__ == "__main__":
    from ptflops import get_model_complexity_info
    from torchinfo import summary

    ####################################
    ##### N E T W O R K  T E S T S  ####
    ####################################
    

    tmp = (3,16,224,224)
    _, model = csn()
    model=model.cuda()
    macs, params = get_model_complexity_info(model, tmp, as_strings=True,print_per_layer_stat=False, verbose=False)
    print('-- CSN R101 --- ')
    #summary(model, data=torch.rand(tmp).cuda())
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of Parameters: ', params))
    print('\n')   
    
    
    tmp = (3,16,224,224)
    _, model = r2_plus_1d()
    model=model.cuda()
    macs, params = get_model_complexity_info(model, tmp, as_strings=True,print_per_layer_stat=False, verbose=False)
    print('-- R50 2+1D --- ')
    #summary(model, data=torch.rand(tmp).cuda())
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of Parameters: ', params))
    print('\n')   
    
    _, model = x3d_xs()
    tmp = (3,4,182,182)
    model=model.cuda()
    macs, params = get_model_complexity_info(model, tmp, as_strings=True,print_per_layer_stat=False, verbose=False)
    print('-- X3D XS --- ')
    #summary(model, data=torch.rand(tmp).cuda())
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of Parameters: ', params))
    print('\n')
    
    _, model = x3d_s()
    tmp = (3,13,182,182)
    model=model.cuda()
    macs, params = get_model_complexity_info(model, tmp, as_strings=True,print_per_layer_stat=False, verbose=False)
    print('-- X3D S --- ')
    #summary(model, data=torch.rand(tmp).cuda())
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of Parameters: ', params))
    print('\n')
    
    _, model = x3d_m()
    tmp = (3,16,224,224)
    model=model.cuda()
    macs, params = get_model_complexity_info(model, tmp, as_strings=True,print_per_layer_stat=False, verbose=False)
    print('-- X3D M --- ')
    summary(model, data=torch.rand(tmp).cuda())
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of Parameters: ', params))
    print('\n')
    
    
    _, model = x3d_l()
    tmp = (3,16,356,356)
    model=model.cuda()
    macs, params = get_model_complexity_info(model, tmp, as_strings=True,print_per_layer_stat=False, verbose=False)
    print('-- X3D L --- ')
    #summary(model, data=torch.rand(tmp).cuda())
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of Parameters: ', params))
    print('\n')
    
    
    _, model = slow_4x16_r50()
    tmp = (3,4,224,224)
    model=model.cuda()
    macs, params = get_model_complexity_info(model, tmp, as_strings=True,print_per_layer_stat=False, verbose=False)
    print('-- SLOW 4x16 R50 --- ')
    #summary(model, data=torch.rand(tmp).cuda())
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of Parameters: ', params))
    print('\n')
    

    _, model = slow_8x8_r50()
    tmp = (3,8,224,224)
    model=model.cuda()
    macs, params = get_model_complexity_info(model, tmp, as_strings=True,print_per_layer_stat=False, verbose=False)
    print('-- SLOW 8x8 R50 --- ')
    #summary(model, data=torch.rand(tmp).cuda())
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of Parameters: ', params))
    print('\n')
    
    
    _, model = slowfast_4x16_r50()
    tmp = ((3,4,224,224),(3,16,224,224))
    model=model.cuda()
    #macs, params = get_model_complexity_info(model, tmp, as_strings=True,print_per_layer_stat=False, verbose=False)
    print('-- SLOWFAST 4x16 R50 --- ')
    #summary(model, data=[torch.ones(tmp[0]).unsqueeze(0).cuda(),torch.ones(tmp[1]).unsqueeze(0).cuda()], depth=1)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of Parameters: ', params))
    print('\n')
    
    
    _, model = slowfast_8x8_r50()
    tmp = ((3,8,224,224),(3,8,224,224))
    model=model.cuda()
    #macs, params = get_model_complexity_info(model, tmp, as_strings=True,print_per_layer_stat=False, verbose=False)
    print('-- SLOWFAST 8x8 R50 --- ')
    #summary(model, data=[torch.ones(tmp[0]).unsqueeze(0).cuda(),torch.ones(tmp[1]).unsqueeze(0).cuda()], depth=1)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of Parameters: ', params))
    print('\n')


    tmp = (3,16,224,224)
    _, model = mvitv2_b_32x3(num_frames = tmp[1],frame_size=tmp[-1])
    model=model.cuda()
    macs, params = get_model_complexity_info(model, tmp, as_strings=True,print_per_layer_stat=False, verbose=False)
    print('-- MVITv2 B 32x3 --- ')
    #summary(model, data=torch.rand(tmp).cuda())
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of Parameters: ', params))
    print('\n')
    
    
    
    tmp = (3,16,224,224)
    _, model = mvitv2_s_16x4(num_frames = tmp[1],frame_size=tmp[-1])
    model=model.cuda()
    macs, params = get_model_complexity_info(model, tmp, as_strings=True,print_per_layer_stat=False, verbose=False)
    print('-- MVITv2 S 16x4 --- ')
    #summary(model, data=torch.rand(tmp).cuda())
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of Parameters: ', params))
    print('\n')
    
    
    tmp = (3,16,224,224)
    _, model = rev_mvit(num_frames = tmp[1],frame_size=tmp[-1])
    model=model.cuda()
    macs, params = get_model_complexity_info(model, tmp, as_strings=True,print_per_layer_stat=False, verbose=False)
    print('-- REV MVIT B 16x4 --- ')
    #summary(model, data=torch.rand(tmp).cuda())
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of Parameters: ', params))
    print('\n')
    
    
    tmp = (1, 3,6,224,224)
    _, model = uniformerv2_b_16_8(num_frames = tmp[1],frame_size=tmp[-1])
    model=model.cuda()
    macs, params = get_model_complexity_info(model, tmp, as_strings=True,print_per_layer_stat=False, verbose=False)
    print('-- UNIFORMERV2 B 16 --- ')
    #summary(model, data=torch.rand(tmp).cuda())
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of Parameters: ', params))
    print('\n')
    
    
    tmp = (1, 3,6,224,224)
    _, model = uniformerv2_l_14_8(num_frames = tmp[1],frame_size=tmp[-1])
    model=model.cuda()
    macs, params = get_model_complexity_info(model, tmp, as_strings=True,print_per_layer_stat=False, verbose=False)
    print('-- UNIFORMERV2 L 14 x8 --- ')
    #summary(model, data=torch.rand(tmp).cuda())
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of Parameters: ', params))
    print('\n')

    
    tmp = (1, 3,12,224,224)
    _, model = uniformerv2_l_14_16(num_frames = tmp[1],frame_size=tmp[-1])
    model=model.cuda()
    macs, params = get_model_complexity_info(model, tmp, as_strings=True,print_per_layer_stat=False, verbose=False)
    print('-- UNIFORMERV2 L 14 x16 --- ')
    #summary(model, data=torch.rand(tmp).cuda())
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of Parameters: ', params))
    print('\n')
    

    tmp = (3,16,224,224)
    _, model = swin_t(num_frames = tmp[1],frame_size=tmp[-1])
    model=model.cuda()
    macs, params = get_model_complexity_info(model, tmp, as_strings=True,print_per_layer_stat=False, verbose=False)
    print('-- SWIN T --- ')
    #summary(model, data=torch.rand(tmp).cuda())
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of Parameters: ', params))
    print('\n')
    
    tmp = (3,16,224,224)
    _, model = swin_s(num_frames = tmp[1],frame_size=tmp[-1])
    model=model.cuda()
    macs, params = get_model_complexity_info(model, tmp, as_strings=True,print_per_layer_stat=False, verbose=False)
    print('-- SWIN S --- ')
    #summary(model, data=torch.rand(tmp).cuda())
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of Parameters: ', params))
    print('\n')
    
    
    tmp = (3,16,224,224)
    _, model = swin_b(num_frames = tmp[1],frame_size=tmp[-1])
    model=model.cuda()
    macs, params = get_model_complexity_info(model, tmp, as_strings=True,print_per_layer_stat=False, verbose=False)
    print('-- SWIN B --- ')
    #summary(model, data=torch.rand(tmp).cuda())
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of Parameters: ', params))
    print('\n')
    
    tmp = (3,8,224,224)
    _, model = timesformer(num_frames = tmp[1],frame_size=tmp[-1])
    model=model.cuda()
    macs, params = get_model_complexity_info(model, tmp, as_strings=True,print_per_layer_stat=False, verbose=False)
    print('-- TimeSFormer --- ')
    #summary(model, data=torch.rand(tmp).cuda())
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of Parameters: ', params))
    print('\n')
