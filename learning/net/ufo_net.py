import numpy as np
import os
import os.path as osp
import torch
from torch import nn
import torch.nn.functional as F
import time
import bson
from typing import Tuple
from loguru import logger
from learning.components.mlp import MLP_V2
import pytorch_lightning as pl
from learning.net.resunet import SparseResUNet
from learning.net.pointnet import MiniPointNetfeat
from learning.net.transformer import Transformer
from common.datamodels import PredictionMessage, ActionTypeDef
import MinkowskiEngine as ME


class ActionHead(pl.LightningModule):
    """concat nocs feature, global feature, dense feature as nocs feature input"""
    def __init__(self,
                 global_nn_channels: tuple = (128, 256, 1024),
                 cls_base_nn_channels: tuple = (1024, 256, 128),
                 pointnet_channels: tuple = (3, 64, 128, 512),
                 grasp_nocs_feat_nn_channels: tuple = (512 + 64 + 1024 + 128, 512, 256),
                 grasp_nocs_score_nn_channels: tuple = (256, 128, 4),
                 grasp_nocs_att_nn_channels: tuple = (256, 128, 4),
                 grasp_nocs_offset_nn_channels: tuple = (256, 128, 4*3),
                 offset_nn_channels: tuple = (128, 256, 128, 8),  # only predict (x, y) coordinate
                 att_nn_channels: tuple = (128, 256, 128, 4),
                 nocs_nn_channels: tuple = (128, 256, 128, 3),
                 num_smoothing_actions: int = 2,  # (fling, drag)
                 num_folding_actions: int = 4,  # short: (drag, fold1, fold2, done)
                 num_action_type: int = 4,  # short: (fling, drag, fold1, fold2)
                 min_gt_nocs_ratio: float = 0.2,
                 gt_nocs_ratio_decay_factor: float = 0.98,  # for 100 epoch setting
                 num_pred_fling_candidates: int = 4,  # number of possible fling candidates
                 use_xyz_variety_loss: bool = False,
                 use_gt_nocs_pred_for_distance_weight: bool = False,
                 nocs_distance_weight_alpha: float = 30.0,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.num_smoothing_actions = num_smoothing_actions
        self.num_folding_actions = num_folding_actions
        self.num_action_type = num_action_type
        self.num_pred_fling_candidates = num_pred_fling_candidates

        grasp_nocs_score_nn_channels[-1] = num_pred_fling_candidates
        grasp_nocs_att_nn_channels[-1] = num_pred_fling_candidates
        grasp_nocs_offset_nn_channels[-1] = 3 * num_pred_fling_candidates
        assert grasp_nocs_score_nn_channels[-1] % num_pred_fling_candidates == 0 \
               and grasp_nocs_att_nn_channels[-1] % num_pred_fling_candidates == 0 \
               and grasp_nocs_offset_nn_channels[-1] % num_pred_fling_candidates == 0

        self.nocs_pointnet = MiniPointNetfeat(nn_channels=pointnet_channels)
        self.grasp_nocs_feat_mlp = MLP_V2(grasp_nocs_feat_nn_channels, transpose_input=True)
        self.grasp_nocs_score_mlp = MLP_V2(grasp_nocs_score_nn_channels, transpose_input=True)
        self.grasp_nocs_att_mlp = MLP_V2(grasp_nocs_att_nn_channels, transpose_input=True)
        self.grasp_nocs_offset_mlp = MLP_V2(grasp_nocs_offset_nn_channels, transpose_input=True)
        # predict attention and offset for other actions
        # TODO: remove additional channels for fling action
        self.offset_mlp_list = nn.ModuleList([
            MLP_V2(offset_nn_channels, transpose_input=True) for _ in range(num_action_type)
        ])
        self.att_mlp_list = nn.ModuleList([
            MLP_V2(att_nn_channels, transpose_input=True) for _ in range(num_action_type)
        ])
        self.global_mlp = MLP_V2(global_nn_channels, transpose_input=True)
        self.nocs_mlp = MLP_V2(nocs_nn_channels, transpose_input=True)

        self.smoothing_cls_mlp = MLP_V2(cls_base_nn_channels + (num_smoothing_actions, ), transpose_input=True)
        self.folding_cls_mlp = MLP_V2(cls_base_nn_channels + (num_folding_actions,), transpose_input=True)
        self.smoothed_cls_mlp = MLP_V2(cls_base_nn_channels + (1,), transpose_input=True)
        self.is_folding_cls_mlp = MLP_V2(cls_base_nn_channels + (1,), transpose_input=True)

        self.gt_nocs_ratio = 1.0
        self.min_gt_nocs_ratio = min_gt_nocs_ratio
        self.gt_nocs_ratio_decay_factor = gt_nocs_ratio_decay_factor
        self.use_xyz_variety_loss = use_xyz_variety_loss
        self.use_gt_nocs_pred_for_distance_weight = use_gt_nocs_pred_for_distance_weight
        self.nocs_distance_weight_alpha = nocs_distance_weight_alpha

    def forward(self, dense_feat: torch.Tensor, gt_pc_nocs: torch.Tensor = None):
        dense_feat_extra = self.global_mlp(dense_feat)  # (B, N, C')
        global_feat = torch.max(dense_feat_extra, dim=1)[0]  # (B, C)
        smoothing_logits = self.smoothing_cls_mlp(global_feat)
        folding_logits = self.folding_cls_mlp(global_feat)
        smoothed_logits = self.smoothed_cls_mlp(global_feat)
        is_folding_logits = self.is_folding_cls_mlp(global_feat)

        pred_nocs = self.nocs_mlp(dense_feat)  # (B, N, 3)
        if gt_pc_nocs is not None:
            # use GT NOCS during training
            self.gt_nocs_ratio = max(self.min_gt_nocs_ratio,
                                     self.gt_nocs_ratio_decay_factor ** self.current_epoch)
            use_gt_nocs = torch.rand(1).item() < self.gt_nocs_ratio
        else:
            use_gt_nocs = False
        input_pc_nocs = gt_pc_nocs.transpose(1, 2) if use_gt_nocs else pred_nocs.detach().transpose(1, 2)  # (B, 3, N)
        dense_nocs_feat, _ = self.nocs_pointnet(input_pc_nocs)  # (B, C", N)
        num_pts = dense_feat_extra.shape[1]
        batch_size = dense_feat_extra.shape[0]
        global_feat_expand = global_feat.unsqueeze(-1).expand(-1, -1, num_pts)  # (B, C, N)
        dense_nocs_feat_cat = torch.cat([dense_nocs_feat, global_feat_expand], dim=1).transpose(1, 2)  # (B, N, C+C")
        dense_nocs_feat_cat = torch.cat([dense_nocs_feat_cat, dense_feat], dim=2)  # (B, N, C+C'+C")
        dense_nocs_feat_fuse = self.grasp_nocs_feat_mlp(dense_nocs_feat_cat)  # (B, N, C''')
        nocs_grasp_score = self.grasp_nocs_score_mlp(dense_nocs_feat_fuse).mean(dim=1)  # (B, K)
        nocs_att_logits = self.grasp_nocs_att_mlp(dense_nocs_feat_fuse).reshape(batch_size, num_pts, -1, 1)  # (B, N, K, 1)
        nocs_att_score = F.softmax(nocs_att_logits, dim=1)  # (B, N, K, 1)
        nocs_offset = self.grasp_nocs_offset_mlp(dense_nocs_feat_fuse).reshape(batch_size, num_pts, -1, 3)  # (B, N, K, 3)
        input_pc_nocs_expand = input_pc_nocs.transpose(1, 2).view(batch_size, num_pts, 1, 3)\
            .expand(-1, -1, self.num_pred_fling_candidates, -1)  # (B, N, K, 3)
        grasp_nocs = ((input_pc_nocs_expand + nocs_offset) * nocs_att_score).sum(dim=1)  # (B, K, 3)

        offset_list = []
        att_list = []
        for idx in range(self.num_action_type):
            offset_list.append(self.offset_mlp_list[idx](dense_feat))
            att_list.append(self.att_mlp_list[idx](dense_feat))
        offset_list[0] = nocs_offset
        att_list[0] = nocs_att_logits
        return offset_list, att_list, pred_nocs, grasp_nocs, nocs_grasp_score, dense_nocs_feat_fuse, \
               smoothing_logits, folding_logits, smoothed_logits, is_folding_logits

class VirtualRewardHead(pl.LightningModule):
    """
    predict factorized reward (canonicalization reward + alignment reward)
    """
    def __init__(self,
                 fuse_nn_channels: tuple = ((128 + 6) * 2, 256, 128),
                 canonicalization_nn_channels: tuple = (128, 128, 1),
                 alignment_nn_channels: tuple = (128, 128, 1),
                 deformable_weight_nn_channels: tuple = (128, 128, 1),
                 xyz_distance_weight_alpha: float = 50.0,
                 deformable_weight: float = 0.7,
                 use_dense_feat: bool = True,
                 use_nocs_dense_feat: bool = True,
                 use_dynamic_deformable_weight: bool = False,
                 enable_reward_normalization: bool = False,
                 **kwargs
                 ):
        super().__init__()
        self.save_hyperparameters()

        self.canonicalization_nn_channels = canonicalization_nn_channels
        self.alignment_nn_channels = alignment_nn_channels
        self.xyz_distance_weight_alpha = xyz_distance_weight_alpha
        self.deformable_weight_constant = deformable_weight
        self.use_dynamic_deformable_weight = use_dynamic_deformable_weight
        self.use_dense_feat = use_dense_feat
        self.use_nocs_dense_feat = use_nocs_dense_feat
        self.enable_reward_normalization = enable_reward_normalization

        self.fuse_mlp = MLP_V2(fuse_nn_channels, transpose_input=True)
        self.canonicalization_reward_mlp = MLP_V2(self.canonicalization_nn_channels, transpose_input=True)
        self.alignment_reward_mlp = MLP_V2(self.alignment_nn_channels, transpose_input=True)
        self.deformable_weight_mlp = MLP_V2(deformable_weight_nn_channels, transpose_input=True)
        self.reward_bn = nn.BatchNorm1d(1)

    def forward(self,
                pc_xyz: torch.Tensor,
                pick_pts_xyz: torch.Tensor,
                pc_nocs: torch.Tensor,
                dense_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            pc_xyz: (B, N, 3)
            pick_pts_xyz: (B, 2, K, 3)
            pc_nocs: (B, N, 3)
            dense_feat: (B, N, C)
        Returns:
            canonicalization_reward: (B, K, 1)
            alignment_reward: (B, K, 1)
            deformable_weight: (B, K, 1)
        """
        assert pick_pts_xyz.shape[1] == 2
        B = pc_xyz.shape[0]
        K = pick_pts_xyz.shape[2]
        N = pc_xyz.shape[1]
        pick_pts_xyz_expand = pick_pts_xyz.reshape(B, -1, 3).unsqueeze(2).expand(-1, -1, N, -1)  # (B, 2*K, N, 3)
        pc_xyz_expand = pc_xyz.unsqueeze(1).expand(-1, 2*K, -1, -1)  # (B, 2*K, N, 3)
        xyz_distance = torch.norm(pick_pts_xyz_expand - pc_xyz_expand, dim=-1)  # (B, 2*K, N)

        xyz_distance_weight = torch.exp(-self.xyz_distance_weight_alpha * xyz_distance)  # (B, 2*K, N)
        normalized_xyz_distance_weight = xyz_distance_weight / \
                                         xyz_distance_weight.sum(dim=-1, keepdim=True) + 1e-6  # (B, 2*K, N)
        dense_feat_interpolated = (normalized_xyz_distance_weight.unsqueeze(-1) * dense_feat.unsqueeze(1)).sum(dim=2)  # (B, 2*K, C)
        pc_nocs_interpolated = (normalized_xyz_distance_weight.unsqueeze(-1) * pc_nocs.unsqueeze(1)).sum(dim=2)  # (B, 2*K, 3)
        pc_nocs_interpolated_reshape = pc_nocs_interpolated.reshape(B, 2, K, -1)  # (B, 2, K, 3)
        if self.use_dense_feat:
            dense_feat_interpolated_reshape = dense_feat_interpolated.reshape(B, 2, K, -1)  # (B, 2, K, C)
            feat_cat1 = torch.cat([dense_feat_interpolated_reshape[:, 0, :, :],
                                   pick_pts_xyz[:, 0, :, :],
                                   pc_nocs_interpolated_reshape[:, 0, :, :]], dim=-1)  # (B, K, C+3+3)
            feat_cat2 = torch.cat([dense_feat_interpolated_reshape[:, 1, :, :],
                                   pick_pts_xyz[:, 1, :, :],
                                   pc_nocs_interpolated_reshape[:, 1, :, :]], dim=-1)  # (B, K, C+3+3)
        else:
            feat_cat1 = torch.cat([pick_pts_xyz[:, 0, :, :],
                                   pc_nocs_interpolated_reshape[:, 0, :, :]], dim=-1)
            feat_cat2 = torch.cat([pick_pts_xyz[:, 1, :, :],
                                   pc_nocs_interpolated_reshape[:, 1, :, :]], dim=-1)
        cat_feat = torch.cat([feat_cat1, feat_cat2], dim=-1)  # (B, K, (C+3+3)*2)
        fuse_feat = self.fuse_mlp(cat_feat)  # (B, K, C')
        canonicalization_reward = self.canonicalization_reward_mlp(fuse_feat)  # (B, K, 1)
        alignment_reward = self.alignment_reward_mlp(fuse_feat)  # (B, K, 1)
        if self.use_dynamic_deformable_weight:
            deformable_weight = self.deformable_weight_mlp(fuse_feat)
        else:
            deformable_weight = torch.ones_like(canonicalization_reward, requires_grad=False) * self.deformable_weight_constant
        return canonicalization_reward, alignment_reward, deformable_weight


class UFONet(pl.LightningModule):
    """
    Use Res-UNet3D as backbone, use Point cloud as input
    use Transformer to encode dense per-point feature
    use attention + offset to predict grasp points and release points
    predict K independent grasp-points in NOCS space  for fling action, use variety loss (nocs) for supervision
    factorized reward prediction
    """
    def __init__(self,
                 # sparse uned3d encoder params
                 sparse_unet3d_encoder_params,
                 # transformer params
                 transformer_params,
                 # action head params
                 action_head_params,
                 # virtual reward head params
                 virtual_reward_head_params,
                 # hyper-params
                 num_rotations: int = 10,
                 smoothing_cls_weight: tuple = (0.1, 0.1),
                 folding_cls_weight: tuple = (4.0, 1.0, 1.0, 1.0),
                 primitive_classes: tuple = ('fling', 'drag', 'fold1', 'fold2', 'pick_and_place', 'done'),
                 smoothing_primitive_idxs: tuple = (0, 1),
                 folding_primitive_idxs: tuple = (1, 2, 3, 5),
                 ignore_folding_cls: bool = False,
                 # loss weights
                 loss_cls_weight: float = 0.1,
                 loss_grasp_weight: float = 100.0,
                 loss_nocs_weight: float = 100.0,
                 loss_offset_weight: float = 1.0,
                 loss_canonicalization_reward_weight: float = 1.0,
                 loss_alignment_reward_weight: float = 0.2,
                 loss_ranking_weight: float = 1.0,
                 virtual_grasp_loss_weight_factor: float = 1.0,
                 virtual_nocs_loss_weight_factor: float = 1.0,
                 # optimizer params
                 use_cos_lr: bool = False,
                 cos_t_max: int = 100,
                 batch_size: int = 16,
                 # others
                 use_virtual_reward_for_inference: bool = True,
                 enable_real_world_finetune: bool = False,
                 freeze_reward_mlp_for_real_world_finetune: bool = False,
                 freeze_backbone_for_real_world_finetune: bool = False,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.num_rotations = num_rotations
        self.smoothing_cls_weight = torch.tensor(smoothing_cls_weight)
        self.folding_cls_weight = torch.tensor(folding_cls_weight)
        self.angles = np.linspace(0.0, 2 * np.pi, num_rotations, endpoint=False)
        self.angles[self.angles > np.pi] -= 2 * np.pi
        self.primitives = list(primitive_classes)
        self.smoothing_primitive_idxs = list(smoothing_primitive_idxs)
        self.folding_primitive_idxs = list(folding_primitive_idxs)
        self.ignore_folding_cls = ignore_folding_cls
        # loss weights
        self.loss_cls_weight = loss_cls_weight
        self.loss_grasp_weight = loss_grasp_weight
        self.loss_nocs_weight = loss_nocs_weight
        self.loss_offset_weight = loss_offset_weight
        self.loss_canonicalization_reward_weight = loss_canonicalization_reward_weight
        self.loss_alignment_reward_weight = loss_alignment_reward_weight
        self.loss_ranking_weight = loss_ranking_weight
        self.virtual_grasp_loss_weight_factor = virtual_grasp_loss_weight_factor
        self.virtual_nocs_loss_weight_factor = virtual_nocs_loss_weight_factor
        # optimizer params
        self.use_cos_lr = use_cos_lr
        self.cos_t_max = cos_t_max
        self.batch_size = batch_size
        # others
        self.use_virtual_reward_for_inference = use_virtual_reward_for_inference
        self.enable_real_world_finetune = enable_real_world_finetune
        self.freeze_reward_mlp_for_real_world_finetune = freeze_reward_mlp_for_real_world_finetune
        self.freeze_backbone_for_real_world_finetune = freeze_backbone_for_real_world_finetune

        self.backbone = SparseResUNet(**sparse_unet3d_encoder_params)
        self.transformer = Transformer(**transformer_params)
        self.action_head = ActionHead(**action_head_params)
        self.virtual_reward_head = VirtualRewardHead(**virtual_reward_head_params)

        self.sigmoid = nn.Sigmoid()

    def configure_optimizers(self):
        if self.enable_real_world_finetune:
            if self.freeze_backbone_for_real_world_finetune:
                freeze_heads = [self.action_head.nocs_mlp,
                                self.action_head.grasp_nocs_feat_mlp]
                if self.freeze_reward_mlp_for_real_world_finetune:
                    freeze_heads.extend([self.virtual_reward_head.alignment_reward_mlp,
                                         self.virtual_reward_head.canonicalization_reward_mlp])
                for head in freeze_heads:
                    for param in head.parameters():
                        param.requires_grad = False
                # only update the parameters in action head and virtual reward head
                action_head_params = filter(lambda p: p.requires_grad, self.action_head.parameters())
                virtual_reward_head_params = filter(lambda p: p.requires_grad, self.virtual_reward_head.parameters())
                optimizer = torch.optim.Adam([
                    {'params': action_head_params},
                    {'params': virtual_reward_head_params}
                ], lr=1e-4)
            else:
                optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        if self.use_cos_lr:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.cos_t_max)
        else:
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)
        return [optimizer], [scheduler]

    def forward(self, coords: torch.Tensor, feat: torch.Tensor, pc_xyz: torch.Tensor, gt_pc_nocs: torch.Tensor = None):
        input = ME.SparseTensor(feat, coordinates=coords)
        dense_feat = self.backbone(input)  # (B*N, C)
        dense_feat_att = self.transformer(dense_feat, pc_xyz.view(-1, 3))  # (B, N, C)
        # TODO: add rotation angle prediction
        offset_list, att_list, pred_nocs, grasp_nocs, nocs_grasp_score, dense_nocs_feat_fuse, \
            smoothing_logits, folding_logits, smoothed_logits, is_folding_logits = \
            self.action_head(dense_feat_att, gt_pc_nocs=gt_pc_nocs)
        return dense_feat_att, offset_list, att_list, pred_nocs, grasp_nocs, nocs_grasp_score, dense_nocs_feat_fuse,\
               smoothing_logits, folding_logits, smoothed_logits, is_folding_logits

    @staticmethod
    def bce_loss(prediction, target, weights=None):
        if weights is None:
            weights = 1.0
        valid_count = max(weights[:, 0].sum().item(), 1.0)
        return (weights * nn.BCEWithLogitsLoss(reduction='none')(prediction, target.float())).mean(dim=1).sum() / valid_count

    @staticmethod
    def sym_grasp_huber_variety_cls_loss(pred_grasp_nocs, pred_grasp_score, pred_pc_nocs,
                                         pc_xyz, xyz_target, nocs_target=None, weights=None,
                                         use_xyz_variety_loss=False, alpha=30.0):
        """

        :param pred_grasp_nocs:  (B, K, 3)
        :param pred_grasp_score: (B, K)
        :param pred_pc_nocs: (B, N, 3)
        :param pc_xyz: (B, N, 3)
        :param xyz_target: (B, 2, 3)
        :param nocs_target: (B, 2, 3)
        :param weights:
        :param use_xyz_variety_loss: bool, whether to calculate xyz variety loss
        :param alpha: float, the weight for exponential function in nocs-distance weight calculation
        :return:
        """
        B = pred_grasp_nocs.shape[0]
        K = pred_grasp_nocs.shape[1]
        N = pc_xyz.shape[1]
        device = pred_grasp_nocs.device
        if weights is None:
            weights = 1.0
        valid_count = max(weights.sum().item(), 1.0)

        if nocs_target is None:
            left_xyz_target_expand = xyz_target[:, 0, :].unsqueeze(1).expand(-1, N, -1)  # (B, N, 3)
            right_xyz_target_expand = xyz_target[:, 1, :].unsqueeze(1).expand(-1, N, -1)  # (B, N, 3)
            left_target_xyz_distance = torch.norm(left_xyz_target_expand - pc_xyz, dim=-1)  # (B, N)
            right_target_xyz_distance = torch.norm(right_xyz_target_expand - pc_xyz, dim=-1)  # (B, N)
            left_xyz_distance_weight = torch.exp(-alpha * left_target_xyz_distance)  # (B, N)
            right_xyz_distance_weight = torch.exp(-alpha * right_target_xyz_distance)  # (B, N)
            normalized_left_xyz_distance_weight = left_xyz_distance_weight / left_xyz_distance_weight.sum(dim=-1, keepdim=True)  # (B, N)
            normalized_right_xyz_distance_weight = right_xyz_distance_weight / right_xyz_distance_weight.sum(dim=-1, keepdim=True)  # (B, N)
            # detach pred_pc_nocs, so it won't be backpropagated
            left_nocs_target = torch.sum(pred_pc_nocs.detach() * normalized_left_xyz_distance_weight.unsqueeze(-1), dim=1)  # (B, 3)
            right_nocs_target = torch.sum(pred_pc_nocs.detach() * normalized_right_xyz_distance_weight.unsqueeze(-1), dim=1)  # (B, 3)
            nocs_target = torch.stack([left_nocs_target, right_nocs_target], dim=1)  # (B, 2, 3)

        # nocs variety loss
        nocs_metric = torch.nn.HuberLoss(delta=0.1, reduction='none')
        left_target_nocs = nocs_target[:, 0, :].unsqueeze(1).expand(-1, K, -1)  # (B, K, 3)
        right_target_nocs = nocs_target[:, 1, :].unsqueeze(1).expand(-1, K, -1)  # (B, K, 3)

        left_grasp_loss_nocs = nocs_metric(pred_grasp_nocs, left_target_nocs).mean(dim=-1)  # (B, K)
        right_grasp_loss_nocs = nocs_metric(pred_grasp_nocs, right_target_nocs).mean(dim=-1)  # (B, K)

        left_variety_loss_nocs, left_target_idxs_nocs = torch.min(left_grasp_loss_nocs, dim=1)  # (B, )
        right_variety_loss_nocs, right_target_idxs_nocs = torch.min(right_grasp_loss_nocs, dim=1)  # (B, )
        variety_loss_nocs = (left_variety_loss_nocs + right_variety_loss_nocs) / 2.0  # (B, )
        loss_grasp_variety_nocs = (variety_loss_nocs * weights).sum() / valid_count

        # binary classfication loss for each grasp point
        target_scores = torch.zeros_like(pred_grasp_score)  # (B, K)
        batch_range = torch.arange(B, device=device)  # (B, )
        target_scores[batch_range, left_target_idxs_nocs] = 1.0
        target_scores[batch_range, right_target_idxs_nocs] = 1.0

        cls_metric = nn.BCEWithLogitsLoss(reduction='none')
        loss_grasp_cls = (cls_metric(pred_grasp_score, target_scores).mean(dim=-1) * weights).sum() / valid_count

        if use_xyz_variety_loss:
            # xyz variety loss with nocs-distance as weights
            xyz_metric = torch.nn.HuberLoss(delta=0.1, reduction='none')
            left_target_xyz = xyz_target[:, 0, :].unsqueeze(1).expand(-1, K, -1)  # (B, K, 3)
            right_target_xyz = xyz_target[:, 1, :].unsqueeze(1).expand(-1, K, -1)  # (B, K, 3)

            pred_grasp_nocs_expand = pred_grasp_nocs.unsqueeze(2).expand(-1, -1, N, -1)  # (B, K, N, 3)
            # detach pred_grasp_nocs to avoid gradient flow back to pred_grasp_nocs
            pred_pc_nocs_expand = pred_pc_nocs.detach().unsqueeze(1).expand(-1, K, -1, -1)  # (B, K, N, 3)
            nocs_distance = torch.norm(pred_grasp_nocs_expand - pred_pc_nocs_expand, dim=-1)  # (B, K, N)
            nocs_distance_weight = torch.exp(-alpha * nocs_distance)  # (B, K, N)
            normalized_nocs_distance_weight = nocs_distance_weight / nocs_distance_weight.sum(dim=-1, keepdim=True) + 1e-6 # (B, K, N)
            
            pc_xyz_expand = pc_xyz.unsqueeze(1).expand(-1, K, -1, -1)  # (B, K, N, 3)
            pred_grasp_xyz = torch.sum(pc_xyz_expand * normalized_nocs_distance_weight.unsqueeze(-1), dim=2)  # (B, K, 3)

            left_grasp_loss_xyz = xyz_metric(pred_grasp_xyz, left_target_xyz).mean(dim=-1)  # (B, K)
            right_grasp_loss_xyz = xyz_metric(pred_grasp_xyz, right_target_xyz).mean(dim=-1)  # (B, K)
            batch_range = torch.arange(B, device=device)  # (B, )
            left_variety_loss_xyz = left_grasp_loss_xyz[batch_range, left_target_idxs_nocs]  # (B, )
            right_variety_loss_xyz = right_grasp_loss_xyz[batch_range, right_target_idxs_nocs]  # (B, )
            variety_loss_xyz = (left_variety_loss_xyz + right_variety_loss_xyz) / 2.0  # (B, )
            loss_grasp_variety_xyz = (variety_loss_xyz * weights).sum() / valid_count

            return loss_grasp_variety_nocs, loss_grasp_cls, loss_grasp_variety_xyz
        else:
            return loss_grasp_variety_nocs, loss_grasp_cls, torch.zeros_like(loss_grasp_variety_nocs)

    @staticmethod
    def sym_nocs_huber_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        metric = torch.nn.HuberLoss(delta=0.1, reduction='none')
        sym_target = target.clone()
        # symmetric target: 180 deg rotation around z-axis in NOCS space
        sym_target[:, :, 0] = 1.0 - sym_target[:, :, 0]
        sym_target[:, :, 1] = 1.0 - sym_target[:, :, 1]
        loss = torch.minimum(metric(prediction, target).mean((1, 2)),
                             metric(prediction, sym_target).mean((1, 2))
                             ).mean()
        return loss

    def expand_to(self, x):
        return x.view(-1, 1)

    @staticmethod
    def ranking_loss(pred1: torch.Tensor, pred2: torch.Tensor, gt_score: torch.Tensor) -> torch.Tensor:
        """
        Input:
            pred1: (B, K) Tensor
            pred2: (B, K) Tensor
            gt_score: (B, K, 2) Tensor
        """
        pred_all = torch.stack([pred1, pred2], dim=-1)  # (B, K, 2)
        pred_score_all = F.softmax(pred_all, dim=-1)  # (B, K, 2)
        loss = - (gt_score[:, :, 0] * torch.log(pred_score_all[:, :, 0]) +
                  gt_score[:, :, 1] * torch.log(pred_score_all[:, :, 1]))
        loss = loss.mean()
        return loss

    def forward_real_world_loss(self, input_batch: tuple):
        """Calculate loss from real-world data"""
        coords, feat, pts_xyz, gripper_points, grasp_point_pair1, grasp_point_pair2, grasp_pair_scores_gt, \
            primitive_index_batch, smoothed_score, folding_score = \
            tuple(x.to(self.device) for x in input_batch)
        # network forward
        dense_feat, pred_offset_list, pred_att_list, pred_nocs, pred_grasp_nocs, pred_nocs_grasp_score, \
            dense_nocs_feat_fuse, \
            pred_smoothing_logits, pred_folding_logits, pred_smoothed_logits, pred_is_folding_logits = \
            self.forward(coords, feat, pts_xyz)
        # get reward_dense_feat
        if self.virtual_reward_head.use_nocs_dense_feat:
            reward_dense_feat = dense_nocs_feat_fuse
        else:
            reward_dense_feat = dense_feat
        # calculate virtual reward for two pair groups
        pred_canonicalization_reward_pair1, pred_alignment_reward_pair1, deformable_weight = self.virtual_reward_head(
            pts_xyz.detach(),
            grasp_point_pair1.detach(),
            pred_nocs.detach(),
            reward_dense_feat.detach()) # (B, K, 1)
        pred_canonicalization_reward_pair2, pred_alignment_reward_pair2, deformable_weight = self.virtual_reward_head(
            pts_xyz.detach(),
            grasp_point_pair2.detach(),
            pred_nocs.detach(),
            reward_dense_feat.detach()) # (B, K, 1)
        pred_reward_pair1 = (1 - deformable_weight) * pred_alignment_reward_pair1 + \
                                    deformable_weight * pred_canonicalization_reward_pair1  # (B, K, 1)
        pred_reward_pair2 = (1 - deformable_weight) * pred_alignment_reward_pair2 + \
                                    deformable_weight * pred_canonicalization_reward_pair2  # (B, K, 1)
        # normalize rewards
        if self.virtual_reward_head.enable_reward_normalization:
            num_grasp_pairs = pred_reward_pair1.shape[1]  # K
            pred_reward_cat = torch.cat([pred_reward_pair1, pred_reward_pair2], dim=1)  # (B, 2K, 1)
            pred_normalized_reward_cat = self.virtual_reward_head.reward_bn(
                pred_reward_cat.transpose(1, 2)).transpose(1, 2)  # (B, 2K, 1)
            pred_reward_pair1 = pred_normalized_reward_cat[:, :num_grasp_pairs, :]  # (B, K, 1)
            pred_reward_pair2 = pred_normalized_reward_cat[:, num_grasp_pairs:, :]  # (B, K, 1)
        pred_reward_pair1 = pred_reward_pair1[:, :, 0]  # (B, K)
        pred_reward_pair2 = pred_reward_pair2[:, :, 0]  # (B, K)
        loss_ranking = self.ranking_loss(pred_reward_pair1, pred_reward_pair2, grasp_pair_scores_gt) * self.loss_ranking_weight

        # find prediction action label
        pred_smoothing_action_idx = torch.tensor(self.smoothing_primitive_idxs).to(self.device)[
            torch.argmax(pred_smoothing_logits, dim=-1)]
        pred_folding_action_idx = torch.tensor(self.folding_primitive_idxs).to(self.device)[
            torch.argmax(pred_folding_logits, dim=-1)]
        is_smoothing = folding_score == 0.
        pred_labels = torch.where(is_smoothing, pred_smoothing_action_idx, pred_folding_action_idx)

        # grasp loss for NOCS coordintes of grasp points for fling action
        batch_size = pts_xyz.shape[0]
        weights = torch.zeros(batch_size, device=self.device)  # (B, )
        weights[primitive_index_batch == 0] = 1.0
        loss_grasp_variety_nocs, loss_grasp_cls, loss_grasp_variety_xyz = \
            self.sym_grasp_huber_variety_cls_loss(pred_grasp_nocs,
                                                  pred_nocs_grasp_score,
                                                  pred_nocs,
                                                  pts_xyz,
                                                  gripper_points[:, 0:3:2, :],
                                                  None,
                                                  weights=weights,
                                                  use_xyz_variety_loss=True,
                                                  alpha=self.action_head.nocs_distance_weight_alpha)
        loss_grasp_variety_xyz *= self.loss_grasp_weight * 0.1
        loss_grasp_variety_nocs *= self.loss_grasp_weight
        loss_grasp_cls *= self.loss_cls_weight
        # calculate grasp prediction error
        num_pts = pred_nocs.shape[1]  # N
        num_candidates = self.action_head.num_pred_fling_candidates  # K

        pred_grasp_nocs_expand = pred_grasp_nocs.detach().unsqueeze(1).expand(-1, num_pts, -1, -1)  # (B, N, K, 3)
        pred_nocs_expand = pred_nocs.detach().unsqueeze(2).expand(-1, -1, num_candidates, -1)  # (B, N, K, 3)
        grasp_nocs_dist = torch.norm(pred_grasp_nocs_expand - pred_nocs_expand, dim=-1)  # (B, N, K)
        min_idxs = torch.argmin(grasp_nocs_dist, dim=1).unsqueeze(-1).expand(-1, -1, 2)  # (B, K, 2)
        pred_grasp_point = torch.gather(pts_xyz[:, :, :2], 1, min_idxs)  # (B, K, 2) we only care about x,y coordinates
        target_grasp_point = gripper_points[:, 0:3:2, :2]  # (B, 2, 2) we only care about x,y coordinates
        left_target_grasp_point = target_grasp_point[:, 0, :].unsqueeze(1).expand(-1, num_candidates, -1)  # (B, K, 2)
        right_target_grasp_point = target_grasp_point[:, 1, :].unsqueeze(1).expand(-1, num_candidates, -1)  # (B, K, 2)
        valid_count = max(weights.sum().item(), 1.0)
        fling_grasp_err = ((torch.norm(pred_grasp_point - left_target_grasp_point, dim=-1).min(dim=1)[0] +
                            torch.norm(pred_grasp_point - right_target_grasp_point, dim=-1).min(dim=1)[0])
                           / 2.0 * weights).sum() / valid_count

        # offset loss for grasp and release points (other actions)
        loss_grasp_offset_list = [None]
        loss_release_offset_list = [None]
        grasp_point_err_list = [None]
        release_point_err_list = [None]
        for action_idx in range(len(pred_offset_list)):
            if action_idx == 0:
                continue
            pc_xy = pts_xyz[:, :, :2]  # (B, N, 2)
            for tag in ('pick', 'place'):
                if tag == 'pick':  # pick
                    pred_offset1 = pred_offset_list[action_idx][:, :,
                                   :2]  # (B, N, 2) we only care about (x, y) coordniates
                    pred_offset2 = pred_offset_list[action_idx][:, :,
                                   2:4]  # (B, N, 2) we only care about (x, y) coordniates
                    pred_att_prob1 = F.softmax(pred_att_list[action_idx][:, :, 0].unsqueeze(-1), dim=1)  # (B, N, 1)
                    pred_att_prob2 = F.softmax(pred_att_list[action_idx][:, :, 1].unsqueeze(-1), dim=1)  # (B, N, 1)

                    pred_point1 = ((pred_offset1 + pc_xy) * pred_att_prob1).sum(dim=1)  # (B, 2)
                    pred_point2 = ((pred_offset2 + pc_xy) * pred_att_prob2).sum(dim=1)  # (B, 2)
                    gt_point1, gt_point2 = gripper_points[:, 0, :2], gripper_points[:, 2, :2]  # (B, 2)
                else:  # place
                    pred_offset1 = pred_offset_list[action_idx][:, :,
                                   4:6]  # (B, N, 2) we only care about (x, y) coordniates
                    pred_offset2 = pred_offset_list[action_idx][:, :,
                                   6:8]  # (B, N, 2) we only care about (x, y) coordniates
                    pred_att_prob1 = F.softmax(pred_att_list[action_idx][:, :, 2].unsqueeze(-1), dim=1)  # (B, N, 1)
                    pred_att_prob2 = F.softmax(pred_att_list[action_idx][:, :, 3].unsqueeze(-1), dim=1)  # (B, N, 1)

                    pred_point1 = ((pred_offset1 + pc_xy) * pred_att_prob1).sum(dim=1)  # (B, 2)
                    pred_point2 = ((pred_offset2 + pc_xy) * pred_att_prob2).sum(dim=1)  # (B, 2)
                    gt_point1, gt_point2 = gripper_points[:, 1, :2], gripper_points[:, 3, :2]  # (B, 2)

                weights = torch.zeros_like(pred_point1)
                weights[primitive_index_batch == action_idx, :] = 1.0
                valid_count = max(weights[:, 0].sum().item(), 1.0)
                loss_offset1 = (nn.MSELoss(reduction='none')(pred_point1, gt_point1) * weights).sum() / valid_count
                loss_offset2 = (nn.MSELoss(reduction='none')(pred_point2, gt_point2) * weights).sum() / valid_count
                loss_offset = (loss_offset1 + loss_offset2) / 2 * self.loss_offset_weight
                if tag == 'pick':
                    loss_grasp_offset_list.append(loss_offset)
                else:
                    loss_release_offset_list.append(loss_offset)

                err1 = (torch.norm(pred_point1 - gt_point1, dim=-1) * weights[:, 0]).sum() / valid_count
                err2 = (torch.norm(pred_point2 - gt_point2, dim=-1) * weights[:, 0]).sum() / valid_count
                err = (err1 + err2) / 2
                if tag == 'pick':
                    grasp_point_err_list.append(err)
                else:
                    release_point_err_list.append(err)

        # classification loss: smoothing action
        smoothing_action_idxs = []
        smoothing_bs_weight = []
        for bs_idx, action_idx in enumerate(primitive_index_batch):
            if is_smoothing[bs_idx] == 1 and action_idx.item() in self.smoothing_primitive_idxs:
                smoothing_action_idxs.append(self.smoothing_primitive_idxs.index(action_idx.item()))
                smoothing_bs_weight.append(1.0)
            else:
                smoothing_action_idxs.append(0)
                smoothing_bs_weight.append(0.)
        smoothing_action_idxs = torch.tensor(smoothing_action_idxs, device=self.device)
        smoothing_bs_weight = torch.tensor(smoothing_bs_weight, device=self.device)
        loss_smoothing_cls = self.loss_cls_weight * (nn.CrossEntropyLoss(
            weight=self.smoothing_cls_weight.to(self.device), reduction='none')(
            pred_smoothing_logits, smoothing_action_idxs) * smoothing_bs_weight).sum() / \
                             torch.maximum(smoothing_bs_weight.sum(), torch.ones_like(smoothing_bs_weight.sum()))

        # classification loss: is_smoothed
        loss_smoothed_score = self.loss_cls_weight * (nn.BCEWithLogitsLoss(
            reduction='none')(pred_smoothed_logits, smoothed_score.unsqueeze(-1))).mean()
        # classification loss: is_folding
        loss_is_folding = self.loss_cls_weight * (nn.BCEWithLogitsLoss(
            reduction='none')(pred_is_folding_logits, folding_score.unsqueeze(-1))).mean()
        if self.ignore_folding_cls:
            loss_is_folding = torch.zeros_like(loss_is_folding, requires_grad=True)

        # find predicted classification labels
        target_smoothed = smoothed_score == 1
        pred_smoothed = (torch.sigmoid(pred_smoothed_logits) > 0.5)[:, 0]
        target_is_folding = folding_score == 1
        pred_is_folding = (torch.sigmoid(pred_is_folding_logits) > 0.5)[:, 0]

        # summarize loss and errors
        valid_action_num = len(loss_grasp_offset_list)
        loss_dict = dict()
        for action_idx in range(1, valid_action_num):
            loss_dict['loss_grasp_offset_{}'.format(self.primitives[action_idx])] = loss_grasp_offset_list[action_idx]
            loss_dict['loss_release_offset_{}'.format(self.primitives[action_idx])] = loss_release_offset_list[action_idx]
            loss_dict['err_{}_grasp'.format(self.primitives[action_idx])] = grasp_point_err_list[action_idx]
            loss_dict['err_{}_release'.format(self.primitives[action_idx])] = release_point_err_list[action_idx]
        loss_dict.update({'loss_ranking': loss_ranking,
                          'loss_grasp_variety_nocs': loss_grasp_variety_nocs,
                          'loss_grasp_variety_xyz': loss_grasp_variety_xyz,
                          'loss_grasp_cls': loss_grasp_cls,
                          'err_fling_grasp': fling_grasp_err,
                          'loss_smoothing_cls': loss_smoothing_cls,
                          'loss_smoothed': loss_smoothed_score,
                          'loss_is_folding': loss_is_folding,
                          "accuracy": (pred_labels == primitive_index_batch).float().mean(),
                          "smoothed_accuracy": (pred_smoothed == target_smoothed).float().mean(),
                          "is_folding_accuracy": (pred_is_folding == target_is_folding).float().mean(),
                          'lr': self.optimizers().optimizer.param_groups[0]['lr'],
                          })
        loss_all = torch.zeros_like(loss_dict['loss_grasp_variety_xyz'])
        for key, item in loss_dict.items():
            if 'loss' in key:
                loss_all = loss_all + item

        loss_dict['real-world_loss'] = loss_all
        return loss_dict

    def forward_virtual_reward_loss(self, input_batch: tuple):
        """
        calculate virtual reward loss for a pair of grasp points
        """
        coords, feat, pts_xyz_torch, pts_nocs_torch, gripper_points, grasp_points_nocs_torch, valid_poses_weight, \
            gt_canonicalization_reward, gt_alignment_reward = \
            tuple(x.to(self.device) for x in input_batch)
        # network forward
        dense_feat, pred_offset_list, pred_att_list, pred_nocs, pred_grasp_nocs, pred_nocs_grasp_score, \
            dense_nocs_feat_fuse, \
            pred_smoothing_logits, pred_folding_logits, pred_smoothed_logits, pred_is_folding_logits = \
            self.forward(coords, feat, pts_xyz_torch)

        # grasp loss for NOCS coordintes of grasp points for fling action
        assert len(valid_poses_weight.shape) == 1
        loss_grasp_variety_nocs, loss_grasp_cls, loss_grasp_variety_xyz = \
            self.sym_grasp_huber_variety_cls_loss(pred_grasp_nocs,
                                                  pred_nocs_grasp_score,
                                                  pts_nocs_torch if self.action_head.use_gt_nocs_pred_for_distance_weight else pred_nocs,
                                                  pts_xyz_torch,
                                                  gripper_points[:, 0:3:2, :],
                                                  grasp_points_nocs_torch,
                                                  weights=valid_poses_weight,
                                                  use_xyz_variety_loss=self.action_head.use_xyz_variety_loss,
                                                  alpha=self.action_head.nocs_distance_weight_alpha)
        loss_grasp_variety_xyz *= self.loss_grasp_weight * self.virtual_grasp_loss_weight_factor
        loss_grasp_variety_nocs *= self.loss_grasp_weight * self.virtual_grasp_loss_weight_factor
        loss_grasp_cls *= self.loss_cls_weight * self.virtual_grasp_loss_weight_factor
        # calculate grasp prediction error
        num_pts = pred_nocs.shape[1]  # N
        num_candidates = self.action_head.num_pred_fling_candidates  # K

        pred_grasp_nocs_expand = pred_grasp_nocs.detach().unsqueeze(1).expand(-1, num_pts, -1, -1)  # (B, N, K, 3)
        pred_nocs_expand = pred_nocs.detach().unsqueeze(2).expand(-1, -1, num_candidates, -1)  # (B, N, K, 3)
        grasp_nocs_dist = torch.norm(pred_grasp_nocs_expand - pred_nocs_expand, dim=-1)  # (B, N, K)
        min_idxs = torch.argmin(grasp_nocs_dist, dim=1).unsqueeze(-1).expand(-1, -1, 2)  # (B, K, 2)
        pred_grasp_point = torch.gather(pts_xyz_torch[:, :, :2], 1, min_idxs)  # (B, K, 2) we only care about x,y coordinates
        target_grasp_point = gripper_points[:, 0:3:2, :2]  # (B, 2, 2) we only care about x,y coordinates
        left_target_grasp_point = target_grasp_point[:, 0, :].unsqueeze(1).expand(-1, num_candidates, -1)  # (B, K, 2)
        right_target_grasp_point = target_grasp_point[:, 1, :].unsqueeze(1).expand(-1, num_candidates, -1)  # (B, K, 2)
        valid_count = max(valid_poses_weight.sum().item(), 1.0)
        fling_grasp_err = ((torch.norm(pred_grasp_point - left_target_grasp_point, dim=-1).min(dim=1)[0] +
                            torch.norm(pred_grasp_point - right_target_grasp_point, dim=-1).min(dim=1)[0])
                           / 2.0 * valid_poses_weight).sum() / valid_count

        # get pick points
        pick_pts1 = gripper_points[:, 0, :3].unsqueeze(1)  # (B, 1, 3)
        pick_pts2 = gripper_points[:, 2, :3].unsqueeze(1)  # (B, 1, 3)
        pick_pts_xyz = torch.stack([pick_pts1, pick_pts2], dim=1)  # (B, 2, 1, 3)
        if self.virtual_reward_head.use_nocs_dense_feat:
            dense_feat = dense_nocs_feat_fuse
        else:
            dense_feat = dense_feat
        # nocs loss
        loss_nocs = self.loss_nocs_weight * self.sym_nocs_huber_loss(pred_nocs, pts_nocs_torch) * self.virtual_nocs_loss_weight_factor
        # calculate virtual reward
        pred_canonicalization_reward, pred_alignment_reward, deformable_weight = self.virtual_reward_head(
            pts_xyz_torch.detach(),
            pick_pts_xyz.detach(),
            pred_nocs.detach(),
            dense_feat)
        # calculate loss
        loss_canonicalization_reward = F.smooth_l1_loss(pred_canonicalization_reward,
                                                        gt_canonicalization_reward.unsqueeze(1).unsqueeze(1)) \
                                       * self.loss_canonicalization_reward_weight
        loss_alignment_reward = F.smooth_l1_loss(pred_alignment_reward, gt_alignment_reward.unsqueeze(1).unsqueeze(1)) \
                                * self.loss_alignment_reward_weight
        # generate loss dict
        loss_dict = dict(loss_canonicalization_reward=loss_canonicalization_reward,
                         loss_alignment_reward=loss_alignment_reward,
                         loss_grasp_variety_xyz=loss_grasp_variety_xyz,
                         loss_grasp_variety_nocs=loss_grasp_variety_nocs,
                         loss_grasp_cls=loss_grasp_cls,
                         err_fling_grasp=fling_grasp_err,
                         loss_nocs=loss_nocs,
                         lr=self.optimizers().optimizer.param_groups[0]['lr'])
        loss_all = torch.zeros_like(loss_dict['loss_canonicalization_reward'])
        for key, item in loss_dict.items():
            if 'loss' in key:
                loss_all = loss_all + item
        loss_dict['virtual-reward_loss'] = loss_all
        return loss_dict

    def forward_normal_loss(self, input_batch: tuple):
        coords, feat, pc_xyz, pc_nocs, gripper_points, grasp_points_nocs, rotation_cls, \
            reward, primitive_index_batch, smoothed_score, folding_score, folding_step = \
            tuple(x.to(self.device) for x in input_batch)
        # network forward
        dense_feat, pred_offset_list, pred_att_list, pred_nocs, pred_grasp_nocs, pred_nocs_grasp_score, \
            dense_nocs_feat_fuse, \
            pred_smoothing_logits, pred_folding_logits, pred_smoothed_logits, pred_is_folding_logits = \
            self.forward(coords, feat, pc_xyz, gt_pc_nocs=pc_nocs)
        # find prediction action label
        pred_smoothing_action_idx = torch.tensor(self.smoothing_primitive_idxs).to(self.device)[
            torch.argmax(pred_smoothing_logits, dim=-1)]
        pred_folding_action_idx = torch.tensor(self.folding_primitive_idxs).to(self.device)[
            torch.argmax(pred_folding_logits, dim=-1)]
        is_smoothing = folding_score == 0.
        pred_labels = torch.where(is_smoothing, pred_smoothing_action_idx, pred_folding_action_idx)

        # grasp loss for NOCS coordintes of grasp points for fling action
        weights = torch.zeros(self.batch_size, device=self.device)  # (B, )
        weights[primitive_index_batch == 0] = 1.0
        loss_grasp_variety_nocs, loss_grasp_cls, loss_grasp_variety_xyz = \
            self.sym_grasp_huber_variety_cls_loss(pred_grasp_nocs,
                                                  pred_nocs_grasp_score,
                                                  pc_nocs if self.action_head.use_gt_nocs_pred_for_distance_weight else pred_nocs,
                                                  pc_xyz,
                                                  gripper_points[:, 0:3:2, :],
                                                  grasp_points_nocs,
                                                  weights=weights,
                                                  use_xyz_variety_loss=self.action_head.use_xyz_variety_loss,
                                                  alpha=self.action_head.nocs_distance_weight_alpha)
        loss_grasp_variety_xyz *= self.loss_grasp_weight
        loss_grasp_variety_nocs *= self.loss_grasp_weight
        loss_grasp_cls *= self.loss_cls_weight
        # calculate grasp prediction error
        num_pts = pred_nocs.shape[1]  # N
        num_candidates = self.action_head.num_pred_fling_candidates  # K

        pred_grasp_nocs_expand = pred_grasp_nocs.detach().unsqueeze(1).expand(-1, num_pts, -1, -1)  # (B, N, K, 3)
        pred_nocs_expand = pred_nocs.detach().unsqueeze(2).expand(-1, -1, num_candidates, -1)  # (B, N, K, 3)
        grasp_nocs_dist = torch.norm(pred_grasp_nocs_expand - pred_nocs_expand, dim=-1)  # (B, N, K)
        min_idxs = torch.argmin(grasp_nocs_dist, dim=1).unsqueeze(-1).expand(-1, -1, 2)  # (B, K, 2)
        pred_grasp_point = torch.gather(pc_xyz[:, :, :2], 1, min_idxs)  # (B, K, 2) we only care about x,y coordinates
        target_grasp_point = gripper_points[:, 0:3:2, :2]  # (B, 2, 2) we only care about x,y coordinates
        left_target_grasp_point = target_grasp_point[:, 0, :].unsqueeze(1).expand(-1, num_candidates, -1)  # (B, K, 2)
        right_target_grasp_point = target_grasp_point[:, 1, :].unsqueeze(1).expand(-1, num_candidates, -1)  # (B, K, 2)
        valid_count = max(weights.sum().item(), 1.0)
        fling_grasp_err = ((torch.norm(pred_grasp_point - left_target_grasp_point, dim=-1).min(dim=1)[0] +
                            torch.norm(pred_grasp_point - right_target_grasp_point, dim=-1).min(dim=1)[0])
                           / 2.0 * weights).sum() / valid_count

        # offset loss for grasp and release points (other actions)
        loss_grasp_offset_list = [None]
        loss_release_offset_list = [None]
        grasp_point_err_list = [None]
        release_point_err_list = [None]
        for action_idx in range(len(pred_offset_list)):
            if action_idx == 0:
                continue
            pc_xy = pc_xyz[:, :, :2]  # (B, N, 2)
            for tag in ('pick', 'place'):
                if tag == 'pick':  # pick
                    pred_offset1 = pred_offset_list[action_idx][:, :, :2]  # (B, N, 2) we only care about (x, y) coordniates
                    pred_offset2 = pred_offset_list[action_idx][:, :, 2:4]  # (B, N, 2) we only care about (x, y) coordniates
                    pred_att_prob1 = F.softmax(pred_att_list[action_idx][:, :, 0].unsqueeze(-1), dim=1)  # (B, N, 1)
                    pred_att_prob2 = F.softmax(pred_att_list[action_idx][:, :, 1].unsqueeze(-1), dim=1)  # (B, N, 1)

                    pred_point1 = ((pred_offset1 + pc_xy) * pred_att_prob1).sum(dim=1)  # (B, 2)
                    pred_point2 = ((pred_offset2 + pc_xy) * pred_att_prob2).sum(dim=1)  # (B, 2)
                    gt_point1, gt_point2 = gripper_points[:, 0, :2], gripper_points[:, 2, :2]  # (B, 2)
                else:  # place
                    pred_offset1 = pred_offset_list[action_idx][:, :, 4:6]  # (B, N, 2) we only care about (x, y) coordniates
                    pred_offset2 = pred_offset_list[action_idx][:, :, 6:8]  # (B, N, 2) we only care about (x, y) coordniates
                    pred_att_prob1 = F.softmax(pred_att_list[action_idx][:, :, 2].unsqueeze(-1), dim=1)  # (B, N, 1)
                    pred_att_prob2 = F.softmax(pred_att_list[action_idx][:, :, 3].unsqueeze(-1), dim=1)  # (B, N, 1)

                    pred_point1 = ((pred_offset1 + pc_xy) * pred_att_prob1).sum(dim=1)  # (B, 2)
                    pred_point2 = ((pred_offset2 + pc_xy) * pred_att_prob2).sum(dim=1)  # (B, 2)
                    gt_point1, gt_point2 = gripper_points[:, 1, :2], gripper_points[:, 3, :2]  # (B, 2)

                weights = torch.zeros_like(pred_point1)
                weights[primitive_index_batch == action_idx, :] = 1.0
                valid_count = max(weights[:, 0].sum().item(), 1.0)
                loss_offset1 = (nn.MSELoss(reduction='none')(pred_point1, gt_point1) * weights).sum() / valid_count
                loss_offset2 = (nn.MSELoss(reduction='none')(pred_point2, gt_point2) * weights).sum() / valid_count
                loss_offset = (loss_offset1 + loss_offset2) / 2 * self.loss_offset_weight
                if tag == 'pick':
                    loss_grasp_offset_list.append(loss_offset)
                else:
                    loss_release_offset_list.append(loss_offset)

                err1 = (torch.norm(pred_point1 - gt_point1, dim=-1) * weights[:, 0]).sum() / valid_count
                err2 = (torch.norm(pred_point2 - gt_point2, dim=-1) * weights[:, 0]).sum() / valid_count
                err = (err1 + err2) / 2
                if tag == 'pick':
                    grasp_point_err_list.append(err)
                else:
                    release_point_err_list.append(err)

        # classification loss: smoothing action
        smoothing_action_idxs = []
        smoothing_bs_weight = []
        for bs_idx, action_idx in enumerate(primitive_index_batch):
            if is_smoothing[bs_idx] == 1 and action_idx.item() in self.smoothing_primitive_idxs:
                smoothing_action_idxs.append(self.smoothing_primitive_idxs.index(action_idx.item()))
                smoothing_bs_weight.append(1.0)
            else:
                smoothing_action_idxs.append(0)
                smoothing_bs_weight.append(0.)
        smoothing_action_idxs = torch.tensor(smoothing_action_idxs, device=self.device)
        smoothing_bs_weight = torch.tensor(smoothing_bs_weight, device=self.device)
        loss_smoothing_cls = self.loss_cls_weight * (nn.CrossEntropyLoss(
            weight=self.smoothing_cls_weight.to(self.device), reduction='none')(
            pred_smoothing_logits, smoothing_action_idxs) * smoothing_bs_weight).sum() / \
                             torch.maximum(smoothing_bs_weight.sum(), torch.ones_like(smoothing_bs_weight.sum()))

        # classification loss: folding action
        folding_action_idxs = []
        folding_bs_weight = []
        for bs_idx, action_idx in enumerate(primitive_index_batch):
            if is_smoothing[bs_idx] == 0:
                folding_action_idxs.append(self.folding_primitive_idxs.index(action_idx.item()))
                folding_bs_weight.append(1.0)
            else:
                folding_action_idxs.append(0)
                folding_bs_weight.append(0.)
        folding_action_idxs = torch.tensor(folding_action_idxs, device=self.device)
        folding_bs_weight = torch.tensor(folding_bs_weight, device=self.device)
        loss_folding_cls = self.loss_cls_weight * (nn.CrossEntropyLoss(
            weight=self.folding_cls_weight.to(self.device), reduction='none')(
            pred_folding_logits, folding_action_idxs) * folding_bs_weight).sum() / \
                           torch.maximum(folding_bs_weight.sum(), torch.ones_like(folding_bs_weight.sum()))
        if self.ignore_folding_cls:
            loss_folding_cls = torch.zeros_like(loss_folding_cls, requires_grad=True)

        # classification loss: is_smoothed
        loss_smoothed_score = self.loss_cls_weight * (nn.BCEWithLogitsLoss(
            reduction='none')(pred_smoothed_logits, smoothed_score.unsqueeze(-1))).mean()
        # classification loss: is_folding
        loss_is_folding = self.loss_cls_weight * (nn.BCEWithLogitsLoss(
            reduction='none')(pred_is_folding_logits, folding_score.unsqueeze(-1))).mean()
        if self.ignore_folding_cls:
            loss_is_folding = torch.zeros_like(loss_is_folding, requires_grad=True)
        # nocs loss
        loss_nocs = self.loss_nocs_weight * self.sym_nocs_huber_loss(pred_nocs, pc_nocs)

        # find predicted classification labels
        target_smoothed = smoothed_score == 1
        pred_smoothed = (torch.sigmoid(pred_smoothed_logits) > 0.5)[:, 0]
        target_is_folding = folding_score == 1
        pred_is_folding = (torch.sigmoid(pred_is_folding_logits) > 0.5)[:, 0]

        # summarize loss and errors
        valid_action_num = len(loss_grasp_offset_list)
        loss_dict = dict()
        for action_idx in range(1, valid_action_num):
            loss_dict['loss_grasp_offset_{}'.format(self.primitives[action_idx])] = loss_grasp_offset_list[action_idx]
            loss_dict['loss_release_offset_{}'.format(self.primitives[action_idx])] = loss_release_offset_list[action_idx]
            loss_dict['err_{}_grasp'.format(self.primitives[action_idx])] = grasp_point_err_list[action_idx]
            loss_dict['err_{}_release'.format(self.primitives[action_idx])] = release_point_err_list[action_idx]
        loss_dict.update({'loss_grasp_variety_nocs': loss_grasp_variety_nocs,
                          'loss_grasp_variety_xyz': loss_grasp_variety_xyz,
                          'loss_grasp_cls': loss_grasp_cls,
                          'err_fling_grasp': fling_grasp_err,
                          'loss_smoothing_cls': loss_smoothing_cls,
                          'loss_folding_cls': loss_folding_cls,
                          'loss_smoothed': loss_smoothed_score,
                          'loss_is_folding': loss_is_folding,
                          'loss_nocs': loss_nocs,
                          "accuracy": (pred_labels == primitive_index_batch).float().mean(),
                          "smoothed_accuracy": (pred_smoothed == target_smoothed).float().mean(),
                          "is_folding_accuracy": (pred_is_folding == target_is_folding).float().mean(),
                          'lr': self.optimizers().optimizer.param_groups[0]['lr'],
                          })
        loss_all = torch.zeros_like(loss_dict['loss_nocs'])
        for key, item in loss_dict.items():
            if 'loss' in key:
                loss_all = loss_all + item
        loss_dict['normal_loss'] = loss_all
        return loss_dict

    def predict_raw_action_type(self,
                                pc_xyz_batch: torch.Tensor,
                                coords: torch.Tensor,
                                feat: torch.Tensor,
                                only_fling_during_smoothing: bool = True,
                                smoothed_cls_thr: float = 0.1) -> ActionTypeDef:
        """
        Predict raw action type by the action classifier, this action type could be changed in the post-processing if required
        """
        if not torch.is_tensor(coords) and not torch.is_tensor(feat):
            coords = torch.from_numpy(coords).to(device=self.device)
            feat = torch.from_numpy(feat).to(device=self.device)

        # network forward
        dense_feat, pred_offset_list, pred_att_list, pred_nocs, pred_grasp_nocs, pred_grasp_score, \
            dense_nocs_feat_fuse, \
            pred_smoothing_logits, pred_folding_logits, pred_smoothed_logits, pred_is_folding_logits = \
            self.forward(coords, feat, pc_xyz_batch)

        is_smoothing = False
        logger.debug(
            f'pred_is_folding_score : {torch.sigmoid(pred_is_folding_logits)}, pred_smoothed_score: {torch.sigmoid(pred_smoothed_logits)}')
        if torch.sigmoid(pred_smoothed_logits) <= smoothed_cls_thr:
            # not fully smoothed yet
            if only_fling_during_smoothing:
                # override fling action
                action_idx = self.smoothing_primitive_idxs[0]
            else:
                action_idx = self.smoothing_primitive_idxs[torch.argmax(pred_smoothing_logits).item()]
        else:
            # fully smoothed, predict fold1 action directly
            action_idx = self.folding_primitive_idxs[1]  # fold1
        action_str = self.primitives[action_idx]
        action_type = ActionTypeDef.from_string(action_str)
        return action_type

    def predict(self, pc_xyz_batch: torch.Tensor,
                coords: torch.Tensor,
                feat: torch.Tensor,
                action_type: ActionTypeDef,
                return_timing=False):
        timing = {}
        start = time.time()
        if not torch.is_tensor(coords) and not torch.is_tensor(feat):
            coords = torch.from_numpy(coords).to(device=self.device)
            feat = torch.from_numpy(feat).to(device=self.device)

        pre_processing = time.time()
        timing['pre_processing'] = pre_processing - start

        # network forward
        dense_feat, pred_offset_list, pred_att_list, pred_nocs, pred_grasp_nocs, pred_grasp_score, \
            dense_nocs_feat_fuse, \
            pred_smoothing_logits, pred_folding_logits, pred_smoothed_logits, pred_is_folding_logits = \
            self.forward(coords, feat, pc_xyz_batch)

        nn_inference = time.time()
        timing['nn_inference'] = nn_inference - pre_processing

        assert action_type is not None, 'action type must not be NONE！'
        action_str = ActionTypeDef.to_string(action_type)
        if action_str in self.primitives:
            action_idx = self.primitives.index(action_str)
        else:
            logger.warning(f'Action {action_str} does not belong to primitives defined in the AI model!')
            # TODO: more flexible
            action_idx = -1  # for compatibility

        att_map_raw = {
            'fling_pick': pred_att_list[0][:, :, :, 0],  # (B, N, K)
            'drag_pick1': pred_att_list[1][:, :, 0],  # (B, N)
            'drag_pick2': pred_att_list[1][:, :, 1],  # (B, N)
            'drag_place1': pred_att_list[1][:, :, 2],  # (B, N)
            'drag_place2': pred_att_list[1][:, :, 3],  # (B, N)
            'fold1_pick1': pred_att_list[2][:, :, 0],  # (B, N)
            'fold1_pick2': pred_att_list[2][:, :, 1],  # (B, N)
            'fold1_place1': pred_att_list[2][:, :, 2],  # (B, N)
            'fold1_place2': pred_att_list[2][:, :, 3],  # (B, N)
            'fold2_pick1': pred_att_list[3][:, :, 0],  # (B, N)
            'fold2_pick2': pred_att_list[3][:, :, 1],  # (B, N)
            'fold2_place1': pred_att_list[3][:, :, 2],  # (B, N)
            'fold2_place2': pred_att_list[3][:, :, 3],  # (B, N)
        }
        if len(pred_att_list) > 4:
            att_map_raw.update({
                'pnp_pick1': pred_att_list[4][:, :, 0],  # (B, N)
                'pnp_pick2': pred_att_list[4][:, :, 1],  # (B, N)
                'pnp_place1': pred_att_list[4][:, :, 2],  # (B, N)
                'pnp_place2': pred_att_list[4][:, :, 3],  # (B, N)
            })
        nocs_map = pred_nocs[0].detach().cpu().numpy()  # (N, 3)
        attmaps = {key: F.softmax(value, dim=1).detach().cpu().numpy()[0] for key, value in att_map_raw.items()}

        grasp_point_all = np.zeros((0, 3)).astype(np.float32)
        grasp_point_nocs_all = np.zeros((0, 3)).astype(np.float32)
        virtual_reward_all = np.zeros((0, 0, 1)).astype(np.float32)
        real_reward_all = np.zeros((0, 0, 1)).astype(np.float32)

        # always handle fling predictions for whatever action type
        num_pts = pred_nocs.shape[1]  # N
        num_candidates = self.action_head.num_pred_fling_candidates  # K

        pred_grasp_nocs_expand = pred_grasp_nocs.detach()[0].unsqueeze(0).expand(num_pts, -1, -1)  # (N, K, 3)
        pred_nocs_expand = pred_nocs.detach()[0].unsqueeze(1).expand(-1, num_candidates, -1)  # (N, K, 3)
        grasp_nocs_dist = torch.norm(pred_grasp_nocs_expand - pred_nocs_expand, dim=-1)  # (N, K)
        min_idxs = torch.argmin(grasp_nocs_dist, dim=0).unsqueeze(-1).expand(-1, 3)  # (K, 3)
        pred_grasp_point_torch = torch.gather(pc_xyz_batch[0], 0, min_idxs)  # (K, 3)

        grasp_point_all = pred_grasp_point_torch.cpu().numpy()  # (K, 3)
        release_point_all = np.zeros_like(grasp_point_all)  # (K, 3)
        pred_grasp_point_nocs = torch.gather(pred_nocs[0], 0, min_idxs)  # (K, 3)
        grasp_point_nocs_all = pred_grasp_point_nocs.cpu().numpy()  # (K, 3)

        if self.use_virtual_reward_for_inference:
            # generate a Tensor with shape (2, K*K, 3) with stands for K*K pairs of possible grasp points from grasp_point_all
            pred_grasp_point_expand1 = pred_grasp_point_torch.unsqueeze(1).expand(-1, num_candidates, -1)  # (K, K, 3)
            pred_grasp_point_expand2 = pred_grasp_point_torch.unsqueeze(0).expand(num_candidates, -1, -1)  # (K, K, 3)
            grasp_point_all_pairs = torch.stack([pred_grasp_point_expand1, pred_grasp_point_expand2], dim=0)  # (2, K, K, 3)
            grasp_point_all_pairs_reshape = grasp_point_all_pairs.reshape(1, 2, -1, 3)  # (1, 2, K*K, 3)
            # calculate reward for each pair of points
            use_nocs_dense_feat = self.virtual_reward_head.use_nocs_dense_feat
            canonicalization_reward, alignment_reward, deformable_weight = self.virtual_reward_head(pc_xyz_batch,
                                                                                 grasp_point_all_pairs_reshape,
                                                                                 pred_nocs,
                                                                                 dense_feat if not use_nocs_dense_feat
                                                                                 else dense_nocs_feat_fuse)  # (B, K*K, 1)
            canonicalization_reward = canonicalization_reward[0, :, 0].reshape(num_candidates, num_candidates)  # (K, K)
            alignment_reward = alignment_reward[0, :, 0].reshape(num_candidates, num_candidates)  # (K, K)
            deformable_weight = deformable_weight[0, :, 0].reshape(num_candidates, num_candidates)  # (K, K)
            unfactorized_reward = (1 - deformable_weight) * alignment_reward + \
                                    deformable_weight * canonicalization_reward  # (K K)
            virtual_reward_all = unfactorized_reward.unsqueeze(-1)  # (K, K, 1)
            pair_score_all = unfactorized_reward  # (K, K)
        else:
            single_score = torch.sigmoid(pred_grasp_score)[0]  # (K, )
            single_score_expand1 = single_score.unsqueeze(1).expand(-1, num_candidates)  # (K, K)
            single_score_expand2 = single_score.unsqueeze(0).expand(num_candidates, -1)  # (K, K)
            pair_score_all = single_score_expand1 * single_score_expand2  # (K, K)
            virtual_reward_all = pair_score_all.unsqueeze(1)  # (K, K, 1)
        real_reward_all = torch.zeros_like(virtual_reward_all)
        # make sure that the same point is not selected
        self_idxs = np.arange(num_candidates)
        pair_score_all[self_idxs, self_idxs] = -100.0
        flatten_pair_score = pair_score_all.view(-1)  # (K*K, )
        flatten_pair_score_numpy = flatten_pair_score.cpu().numpy()  # (K*K, )
        sorted_pair_idxs = np.argsort(flatten_pair_score_numpy)[::-1]  # (K*K, )
        virtual_reward_all = virtual_reward_all.cpu().numpy()  # (K, K, 1)
        real_reward_all = real_reward_all.cpu().numpy()  # (K, K, 1)

        if action_type != ActionTypeDef.DONE and action_type != ActionTypeDef.FLING:
            # use attention + offset to predict grasp and release points
            for tag in ('pick', 'place'):
                if tag == 'pick':  # pick
                    att1 = pred_att_list[action_idx][0, :, 0].detach().unsqueeze(-1)  # (N, 1)
                    att2 = pred_att_list[action_idx][0, :, 1].detach().unsqueeze(-1)  # (N, 1)
                    att_prob1 = F.softmax(att1, dim=0)  # (N, 1)
                    att_prob2 = F.softmax(att2, dim=0)  # (N, 1)
                    offset1 = pred_offset_list[action_idx][0, :, :2].detach()  # (N, 2)
                    offset2 = pred_offset_list[action_idx][0, :, 2:4].detach()  # (N, 2)
                    pc_xy = pc_xyz_batch[0, :, :2]  # (N, 2)
                    pred_point1_xy = ((offset1 + pc_xy) * att_prob1).sum(dim=0)  # (2, )
                    pred_point2_xy = ((offset2 + pc_xy) * att_prob2).sum(dim=0)  # (2, )
                    # find the nearest point for grasping on input point cloud
                    nearest_pc_idx1 = torch.argmin(torch.norm(pred_point1_xy.unsqueeze(0) - pc_xy, dim=1))
                    nearest_pc_idx2 = torch.argmin(torch.norm(pred_point2_xy.unsqueeze(0) - pc_xy, dim=1))
                    grasp_point1 = pc_xyz_batch[0, nearest_pc_idx1, :].cpu().numpy()  # (3, )
                    grasp_point2 = pc_xyz_batch[0, nearest_pc_idx2, :].cpu().numpy()  # (3, )
                else:  # place
                    att1 = pred_att_list[action_idx][0, :, 2].detach().unsqueeze(-1)  # (N, 1)
                    att2 = pred_att_list[action_idx][0, :, 3].detach().unsqueeze(-1)  # (N, 1)
                    att_prob1 = F.softmax(att1, dim=0)  # (N, 1)
                    att_prob2 = F.softmax(att2, dim=0)  # (N, 1)
                    offset1 = pred_offset_list[action_idx][0, :, 4:6].detach()  # (N, 2)
                    offset2 = pred_offset_list[action_idx][0, :, 6:8].detach()  # (N, 2)
                    pc_xy = pc_xyz_batch[0, :, :2]  # (N, 2)
                    pred_point1_xy = ((offset1 + pc_xy) * att_prob1).sum(dim=0)  # (2, )
                    pred_point2_xy = ((offset2 + pc_xy) * att_prob2).sum(dim=0)  # (2, )
                    # TODO: set release-height more freely
                    release_point1 = torch.cat([pred_point1_xy, torch.tensor([0.05], device=self.device)]).cpu().numpy()  # (3,)
                    release_point2 = torch.cat([pred_point2_xy, torch.tensor([0.05], device=self.device)]).cpu().numpy()  # (3,)
        elif action_type == ActionTypeDef.DONE or action_type == ActionTypeDef.FLING:
            pass
        else:
            raise NotImplementedError

        def action_iterator(random: bool = False, random_top_ratio: float = 1.0):
            """iterate valid pose pairs (sorted by scores) from possible samples"""
            if action_str == 'fling':
                if random:
                    # only for self-supervised data collection
                    # randomly shuffle sorted_pair_idxs
                    # only shuffle top K pairs
                    top_k = int(len(sorted_pair_idxs) * random_top_ratio)
                    np.random.shuffle(sorted_pair_idxs[:top_k])
                for idx in range(sorted_pair_idxs.shape[0]):
                    pair_idx = sorted_pair_idxs[idx]
                    idx1 = pair_idx // num_candidates
                    idx2 = pair_idx % num_candidates
                    poses = np.stack([grasp_point_all[idx1], grasp_point_all[idx2],
                                      release_point_all[idx1], release_point_all[idx2]], axis=0)  # (34 3)
                    # TODO: add predicted rotation angles in poses
                    poses = np.concatenate([poses, np.zeros((4, 1))], axis=1)  # (x, y, z, rot)
                    poses_nocs = np.stack([grasp_point_nocs_all[idx1], grasp_point_nocs_all[idx2]], axis=0)  # (2, 3)
                    yield poses, poses_nocs, (int(idx1), int(idx2))
                return None, None, None
            elif action_str != 'done':
                poses = np.stack([grasp_point1, grasp_point2, release_point1, release_point2], axis=0)
                # TODO: add predicted rotation angles in poses
                poses = np.concatenate([poses, np.zeros((4, 1))], axis=1)  # (x, y, z, rot)
                poses_nocs = np.zeros((2, 3))
                yield poses, poses_nocs, (-1, -1)
                return None, None, None
            elif action_str == 'done':
                poses = np.zeros((4, 4))  # (x, y, z, rot)
                poses_nocs = np.zeros((2, 3))
                yield poses, poses_nocs, None
                return None, None, None
            else:
                raise NotImplementedError

        pred_message = PredictionMessage(action_type=action_type,
                                         action_iterator=action_iterator,
                                         attmaps=attmaps,
                                         nocs_map=nocs_map,
                                         grasp_point_all=grasp_point_all,
                                         grasp_point_nocs_all=grasp_point_nocs_all,
                                         virtual_reward_all=virtual_reward_all,
                                         real_reward_all=real_reward_all)
        if return_timing:
            pred_message.nn_timing = timing
        return pred_message

    @staticmethod
    def save_vis_single(pc_xyz, attmaps, pc_nocs, save_dir, pred_action, pred_keypoints, pred_grasp_nocs, pcd_id=0,
                        pred_keypoints_all=None, pred_grasp_nocs_all=None):
        if pred_action == 'fling':
            attmap = attmaps['fling_pick'].T  # (K, N)
        elif pred_action == 'drag':
            attmap = np.stack(
                [attmaps['drag_pick1'], attmaps['drag_pick2'], attmaps['drag_place1'], attmaps['drag_place2']])
        elif pred_action == 'fold1':
            attmap = np.stack(
                [attmaps['fold1_pick1'], attmaps['fold1_pick2'], attmaps['fold1_place1'], attmaps['fold1_place2']])
        elif pred_action == 'fold2':
            attmap = np.stack(
                [attmaps['fold2_pick1'], attmaps['fold2_pick2'], attmaps['fold2_place1'], attmaps['fold2_place2']])
        elif pred_action == 'pick_and_place':
            attmap = np.stack(
                [attmaps['pnp_pick1'], attmaps['pnp_pick2'], attmaps['pnp_place1'], attmaps['pnp_place2']])
        else:
            attmap = np.stack(
                [attmaps['fold1_pick1'], attmaps['fold1_pick2'], attmaps['fold1_place1'], attmaps['fold1_place2']])
        data_dict = dict(pc_xyz=pc_xyz.tolist(),
                         pc_nocs=pc_nocs.tolist(),
                         attmaps=attmap.tolist(),
                         grasp_nocs=pred_grasp_nocs.tolist(),
                         action=pred_action,
                         pred_keypoints_all=pred_keypoints_all.tolist() if pred_keypoints_all is not None else None,
                         pred_grasp_nocs_all=pred_grasp_nocs_all.tolist() if pred_grasp_nocs_all is not None else None,
                         left_gripper_point_frame1=pred_keypoints[0, :3].tolist(),
                         right_gripper_point_frame1=pred_keypoints[1, :3].tolist(),
                         left_gripper_point_frame2=pred_keypoints[2, :3].tolist(),
                         right_gripper_point_frame2=pred_keypoints[3, :3].tolist(),
                         left_theta_frame1=pred_keypoints[0, 3].tolist(),
                         right_theta_frame1=pred_keypoints[1, 3].tolist(),
                         left_theta_frame2=pred_keypoints[2, 3].tolist(),
                         right_theta_frame2=pred_keypoints[3, 3].tolist())
        os.makedirs(os.path.join(save_dir, 'vis'), exist_ok=True)
        data = bson.BSON.encode(data_dict)
        out_path = osp.join(save_dir, 'vis', '{:0>4d}.bson'.format(pcd_id))
        with open(out_path, 'wb') as f:
            f.write(data)
        print('Saving to {}! Action type: {}'.format(out_path, pred_action))

    def infer_normal(self, batch, batch_idx, is_train=True):
        for idx, data in enumerate(batch):
            if isinstance(data, torch.Tensor) and torch.any(~torch.isfinite(data)):
                logger.error(f'normal data has NaN or Inf: idx {idx}, gt data {data}!')
                raise RuntimeError
        metrics = self.forward_normal_loss(batch)
        for key, value in metrics.items():
            log_key = ('train_' if is_train else 'val_') + key
            self.log(log_key, value)
        return metrics

    def infer_virtual_reward(self, batch, batch_idx, is_train=True):
        for idx, data in enumerate(batch):
            if isinstance(data, torch.Tensor) and torch.any(~torch.isfinite(data)):
                logger.error(f'virtual data has NaN or Inf: idx {idx}, gt data {data}!')
                raise RuntimeError
        metrics = self.forward_virtual_reward_loss(batch)
        for key, value in metrics.items():
            log_key = ('train_' if is_train else 'val_') + key
            self.log(log_key, value)
        return metrics

    def infer_real_world(self, batch, batch_idx, is_train=True):
        for idx, data in enumerate(batch):
            if isinstance(data, torch.Tensor) and torch.any(~torch.isfinite(data)):
                logger.error(f'real world data has NaN or Inf: idx {idx}, gt data {data}!')
                raise RuntimeError
        metrics = self.forward_real_world_loss(batch)
        for key, value in metrics.items():
            log_key = ('train_' if is_train else 'val_') + key
            self.log(log_key, value)
        return metrics

    def training_step(self, batch, batch_idx):
        if self.enable_real_world_finetune:
            if len(batch) == 2 and isinstance(batch[0], tuple) and isinstance(batch[1], tuple):
                # Use min_size Combiner for training with multiple dataloaders.
                # See https://lightning.ai/docs/pytorch/stable/data/iterables.html#multiple-dataloaders
                metrics_virtual_reward = self.infer_virtual_reward(batch[0], batch_idx, is_train=True)
                metrics_real_reward = self.infer_real_world(batch[1], batch_idx, is_train=True)
                metrics = {**metrics_virtual_reward, **metrics_real_reward}
                metrics['loss'] = metrics_virtual_reward['virtual-reward_loss'] + metrics_real_reward['real-world_loss']
            else:
                metrics = self.infer_real_world(batch, batch_idx, is_train=True)
                metrics['loss'] = metrics['real-world_loss']
        else:
            if len(batch) == 2 and isinstance(batch[0], tuple) and isinstance(batch[1], tuple):
                # Use min_size Combiner for training with multiple dataloaders.
                # See https://lightning.ai/docs/pytorch/stable/data/iterables.html#multiple-dataloaders
                metrics_normal = self.infer_normal(batch[0], batch_idx, is_train=True)
                metrics_virtual_reward = self.infer_virtual_reward(batch[1], batch_idx, is_train=True)
                metrics = {**metrics_normal, **metrics_virtual_reward}
                metrics['loss'] = metrics_normal['normal_loss'] + metrics_virtual_reward['virtual-reward_loss']
            else:
                metrics = self.infer_normal(batch, batch_idx, is_train=True)
                metrics['loss'] = metrics['normal_loss']
        if torch.any(~torch.isfinite(metrics['loss'])):
            metrics['loss'] = torch.zeros_like(metrics['loss'], requires_grad=True)
            logger.warning('NaN or Inf detected in loss. Skipping this batch.')
        self.log('train_loss', metrics['loss'])
        return metrics['loss']

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        if self.enable_real_world_finetune:
            if dataloader_idx == 0:
                metrics = self.infer_virtual_reward(batch, batch_idx, is_train=False)
                metrics['loss'] = metrics['virtual-reward_loss']
            elif dataloader_idx == 1 or dataloader_idx is None:
                metrics = self.infer_real_world(batch, batch_idx, is_train=False)
                metrics['loss'] = metrics['real-world_loss']
            else:
                raise NotImplementedError
        else:
            # Use SequentialCombiner for validation.
            # See https://lightning.ai/docs/pytorch/stable/data/iterables.html#multiple-dataloaders
            if dataloader_idx == 0 or dataloader_idx is None:
                metrics = self.infer_normal(batch, batch_idx, is_train=False)
                metrics['loss'] = metrics['normal_loss']
            elif dataloader_idx == 1:
                metrics = self.infer_virtual_reward(batch, batch_idx, is_train=False)
                metrics['loss'] = metrics['virtual-reward_loss']
            else:
                raise NotImplementedError
        self.log('val_loss', metrics['loss'])
        return metrics['loss']
