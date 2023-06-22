# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Union

import torch
from torch import Tensor
import torch.nn.functional as F

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import (InstanceList, OptInstanceList, reduce_mean)
from ..utils import multi_apply, unpack_gt_instances
from .crosskd_single_stage import CrossKDSingleStageDetector

INF = 1e8

@MODELS.register_module()
class CrossKDFCOS(CrossKDSingleStageDetector):
    
    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        tea_x = self.teacher.extract_feat(batch_inputs)
        tea_cls_scores, tea_bbox_preds, tea_centernesses, tea_cls_hold, tea_reg_hold = \
            multi_apply(self.forward_hkd_single, 
                        tea_x, 
                        self.teacher.bbox_head.scales,
                        self.teacher.bbox_head.strides, 
                        module=self.teacher)
        stu_x = self.extract_feat(batch_inputs)
        stu_cls_scores, stu_bbox_preds,stu_centernesses, stu_cls_hold, stu_reg_hold = \
            multi_apply(self.forward_hkd_single, 
                        stu_x,
                        self.bbox_head.scales,
                        self.bbox_head.strides,  
                        module=self)
        reused_cls_scores, reused_bbox_preds, reused_centernesses = multi_apply(
            self.reuse_teacher_head, 
            tea_cls_hold, 
            tea_reg_hold, 
            stu_cls_hold,
            stu_reg_hold,
            self.teacher.bbox_head.scales,
            self.teacher.bbox_head.strides)

        outputs = unpack_gt_instances(batch_data_samples)
        (batch_gt_instances, batch_gt_instances_ignore,
         batch_img_metas) = outputs
        losses = self.loss_by_feat(tea_cls_scores, 
                                   tea_bbox_preds,
                                   tea_centernesses,
                                   tea_x,
                                   stu_cls_scores,
                                   stu_bbox_preds,
                                   stu_centernesses,
                                   stu_x,
                                   reused_cls_scores,
                                   reused_bbox_preds,
                                   reused_centernesses,
                                   batch_gt_instances,
                                   batch_img_metas, 
                                   batch_gt_instances_ignore)
        return losses
    
    def forward_hkd_single(self, x, scale, stride, module):
        cls_feat, reg_feat = x, x
        cls_feat_hold, reg_feat_hold = x, x
        for i, cls_conv in enumerate(module.bbox_head.cls_convs):
            cls_feat = cls_conv(cls_feat, activate=False)
            if i + 1 == self.reused_teacher_head_idx:
                cls_feat_hold = cls_feat
            cls_feat = cls_conv.activate(cls_feat)
        for i, reg_conv in enumerate(module.bbox_head.reg_convs):
            reg_feat = reg_conv(reg_feat, activate=False)
            if i + 1 == self.reused_teacher_head_idx:
                reg_feat_hold = reg_feat
            reg_feat = reg_conv.activate(reg_feat)
        cls_score = module.bbox_head.conv_cls(cls_feat)
        bbox_pred = scale(module.bbox_head.conv_reg(reg_feat)).float()
        if module.bbox_head.centerness_on_reg:
            centerness = module.bbox_head.conv_centerness(reg_feat)
        else:
            centerness = module.bbox_head.conv_centerness(cls_feat)
        if module.bbox_head.norm_on_bbox:
            # bbox_pred needed for gradient computation has been modified
            # by F.relu(bbox_pred) when run with PyTorch 1.10. So replace
            # F.relu(bbox_pred) with bbox_pred.clamp(min=0)
            bbox_pred = bbox_pred.clamp(min=0)
            if not module.bbox_head.training:
                bbox_pred *= stride
        else:
            bbox_pred = bbox_pred.exp()
        return cls_score, bbox_pred, centerness, cls_feat_hold, reg_feat_hold
    
    def reuse_teacher_head(self, tea_cls_feat, tea_reg_feat, stu_cls_feat, stu_reg_feat, scale, stride):
        reused_cls_feat = self.align_scale(stu_cls_feat, tea_cls_feat)
        reused_reg_feat = self.align_scale(stu_reg_feat, tea_reg_feat)
        if self.reused_teacher_head_idx != 0:
            reused_cls_feat = F.relu(reused_cls_feat)
            reused_reg_feat = F.relu(reused_reg_feat)

        module = self.teacher.bbox_head
        for i in range(self.reused_teacher_head_idx, module.stacked_convs):
            reused_cls_feat = module.cls_convs[i](reused_cls_feat)
            reused_reg_feat = module.reg_convs[i](reused_reg_feat)
        reused_cls_score = module.conv_cls(reused_cls_feat)
        reused_bbox_pred = scale(module.conv_reg(reused_reg_feat)).float()
        if module.centerness_on_reg:
            reused_centerness = module.conv_centerness(reused_reg_feat)
        else:
            reused_centerness = module.conv_centerness(reused_cls_feat)
        if module.norm_on_bbox:
            # bbox_pred needed for gradient computation has been modified
            # by F.relu(bbox_pred) when run with PyTorch 1.10. So replace
            # F.relu(bbox_pred) with bbox_pred.clamp(min=0)
            reused_bbox_pred = reused_bbox_pred.clamp(min=0)
            if not module.training:
                reused_bbox_pred *= stride
        else:
            reused_bbox_pred = reused_bbox_pred.exp()
        return reused_cls_score, reused_bbox_pred, reused_centerness
    
    def align_scale(self, stu_feat, tea_feat):
        N, C, H, W = stu_feat.size()
        # normalize student feature
        stu_feat = stu_feat.permute(1, 0, 2, 3).reshape(C, -1)
        stu_mean = stu_feat.mean(dim=-1, keepdim=True)
        stu_std = stu_feat.std(dim=-1, keepdim=True)
        stu_feat = (stu_feat - stu_mean) / (stu_std + 1e-6)
        #
        tea_feat = tea_feat.permute(1, 0, 2, 3).reshape(C, -1)
        tea_mean = tea_feat.mean(dim=-1, keepdim=True)
        tea_std = tea_feat.std(dim=-1, keepdim=True)
        stu_feat = stu_feat * tea_std + tea_mean
        return stu_feat.reshape(C, N, H, W).permute(1, 0, 2, 3)
    
    def loss_by_feat(
            self,
            tea_cls_scores: List[Tensor],
            tea_bbox_preds: List[Tensor],
            tea_centernesses: List[Tensor],
            tea_feats: List[Tensor],
            cls_scores: List[Tensor],
            bbox_preds: List[Tensor],
            centernesses: List[Tensor],
            feats: List[Tensor],
            reused_cls_scores: List[Tensor],
            reused_bbox_preds: List[Tensor],
            reused_centernesses: List[Tensor],
            batch_gt_instances: InstanceList,
            batch_img_metas: List[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (list[Tensor]): Cls and quality scores for each scale
                level has shape (N, num_classes, H, W).
            bbox_preds (list[Tensor]): Box distribution logits for each scale
                level with shape (N, 4*(n+1), H, W), n is max value of integral
                set.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.  It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], Optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert len(cls_scores) == len(bbox_preds) == len(centernesses)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.bbox_head.prior_generator.grid_priors(
            featmap_sizes,
            dtype=bbox_preds[0].dtype,
            device=bbox_preds[0].device)
        labels, bbox_targets = self.bbox_head.get_targets(all_level_points,
                                                batch_gt_instances)

        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds and centerness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.bbox_head.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_centerness = torch.cat(flatten_centerness)
        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        # repeat points to align with bbox_preds
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.bbox_head.num_classes
        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        num_pos = torch.tensor(
            len(pos_inds), dtype=torch.float, device=bbox_preds[0].device)
        num_pos = max(reduce_mean(num_pos), 1.0)
        loss_cls = self.bbox_head.loss_cls(
            flatten_cls_scores, flatten_labels, avg_factor=num_pos)

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]
        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        pos_centerness_targets = self.bbox_head.centerness_target(pos_bbox_targets)
        # centerness weighted iou loss
        pos_centerness_denorm = max(
            reduce_mean(pos_centerness_targets.sum().detach()), 1e-6)

        if len(pos_inds) > 0:
            pos_points = flatten_points[pos_inds]
            pos_decoded_bbox_preds = self.bbox_head.bbox_coder.decode(
                pos_points, pos_bbox_preds)
            pos_decoded_target_preds = self.bbox_head.bbox_coder.decode(
                pos_points, pos_bbox_targets)
            loss_bbox = self.bbox_head.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds,
                weight=pos_centerness_targets,
                avg_factor=pos_centerness_denorm)
            loss_centerness = self.bbox_head.loss_centerness(
                pos_centerness, pos_centerness_targets, avg_factor=num_pos)
        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_centerness = pos_centerness.sum()
        
        # flatten tea_cls_scores, tea_bbox_preds and tea_centernesses
        flatten_tea_cls_scores = [
            tea_cls_scores.permute(0, 2, 3, 1).reshape(-1, self.bbox_head.cls_out_channels)
            for tea_cls_scores in tea_cls_scores
        ]
        flatten_tea_bbox_preds = [
            tea_bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for tea_bbox_pred in tea_bbox_preds
        ]
        flatten_tea_centernesses = [
            tea_centerness.permute(0, 2, 3, 1).reshape(-1, 1)
            for tea_centerness in tea_centernesses
        ]
        flatten_tea_cls_scores = torch.cat(flatten_tea_cls_scores)
        flatten_tea_bbox_preds = torch.cat(flatten_tea_bbox_preds)
        flatten_tea_centernesses = torch.cat(flatten_tea_centernesses)
        
        # flatten reused_cls_scores, reused_bbox_preds and reused_centernesses
        flatten_reused_cls_scores = [
            reused_cls_score.permute(0, 2, 3, 1).reshape(-1, self.bbox_head.cls_out_channels)
            for reused_cls_score in reused_cls_scores
        ]
        flatten_reused_bbox_preds = [
            reused_bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for reused_bbox_pred in reused_bbox_preds
        ]
        flatten_reused_centernesses = [
            reused_centerness.permute(0, 2, 3, 1).reshape(-1, 1)
            for reused_centerness in reused_centernesses
        ]
        flatten_reused_cls_scores = torch.cat(flatten_reused_cls_scores)
        flatten_reused_bbox_preds = torch.cat(flatten_reused_bbox_preds)
        flatten_reused_centernesses = torch.cat(flatten_reused_centernesses)
        
        losses_cls_kd = self.loss_cls_kd(flatten_reused_cls_scores, 
                                         flatten_tea_cls_scores, 
                                         avg_factor=pos_centerness_denorm)
        
        flatten_tea_bbox_preds = self.bbox_head.bbox_coder.decode(
                flatten_points, flatten_tea_bbox_preds)
        flatten_reused_bbox_preds = self.bbox_head.bbox_coder.decode(
                flatten_points, flatten_reused_bbox_preds)
        
        reg_weights = flatten_tea_cls_scores.max(dim=1)[0].sigmoid()

        losses_reg_kd = self.loss_reg_kd(flatten_reused_bbox_preds,
                                        flatten_tea_bbox_preds,
                                        weight=reg_weights, 
                                        avg_factor=pos_centerness_denorm)
        losses = dict(loss_cls=loss_cls,
                      loss_bbox=loss_bbox,
                      loss_centerness=loss_centerness,
                      loss_cls_kd=losses_cls_kd,
                      loss_reg_kd=losses_reg_kd)

        if self.with_feat_distill:
            losses_feat_kd = [
                self.loss_feat_kd(feat, tea_feat)
                for feat, tea_feat in zip(feats, tea_feats)
            ]
            for i, loss in enumerate(losses_feat_kd):
                losses.update({"loss_feat_kd_{}".format(i):loss})
        return losses