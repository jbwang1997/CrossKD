# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Union

import torch
from torch import Tensor
import torch.nn.functional as F

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import (InstanceList, OptInstanceList, OptConfigType, reduce_mean)
from ..utils import multi_apply, unpack_gt_instances
from .crosskd_single_stage import CrossKDSingleStageDetector


@MODELS.register_module()
class CrossKDATSS(CrossKDSingleStageDetector):

    def __init__(self, 
                 kd_cfg: OptConfigType = None,
                 **kwargs) -> None:
        super().__init__(kd_cfg=kd_cfg,**kwargs)
        self.loss_center_kd = None
        if kd_cfg.get('loss_center_kd', None):
            self.loss_center_kd = MODELS.build(kd_cfg['loss_center_kd'])
                
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
                        module=self.teacher)
            
        stu_x = self.extract_feat(batch_inputs)
        stu_cls_scores, stu_bbox_preds, stu_centernesses, stu_cls_hold, stu_reg_hold = \
            multi_apply(self.forward_hkd_single, 
                        stu_x,
                        self.bbox_head.scales, 
                        module=self)
            
        reused_cls_scores, reused_bbox_preds, reused_centernesses = multi_apply(
            self.reuse_teacher_head, 
            tea_cls_hold, 
            tea_reg_hold, 
            stu_cls_hold,
            stu_reg_hold, 
            self.teacher.bbox_head.scales)


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
    
    def forward_hkd_single(self, x, scale, module):
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
        cls_score = module.bbox_head.atss_cls(cls_feat)
        bbox_pred = scale(module.bbox_head.atss_reg(reg_feat)).float()
        centerness = module.bbox_head.atss_centerness(reg_feat)
        return cls_score, bbox_pred, centerness, cls_feat_hold, reg_feat_hold
    
    def reuse_teacher_head(self, tea_cls_feat, tea_reg_feat, stu_cls_feat,
                           stu_reg_feat, scale):
        reused_cls_feat = self.align_scale(stu_cls_feat, tea_cls_feat)
        reused_reg_feat = self.align_scale(stu_reg_feat, tea_reg_feat)
        if self.reused_teacher_head_idx != 0:
            reused_cls_feat = F.relu(reused_cls_feat)
            reused_reg_feat = F.relu(reused_reg_feat)

        module = self.teacher.bbox_head
        for i in range(self.reused_teacher_head_idx, module.stacked_convs):
            reused_cls_feat = module.cls_convs[i](reused_cls_feat)
            reused_reg_feat = module.reg_convs[i](reused_reg_feat)
        reused_cls_score = module.atss_cls(reused_cls_feat)
        reused_bbox_pred = scale(module.atss_reg(reused_reg_feat)).float()
        reused_centerness = module.atss_centerness(reused_reg_feat)
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

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.bbox_head.prior_generator.num_levels

        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.bbox_head.get_anchors(
            featmap_sizes, batch_img_metas, device=device)

        cls_reg_targets = self.bbox_head.get_targets(
            anchor_list,
            valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore)

        (anchor_list, labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, avg_factor) = cls_reg_targets

        avg_factor = reduce_mean(
            torch.tensor(avg_factor, dtype=torch.float, device=device)).item()

        losses_cls, losses_bbox, loss_centerness, \
            bbox_avg_factor = multi_apply(
                self.bbox_head.loss_by_feat_single,
                anchor_list,
                cls_scores,
                bbox_preds,
                centernesses,
                labels_list,
                label_weights_list,
                bbox_targets_list,
                avg_factor=avg_factor)

        bbox_avg_factor = sum(bbox_avg_factor)
        bbox_avg_factor = reduce_mean(bbox_avg_factor).clamp_(min=1).item()
        losses_bbox = list(map(lambda x: x / bbox_avg_factor, losses_bbox))
        losses = dict(loss_cls=losses_cls, loss_bbox=losses_bbox, loss_centerness=loss_centerness)

        losses_cls_kd, losses_reg_kd, losses_center_kd = multi_apply(
            self.pred_imitation_loss_single,
            labels_list,
            anchor_list,
            tea_cls_scores,
            tea_bbox_preds,
            tea_centernesses,
            reused_cls_scores,
            reused_bbox_preds,
            reused_centernesses,
            label_weights_list,
            avg_factor=avg_factor)
        losses.update(dict(loss_cls_kd=losses_cls_kd, loss_reg_kd=losses_reg_kd, losses_center_kd=losses_center_kd))
        
        if self.with_feat_distill:
            losses_feat_kd = [
                self.loss_feat_kd(feat, tea_feat)
                for feat, tea_feat in zip(feats, tea_feats)
            ]
            losses.update(loss_feat_kd=losses_feat_kd)
        return losses
    
    
    def pred_imitation_loss_single(self, 
                                   labels,
                                   anchors,
                                   tea_cls_score, 
                                   tea_bbox_pred,
                                   tea_centernesses,
                                   reused_cls_score, 
                                   reused_bbox_pred,
                                   reused_centernesses,
                                   label_weights, 
                                   avg_factor):
        # classification branch distillation
        tea_cls_score = tea_cls_score.permute(0, 2, 3, 1).reshape(-1, self.bbox_head.cls_out_channels)
        reused_cls_score = reused_cls_score.permute(0, 2, 3, 1).reshape(-1, self.bbox_head.cls_out_channels)
        label_weights = label_weights.reshape(-1)
        loss_cls_kd = self.loss_cls_kd(
            reused_cls_score,
            tea_cls_score,
            label_weights,
            avg_factor=avg_factor)

        # regression branch distillation
        bbox_coder = self.bbox_head.bbox_coder
        tea_bbox_pred = tea_bbox_pred.permute(0, 2, 3, 1).reshape(-1, bbox_coder.encode_size)
        reused_bbox_pred = reused_bbox_pred.permute(0, 2, 3, 1).reshape(-1, bbox_coder.encode_size)
        anchors = anchors.reshape(-1, anchors.size(-1))
        tea_bbox_pred = bbox_coder.decode(anchors, tea_bbox_pred)
        reused_bbox_pred = bbox_coder.decode(anchors, reused_bbox_pred)
        
        reg_weights = tea_cls_score.max(dim=1)[0].sigmoid()
        reg_weights[label_weights == 0] = 0

        loss_reg_kd = self.loss_reg_kd(
            reused_bbox_pred,
            tea_bbox_pred,
            weight=reg_weights,
            avg_factor=avg_factor)
        
        # centernesses branch distillation
        labels = labels.reshape(-1)
        bg_class_ind = self.bbox_head.num_classes
        pos_inds = ((labels >= 0) & (labels < bg_class_ind)).nonzero().squeeze(1)
        tea_centernesses = tea_centernesses.permute(0, 2, 3, 1).reshape(-1)
        reused_centernesses = reused_centernesses.permute(0, 2, 3, 1).reshape(-1)

        if len(pos_inds) > 0:
            loss_center_kd = self.loss_center_kd(
                reused_centernesses[pos_inds],
                tea_centernesses[pos_inds].sigmoid(),
                avg_factor=avg_factor)
        else:
            loss_center_kd = reused_centernesses.new_tensor(0.)
        return loss_cls_kd, loss_reg_kd, loss_center_kd