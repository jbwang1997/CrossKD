# Copyright (c) OpenMMLab. All rights reserved.
from pathlib import Path
import warnings
from typing import List, Union

import torch
import torch.nn.functional as F
from torch import Tensor

from mmengine.config import Config
from mmengine.runner import load_checkpoint

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import InstanceList, OptInstanceList, reduce_mean

from ..utils import multi_apply, unpack_gt_instances
from .crosskd_single_stage import CrossKDSingleStageDetector


@MODELS.register_module()
class CrossKDYolov3(CrossKDSingleStageDetector):

    def __init__(
        self,
        backbone,
        neck,
        bbox_head,
        teacher_config,
        teacher_ckpt=None,
        kd_cfg=None,
        train_cfg=None,
        test_cfg=None,
        data_preprocessor=None,
        init_cfg=None
    ) -> None:
        super(CrossKDSingleStageDetector, self).__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
        # Build teacher model
        if isinstance(teacher_config, (str, Path)):
            teacher_config = Config.fromfile(teacher_config)
        self.teacher = MODELS.build(teacher_config['model'])
        if teacher_ckpt is not None:
            load_checkpoint(self.teacher, teacher_ckpt, map_location='cpu')
        # In order to reforward teacher model,
        # set requires_grad of teacher model to False
        self.freeze(self.teacher)
        self.loss_cls_kd = MODELS.build(kd_cfg['loss_cls_kd'])
        self.loss_conf_kd = MODELS.build(kd_cfg['loss_conf_kd'])
        self.loss_xy_kd = MODELS.build(kd_cfg['loss_xy_kd'])
        self.loss_wh_kd = MODELS.build(kd_cfg['loss_wh_kd'])

        self.with_feat_distill = False
        if kd_cfg.get('loss_feat_kd', None):
            self.loss_feat_kd = MODELS.build(kd_cfg['loss_feat_kd'])
            self.with_feat_distill = True

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
        tea_pred_maps = self.teacher.bbox_head(tea_x)[0]
        stu_x = self.extract_feat(batch_inputs)
        stu_pred_maps = self.bbox_head(stu_x)[0]
        reused_x = multi_apply(self.align_scale, stu_x, tea_x)[0]
        reused_pred_maps = self.teacher.bbox_head(reused_x)[0]

        outputs = unpack_gt_instances(batch_data_samples)
        (batch_gt_instances, batch_gt_instances_ignore,
         batch_img_metas) = outputs
        losses = self.loss_by_feat(tea_pred_maps, tea_x, stu_pred_maps,
                                   stu_x, reused_pred_maps,
                                   batch_gt_instances, batch_img_metas,
                                   batch_gt_instances_ignore)
        return losses

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
        return stu_feat.reshape(C, N, H, W).permute(1, 0, 2, 3), 

    def loss_by_feat(
            self,
            tea_pred_maps: List[Tensor],
            tea_x: List[Tensor],
            stu_pred_maps: List[Tensor],
            stu_x: List[Tensor],
            reused_pred_maps: List[Tensor],
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

        num_imgs = len(batch_img_metas)
        device = stu_pred_maps[0][0].device

        featmap_sizes = [
            stu_pred_maps[i].shape[-2:] for i in \
                range(self.bbox_head.num_levels)]
        mlvl_anchors = self.bbox_head.prior_generator.grid_priors(
            featmap_sizes, device=device)
        anchor_list = [mlvl_anchors for _ in range(num_imgs)]

        responsible_flag_list = []
        for img_id in range(num_imgs):
            responsible_flag_list.append(
                self.bbox_head.responsible_flags(
                    featmap_sizes,
                    batch_gt_instances[img_id].bboxes,
                    device))

        target_maps_list, neg_maps_list = self.bbox_head.get_targets(
            anchor_list, responsible_flag_list, batch_gt_instances)

        losses_cls, losses_conf, losses_xy, losses_wh = multi_apply(
            self.bbox_head.loss_by_feat_single, stu_pred_maps,
            target_maps_list, neg_maps_list)

        losses = dict(
            loss_cls=losses_cls,
            loss_conf=losses_conf,
            loss_xy=losses_xy,
            loss_wh=losses_wh)
        
        # start prediction mimicking
        losses_cls_kd, losses_conf_kd, losses_xy_kd, losses_wh_kd = \
            multi_apply(
                self.pred_mimicking_loss_single,
                tea_pred_maps,
                reused_pred_maps,
                target_maps_list,
                neg_maps_list)

        losses.update(dict(
            loss_cls_kd=losses_cls_kd,
            loss_conf_kd=losses_conf_kd,
            loss_xy_kd=losses_xy_kd,
            loss_wh_kd=losses_wh_kd,))
        
        if self.with_feat_distill:
            losses_feat_kd = [
                self.loss_feat_kd(stu_feat, tea_feat)
                for stu_feat, tea_feat in zip(stu_x, tea_x)
            ]
            losses.update(loss_feat_kd=losses_feat_kd)
        return losses

    def pred_mimicking_loss_single(self, tea_pred_map, reused_pred_map,
                                   target_map, neg_map):
        num_imgs = len(tea_pred_map)
        neg_mask = neg_map.float()
        pos_mask = target_map[..., 4]
        pos_and_neg_mask = neg_mask + pos_mask
        pos_and_neg_mask = pos_and_neg_mask.reshape(-1, 1)
        pos_mask = pos_mask.unsqueeze(dim=-1)
        if torch.max(pos_and_neg_mask) > 1.:
            warnings.warn('There is overlap between pos and neg sample.')
            pos_and_neg_mask = pos_and_neg_mask.clamp(min=0., max=1.)

        tea_pred_map = tea_pred_map.permute(
            0, 2, 3, 1).reshape(-1, self.bbox_head.num_attrib)
        tea_pred_xy = tea_pred_map[..., :2]
        tea_pred_wh = tea_pred_map[..., 2:4]
        tea_pred_conf = tea_pred_map[..., [4]]
        tea_pred_label = tea_pred_map[..., 5:]

        reused_pred_map = reused_pred_map.permute(
            0, 2, 3, 1).reshape(-1, self.bbox_head.num_attrib)
        reused_pred_xy = reused_pred_map[..., :2]
        reused_pred_wh = reused_pred_map[..., 2:4]
        reused_pred_conf = reused_pred_map[..., [4]]
        reused_pred_label = reused_pred_map[..., 5:]

        pos_weight = tea_pred_conf.sigmoid()
        loss_cls_kd = self.loss_cls_kd(
            reused_pred_label, tea_pred_label, weight=pos_weight)
        loss_conf_kd = self.loss_conf_kd(
            reused_pred_conf, tea_pred_conf, weight=pos_and_neg_mask)
        loss_xy_kd = self.loss_xy_kd(
            reused_pred_xy, tea_pred_xy, weight=pos_weight)
        loss_wh_kd = self.loss_wh_kd(
            reused_pred_wh, tea_pred_wh, weight=pos_weight)
        
        return loss_cls_kd, loss_conf_kd, loss_xy_kd, loss_wh_kd
