# Copyright (c) OpenMMLab. All rights reserved.
import copy
from pathlib import Path
from typing import Any, List, Optional, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import SampleList, DetDataSample
from mmdet.structures.bbox import bbox2roi
from mmdet.utils import (ConfigType, InstanceList, OptConfigType,
                         OptMultiConfig)
from ..utils import unpack_gt_instances
from .two_stage import TwoStageDetector


@MODELS.register_module()
class CrossKDTwoStageDetector(TwoStageDetector):
    r"""Implementation of `Distilling the Knowledge in a Neural Network.
    <https://arxiv.org/abs/1503.02531>`_.

    Args:
        backbone (:obj:`ConfigDict` or dict): The backbone module.
        neck (:obj:`ConfigDict` or dict): The neck module.
        bbox_head (:obj:`ConfigDict` or dict): The bbox head module.
        teacher_config (:obj:`ConfigDict` | dict | str | Path): Config file
            path or the config object of teacher model.
        teacher_ckpt (str, optional): Checkpoint path of teacher model.
            If left as None, the model will not load any weights.
            Defaults to True.
        eval_teacher (bool): Set the train mode for teacher.
            Defaults to True.
        train_cfg (:obj:`ConfigDict` or dict, optional): The training config
            of ATSS. Defaults to None.
        test_cfg (:obj:`ConfigDict` or dict, optional): The testing config
            of ATSS. Defaults to None.
        data_preprocessor (:obj:`ConfigDict` or dict, optional): Config of
            :class:`DetDataPreprocessor` to process the input data.
            Defaults to None.
    """

    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 rpn_head: OptConfigType = None,
                 roi_head: OptConfigType = None,
                 teacher_config: Union[ConfigType, str, Path] = None,
                 teacher_ckpt: Optional[str] = None,
                 kd_cfg: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
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
        self.loss_reg_kd = MODELS.build(kd_cfg['loss_reg_kd'])
        if self.roi_head.with_mask:
            self.loss_mask_kd = MODELS.build(kd_cfg['loss_mask_kd'])

        self.with_feat_distill = False
        if 'loss_feat_kd' in kd_cfg:
            self.loss_feat_kd = MODELS.build(kd_cfg['loss_feat_kd'])
            self.with_feat_distill = True


        
    @staticmethod
    def freeze(model: nn.Module):
        """Freeze the model."""
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    def cuda(self, device: Optional[str] = None) -> nn.Module:
        """Since teacher is registered as a plain object, it is necessary to
        put the teacher model to cuda when calling ``cuda`` function."""
        self.teacher.cuda(device=device)
        return super().cuda(device=device)

    def to(self, device: Optional[str] = None) -> nn.Module:
        """Since teacher is registered as a plain object, it is necessary to
        put the teacher model to other device when calling ``to`` function."""
        self.teacher.to(device=device)
        return super().to(device=device)

    def train(self, mode: bool = True) -> None:
        """Set the same train mode for teacher and student model."""
        self.teacher.train(False)
        super().train(mode)

    def __setattr__(self, name: str, value: Any) -> None:
        """Set attribute, i.e. self.name = value

        This reloading prevent the teacher model from being registered as a
        nn.Module. The teacher module is registered as a plain object, so that
        the teacher parameters will not show up when calling
        ``self.parameters``, ``self.modules``, ``self.children`` methods.
        """
        if name == 'teacher':
            object.__setattr__(self, name, value)
        else:
            super().__setattr__(name, value)
    
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

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components
        """
        stu_x = self.extract_feat(batch_inputs)
        tea_x = self.teacher.extract_feat(batch_inputs)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_data_samples = copy.deepcopy(batch_data_samples)
            # set cat_id of gt_labels to 0 in RPN
            for data_sample in rpn_data_samples:
                data_sample.gt_instances.labels = \
                    torch.zeros_like(data_sample.gt_instances.labels)

            rpn_losses, rpn_results_list = self.rpn_head.loss_and_predict(
                stu_x, rpn_data_samples, proposal_cfg=proposal_cfg)
            # avoid get same name with roi_head loss
            keys = rpn_losses.keys()
            for key in list(keys):
                if 'loss' in key and 'rpn' not in key:
                    rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
            losses.update(rpn_losses)
        else:
            assert batch_data_samples[0].get('proposals', None) is not None
            # use pre-defined proposals in InstanceData for the second stage
            # to extract ROI features.
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        roi_losses = self.roi_head_loss_with_kd(
            stu_x, tea_x, rpn_results_list,batch_data_samples)
        losses.update(roi_losses)
        
        if self.with_feat_distill:
            losses_feat_kd = [
                self.loss_feat_kd(feat, tea_feat)
                for feat, tea_feat in zip(stu_x, tea_x)
            ]
            losses.update(losses_feat_kd=losses_feat_kd)
        
        return losses
    
    def roi_head_loss_with_kd(self,
                              stu_x: Tuple[Tensor],
                              tea_x: Tuple[Tensor],
                              rpn_results_list: InstanceList,
                              batch_data_samples: List[DetDataSample]):
        assert len(rpn_results_list) == len(batch_data_samples)
        outputs = unpack_gt_instances(batch_data_samples)
        batch_gt_instances, batch_gt_instances_ignore, _ = outputs
        roi_head = self.roi_head

        # assign gts and sample proposals
        num_imgs = len(batch_data_samples)
        sampling_results = []
        for i in range(num_imgs):
            # rename rpn_results.bboxes to rpn_results.priors
            rpn_results = rpn_results_list[i]
            rpn_results.priors = rpn_results.pop('bboxes')

            assign_result = roi_head.bbox_assigner.assign(
                rpn_results, batch_gt_instances[i],
                batch_gt_instances_ignore[i])
            sampling_result = roi_head.bbox_sampler.sample(
                assign_result,
                rpn_results,
                batch_gt_instances[i],
                feats=[lvl_feat[i][None] for lvl_feat in stu_x])
            sampling_results.append(sampling_result)

        losses = dict()
        # bbox head loss
        if roi_head.with_bbox:
            bbox_results = self.bbox_loss_with_kd(
                stu_x, tea_x, sampling_results)
            losses.update(bbox_results['loss_bbox'])
            losses.update(bbox_results['loss_bbox_kd'])

        # mask head forward and loss
        if roi_head.with_mask:
            mask_results = self.mask_loss_with_kd(
                stu_x, tea_x, sampling_results, 
                bbox_results['stu_bbox_feats'],
                bbox_results['tea_bbox_feats'],
                bbox_results['reused_bbox_feats'],
                batch_gt_instances)
            losses.update(mask_results['loss_mask'])
            losses.update(mask_results['loss_mask_kd'])

        return losses
    
    def bbox_loss_with_kd(self, stu_x, tea_x, sampling_results):
        rois = bbox2roi([res.priors for res in sampling_results])

        stu_head, tea_head = self.roi_head, self.teacher.roi_head
        stu_bbox_results = stu_head._bbox_forward(stu_x, rois)
        tea_bbox_results = tea_head._bbox_forward(tea_x, rois)
        reused_bbox_results = tea_head._bbox_forward(stu_x, rois)

        bbox_loss_and_target = stu_head.bbox_head.loss_and_target(
            cls_score=stu_bbox_results['cls_score'],
            bbox_pred=stu_bbox_results['bbox_pred'],
            rois=rois,
            sampling_results=sampling_results,
            rcnn_train_cfg=stu_head.train_cfg)
        # stu_bbox_results.update(loss_bbox=bbox_loss_and_target['loss_bbox'])

        losses_kd = dict()
        # classification KD
        reused_cls_scores = reused_bbox_results['cls_score']
        tea_cls_scores = tea_bbox_results['cls_score']
        avg_factor = sum([res.avg_factor for res in sampling_results])
        loss_cls_kd = self.loss_cls_kd(
            reused_cls_scores,
            tea_cls_scores,
            avg_factor=avg_factor)
        losses_kd['loss_cls_kd'] = loss_cls_kd

        # regression KD
        # distill all box predictions
        # assert stu_head.bbox_head.reg_class_agnostic \
        #     == tea_head.bbox_head.reg_class_agnostic
        # reused_bbox_preds = reused_bbox_results['bbox_pred'].reshape(-1, 4)
        # tea_bbox_preds = tea_bbox_results['bbox_pred'].reshape(-1, 4)
        # num_classes = stu_head.bbox_head.num_classes
        # rois = rois[:, 1:]
        # if stu_head.bbox_head.reg_class_agnostic:
        #     reg_weights = tea_cls_scores.softmax(dim=1)[:, :num_classes].sum(dim=1)
        # else:
        #     rois = rois.repeat_interleave(num_classes, dim=0)
        #     reg_weights = tea_cls_scores.softmax(dim=1)[:, :num_classes].reshape(-1)
        # coder = stu_head.bbox_head.bbox_coder
        # decoded_reused_bboxes = coder.decode(rois, reused_bbox_preds)
        # decoded_tea_bboxes = coder.decode(rois, tea_bbox_preds)
        # loss_reg_kd = self.loss_reg_kd(
        #     decoded_reused_bboxes,
        #     decoded_tea_bboxes,
        #     weight=reg_weights,
        #     avg_factor=reg_weights.sum())
        # losses_kd['loss_reg_kd'] = loss_reg_kd


        # distill max score box predictions
        # assert stu_head.bbox_head.reg_class_agnostic \
        #     == tea_head.bbox_head.reg_class_agnostic
        # num_classes = stu_head.bbox_head.num_classes
        # reused_bbox_preds = reused_bbox_results['bbox_pred']
        # tea_bbox_preds = tea_bbox_results['bbox_pred']
        # tea_cls_scores = tea_cls_scores.softmax(dim=1)[:, :num_classes]
        # reg_weights, reg_distill_idx = tea_cls_scores.max(dim=1)
        # if not stu_head.bbox_head.reg_class_agnostic:
        #     reg_distill_idx = reg_distill_idx[:, None, None].repeat(1, 1, 4)
        #     reused_bbox_preds = reused_bbox_preds.reshape(-1, num_classes, 4)
        #     reused_bbox_preds = reused_bbox_preds.gather(dim=1, index=reg_distill_idx)
        #     reused_bbox_preds = reused_bbox_preds.squeeze(1)
        #     tea_bbox_preds = tea_bbox_preds.reshape(-1, num_classes, 4)
        #     tea_bbox_preds = tea_bbox_preds.gather(dim=1, index=reg_distill_idx)
        #     tea_bbox_preds = tea_bbox_preds.squeeze(1)

        # rois = rois[:, 1:]
        # coder = stu_head.bbox_head.bbox_coder
        # decoded_reused_bboxes = coder.decode(rois, reused_bbox_preds)
        # decoded_tea_bboxes = coder.decode(rois, tea_bbox_preds)
        # loss_reg_kd = self.loss_reg_kd(
        #     decoded_reused_bboxes,
        #     decoded_tea_bboxes,
        #     weight=reg_weights,
        #     avg_factor=reg_weights.sum())
        # losses_kd['loss_reg_kd'] = loss_reg_kd

        # l1 loss
        assert stu_head.bbox_head.reg_class_agnostic \
            == tea_head.bbox_head.reg_class_agnostic
        num_classes = stu_head.bbox_head.num_classes
        reused_bbox_preds = reused_bbox_results['bbox_pred']
        tea_bbox_preds = tea_bbox_results['bbox_pred']
        tea_cls_scores = tea_cls_scores.softmax(dim=1)[:, :num_classes]
        reg_weights, reg_distill_idx = tea_cls_scores.max(dim=1)
        if not stu_head.bbox_head.reg_class_agnostic:
            reg_distill_idx = reg_distill_idx[:, None, None].repeat(1, 1, 4)
            reused_bbox_preds = reused_bbox_preds.reshape(-1, num_classes, 4)
            reused_bbox_preds = reused_bbox_preds.gather(dim=1, index=reg_distill_idx)
            reused_bbox_preds = reused_bbox_preds.squeeze(1)
            tea_bbox_preds = tea_bbox_preds.reshape(-1, num_classes, 4)
            tea_bbox_preds = tea_bbox_preds.gather(dim=1, index=reg_distill_idx)
            tea_bbox_preds = tea_bbox_preds.squeeze(1)

        loss_reg_kd = self.loss_reg_kd(
            reused_bbox_preds,
            tea_bbox_preds,
            weight=reg_weights[:, None],
            avg_factor=reg_weights.sum() * 4)
        losses_kd['loss_reg_kd'] = loss_reg_kd

        bbox_results = dict()
        for key, value in stu_bbox_results.items():
            bbox_results['stu_' + key] = value
        for key, value in tea_bbox_results.items():
            bbox_results['tea_' + key] = value
        for key, value in reused_bbox_results.items():
            bbox_results['reused_' + key] = value
        bbox_results['loss_bbox'] = bbox_loss_and_target['loss_bbox']
        bbox_results['loss_bbox_kd'] = losses_kd
        return bbox_results
    
    def mask_loss_with_kd(self, stu_x, tea_x,
                          sampling_results,
                          stu_bbox_feats,
                          tea_bbox_feats,
                          reused_bbox_feats,
                          batch_gt_instances) -> dict:
        stu_head, tea_head = self.roi_head, self.teacher.roi_head
        if not stu_head.share_roi_extractor:
            pos_rois = bbox2roi([res.pos_priors for res in sampling_results])
            stu_mask_results = stu_head._mask_forward(stu_x, pos_rois)
            tea_mask_results = tea_head._mask_forward(tea_x, pos_rois)
            reused_mask_results = tea_head._mask_forward(stu_x, pos_rois)
        else:
            pos_inds = []
            device = stu_bbox_feats.device
            for res in sampling_results:
                pos_inds.append(
                    torch.ones(
                        res.pos_priors.shape[0],
                        device=device,
                        dtype=torch.uint8))
                pos_inds.append(
                    torch.zeros(
                        res.neg_priors.shape[0],
                        device=device,
                        dtype=torch.uint8))
            pos_inds = torch.cat(pos_inds)

            stu_mask_results = stu_head._mask_forward(
                stu_x, pos_inds=pos_inds, bbox_feats=stu_bbox_feats)
            tea_mask_results = tea_head._mask_forward(
                tea_x, pos_inds=pos_inds, bbox_feats=tea_bbox_feats)
            reused_mask_results = tea_head._mask_forward(
                stu_x, pos_inds=pos_inds, bbox_feats=reused_bbox_feats)

        mask_loss_and_target = stu_head.mask_head.loss_and_target(
            mask_preds=stu_mask_results['mask_preds'],
            sampling_results=sampling_results,
            batch_gt_instances=batch_gt_instances,
            rcnn_train_cfg=stu_head.train_cfg)
        
        losses_kd = dict()
        num_classes = stu_head.bbox_head.num_classes
        num_rois = stu_bbox_feats.size(0)
        reused_mask_preds = reused_mask_results['mask_preds']
        reused_mask_preds = reused_mask_preds.permute(0, 2, 3, 1)
        reused_mask_preds = reused_mask_preds.reshape(-1, num_classes)
        tea_mask_preds = tea_mask_results['mask_preds']
        tea_mask_preds = tea_mask_preds.permute(0, 2, 3, 1)
        tea_mask_preds = tea_mask_preds.reshape(-1, num_classes)
        loss_mask_kd = self.loss_mask_kd(
            reused_mask_preds,
            tea_mask_preds,
            avg_factor=num_rois)
        losses_kd['loss_mask_kd'] = loss_mask_kd

        mask_results = dict()
        for key, value in stu_mask_results.items():
            mask_results['stu_' + key] = value
        for key, value in tea_mask_results.items():
            mask_results['tea_' + key] = value
        for key, value in reused_mask_results.items():
            mask_results['reused_' + key] = value

        mask_results.update(loss_mask=mask_loss_and_target['loss_mask'])
        mask_results.update(loss_mask_kd=losses_kd)
        return mask_results