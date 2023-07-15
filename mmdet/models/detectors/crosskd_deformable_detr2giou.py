# Copyright (c) OpenMMLab. All rights reserved.
from pathlib import Path
from typing import Any, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.structures.bbox import cat_boxes, bbox_cxcywh_to_xyxy
from mmdet.utils import (ConfigType, InstanceList, OptConfigType,
                         OptInstanceList, OptMultiConfig, reduce_mean)
from ..utils import images_to_levels, multi_apply, unpack_gt_instances
from .deformable_detr import DeformableDETR
from mmdet.models.layers.transformer.utils import inverse_sigmoid


@MODELS.register_module()
class CrossKDDeformableDETR2GIOU(DeformableDETR):
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
                 *args,
                 teacher_ckpt=None,
                 teacher_config=None,
                 kd_cfg=None,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
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
        self.loss_iou_kd = MODELS.build(kd_cfg['loss_iou_kd'])
        self.with_feat_distill = False
        if kd_cfg.get('loss_feat_kd', None):
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
    
    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (bs, dim, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components
        """
        stu_feats = self.extract_feat(batch_inputs)
        tea_feats = self.teacher.extract_feat(batch_inputs)

        head_inputs_dict, kd_inputs_dict = self.forward_transformer_with_kd(
            stu_feats, tea_feats, batch_data_samples)

        losses = dict()
        losses.update(self.bbox_head.loss(
            **head_inputs_dict, batch_data_samples=batch_data_samples))
        losses.update(self.cal_kd_loss(
            **kd_inputs_dict,
            batch_data_samples=batch_data_samples))
        return losses
    
    def forward_transformer_with_kd(self,
                                    stu_feats,
                                    tea_feats,
                                    batch_data_samples=None):

        # forward student
        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
            stu_feats, batch_data_samples)
        encoder_outputs_dict = self.forward_encoder(**encoder_inputs_dict)
        tmp_dec_in, head_inputs_dict = self.pre_decoder(**encoder_outputs_dict)
        decoder_inputs_dict.update(tmp_dec_in)
        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
        head_inputs_dict.update(decoder_outputs_dict)

        # forward teacher
        teacher = self.teacher
        tea_enc_inputs_dict, tea_dec_inputs_dict = \
            teacher.pre_transformer(tea_feats, batch_data_samples)
        tea_enc_outputs_dict = teacher.forward_encoder(**tea_enc_inputs_dict)
        tea_tmp_dec_in, _ = \
            teacher.pre_decoder(**tea_enc_outputs_dict)
        tea_dec_inputs_dict.update(tea_tmp_dec_in)
        tea_dec_outputs_dict = teacher.forward_decoder(**tea_dec_inputs_dict)
        # use student query and query_pos
        tea_dec_inputs_dict['tea_memory'] = tea_dec_inputs_dict.pop('memory')
        tea_dec_inputs_dict['stu_memory'] = decoder_inputs_dict['memory']
        kd_inputs_dict = self.cross_forward_teacher_decoder(**tea_dec_inputs_dict)
        
        return head_inputs_dict, kd_inputs_dict
    
    def cross_forward_teacher_decoder(
            self, query, query_pos, stu_memory, tea_memory,
            memory_mask, reference_points, spatial_shapes,
            level_start_index, valid_ratios):
        decoder = self.teacher.decoder

        reused_intermediate = []
        tea_intermediate = []
        references = [reference_points]
        for layer_id, layer in enumerate(decoder.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = \
                    reference_points[:, :, None] * \
                    torch.cat([valid_ratios, valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = \
                    reference_points[:, :, None] * \
                    valid_ratios[:, None]
            reused_query = layer(
                query,
                query_pos=query_pos,
                value=stu_memory,
                key_padding_mask=memory_mask,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                reference_points=reference_points_input)
            query = layer(
                query,
                query_pos=query_pos,
                value=tea_memory,
                key_padding_mask=memory_mask,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                reference_points=reference_points_input)
            
            if self.with_box_refine:
                reg_branches = self.teacher.bbox_head.reg_branches
                tmp_reg_preds = reg_branches[layer_id](query)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp_reg_preds + inverse_sigmoid(
                        reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp_reg_preds
                    new_reference_points[..., :2] = tmp_reg_preds[
                        ..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            reused_intermediate.append(reused_query)
            tea_intermediate.append(query)
            references.append(reference_points)
        
        kd_inputs_dict = dict()
        kd_inputs_dict['reused_hidden_states'] = torch.stack(reused_intermediate)
        kd_inputs_dict['tea_hidden_states'] = torch.stack(tea_intermediate)
        kd_inputs_dict['references'] = references
        return kd_inputs_dict

    def cal_kd_loss(self, reused_hidden_states, tea_hidden_states, references, batch_data_samples):
        head = self.teacher.bbox_head
        reused_cls_scores_layers, reused_bbox_preds_layers = \
            head(reused_hidden_states, references)
        tea_cls_scores_layers, tea_bbox_preds_layers = \
            head(tea_hidden_states, references)

        kd_losses = dict()
        for layer_idx in range(reused_cls_scores_layers.size(0)):
            # class distillation
            cls_out_channels = head.cls_out_channels
            reused_cls_score = reused_cls_scores_layers[layer_idx]
            reused_cls_score = reused_cls_score.reshape(-1, cls_out_channels)
            tea_cls_score = tea_cls_scores_layers[layer_idx]
            tea_cls_score = tea_cls_score.reshape(-1, cls_out_channels)
            num_query = reused_cls_score.shape[0]
            loss_cls_kd = self.loss_cls_kd(
                reused_cls_score, tea_cls_score, avg_factor=num_query)
            kd_losses[f'd{layer_idx}.loss_cls_kd'] = loss_cls_kd
            
            # reg distillation
            reused_bbox_preds = reused_bbox_preds_layers[layer_idx]
            tea_bbox_preds = tea_bbox_preds_layers[layer_idx]
            factors = []
            for data_sample, bbox_pred in zip(
                batch_data_samples, reused_bbox_preds):
                img_meta = data_sample.metainfo
                img_h, img_w, = img_meta['img_shape']
                factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                            img_h]).unsqueeze(0).repeat(
                                                bbox_pred.size(0), 1)
                factors.append(factor)
            factors = torch.cat(factors, 0)

            reused_bbox_preds = reused_bbox_preds.reshape(-1, 4)
            reused_bboxes = bbox_cxcywh_to_xyxy(reused_bbox_preds) * factors
            tea_bbox_preds = tea_bbox_preds.reshape(-1, 4)
            tea_bboxes = bbox_cxcywh_to_xyxy(tea_bbox_preds) * factors
            reg_weights = tea_cls_score.sigmoid().max(dim=1, keepdim=True)[0]
            reg_weights = reg_weights.repeat(1, 4)
            loss_iou_kd = self.loss_iou_kd(
                reused_bboxes,
                tea_bboxes, 
                weight=reg_weights,
                avg_factor=reg_weights.sum())
            
            loss_reg_kd = self.loss_reg_kd(reused_bbox_preds, 
                                           tea_bbox_preds, 
                                           weight=reg_weights, 
                                           avg_factor=reg_weights.sum())
            
            kd_losses[f'd{layer_idx}.loss_reg_kd'] = loss_reg_kd
            kd_losses[f'd{layer_idx}.loss_iou_kd'] = loss_iou_kd
        
        return kd_losses
