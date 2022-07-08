# Copyright (c) OpenMMLab. All rights reserved.
import unittest
from unittest import TestCase

import torch
from mmengine.config import ConfigDict
from mmengine.data import InstanceData
from parameterized import parameterized

from mmdet.core import mask_target
from mmdet.models.roi_heads.mask_heads import MaskIoUHead
from mmdet.models.utils import unpack_gt_instances
from mmdet.testing import (demo_mm_inputs, demo_mm_proposals,
                           demo_mm_sampling_results)


class TestMaskIoUHead(TestCase):

    @parameterized.expand(['cpu', 'cuda'])
    def test_mask_iou_head_loss_and_target(self, device):
        if device == 'cuda':
            if not torch.cuda.is_available():
                return unittest.skip('test requires GPU and torch+cuda')

        mask_iou_head = MaskIoUHead(num_classes=4)
        mask_iou_head.to(device=device)

        s = 256
        image_shapes = [(3, s, s)]
        packed_inputs = demo_mm_inputs(
            batch_size=1,
            image_shapes=image_shapes,
            num_items=[1],
            num_classes=4,
            with_mask=True)
        proposals_list = demo_mm_proposals(
            image_shapes=image_shapes, num_proposals=100)
        batch_data_samples = []
        for i in range(len(packed_inputs)):
            batch_data_samples.append(
                packed_inputs[i]['data_sample'].to(device=device))
            proposals_list[i] = proposals_list[i].to(device=device)
        train_cfg = ConfigDict(dict(mask_size=28, mask_thr_binary=0.5))

        # prepare ground truth
        data_samples = [inputs['data_sample'] for inputs in packed_inputs]
        (batch_gt_instances, batch_gt_instances_ignore,
         _) = unpack_gt_instances(data_samples)
        sampling_results = demo_mm_sampling_results(
            proposals_list=proposals_list,
            batch_gt_instances=batch_gt_instances,
            batch_gt_instances_ignore=batch_gt_instances_ignore)

        # prepare mask feats, pred and target
        pos_proposals = [res.pos_priors for res in sampling_results]
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]
        gt_masks = [res.masks for res in batch_gt_instances]
        mask_targets = mask_target(pos_proposals, pos_assigned_gt_inds,
                                   gt_masks, train_cfg)
        mask_feats = torch.rand((mask_targets.size(0), 256, 14, 14)).to(device)
        mask_preds = torch.rand((mask_targets.size(0), 4, 28, 28)).to(device)

        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        pos_mask_pred = mask_preds[range(mask_preds.size(0)), pos_labels]
        mask_iou_pred = mask_iou_head(mask_feats, pos_mask_pred)
        pos_mask_iou_pred = mask_iou_pred[range(mask_iou_pred.size(0)),
                                          pos_labels]

        mask_iou_head.loss_and_target(pos_mask_iou_pred, pos_mask_pred,
                                      mask_targets, sampling_results,
                                      batch_gt_instances, train_cfg)

    @parameterized.expand(['cpu', 'cuda'])
    def test_mask_iou_head_predict_by_feat(self, device):
        if device == 'cuda':
            if not torch.cuda.is_available():
                return unittest.skip('test requires GPU and torch+cuda')

        mask_iou_head = MaskIoUHead(num_classes=4)
        mask_iou_head.to(device=device)

        s = 128
        num_samples = 2
        num_classes = 4
        img_metas = {
            'img_shape': (s, s, 3),
            'scale_factor': (1, 1),
            'ori_shape': (s, s, 3)
        }
        results = InstanceData(metainfo=img_metas)
        results.bboxes = torch.rand((num_samples, 4)).to(device)
        results.scores = torch.rand((num_samples, )).to(device)
        results.labels = torch.randint(
            num_classes, (num_samples, ), dtype=torch.long).to(device)

        mask_feats = torch.rand((num_samples, 256, 14, 14)).to(device)
        mask_preds = torch.rand((num_samples, num_classes, 28, 28)).to(device)
        mask_iou_preds = mask_iou_head(
            mask_feats, mask_preds[range(results.labels.size(0)),
                                   results.labels])

        mask_iou_head.predict_by_feat(
            mask_iou_preds=[mask_iou_preds], results_list=[results])
