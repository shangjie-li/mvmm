from collections import namedtuple
import os
import numpy as np
import torch
import torch.nn as nn

from layers import rv_backbones, pv_bridges, bev_backbones, dense_heads
from ops.iou3d_nms import iou3d_nms_utils
from utils import model_nms_utils
from utils.spconv_utils import find_all_spconv_keys


class MVMM(nn.Module):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.dataset = dataset
        self.class_names = dataset.class_names
        self.register_buffer('global_step', torch.LongTensor(1).zero_())

        self.module_topology = [
            'rv_backbone', 'pv_bridge', 'bev_backbone', 'dense_head',
        ]
        self.module_list = self.build_modules()

    @property
    def mode(self):
        return 'TRAIN' if self.training else 'TEST'

    def update_global_step(self):
        self.global_step += 1

    def build_modules(self):
        model_info_dict = {
            'module_list': [],
            'point_cloud_range': self.dataset.point_cloud_range,
            'training': self.dataset.training,
            'num_point_features': self.dataset.used_point_features,
        }
        for module_name in self.module_topology:
            module, model_info_dict = getattr(self, 'build_%s' % module_name)(
                model_info_dict=model_info_dict
            )
            self.add_module(module_name, module)
        return model_info_dict['module_list']

    def build_rv_backbone(self, model_info_dict):
        if self.model_cfg.get('RV_BACKBONE', None) is None:
            return None, model_info_dict
        
        rv_backbone_module = rv_backbones.__all__[self.model_cfg.RV_BACKBONE.NAME](
            model_cfg=self.model_cfg.RV_BACKBONE,
            input_channels=model_info_dict['num_point_features'],
        )
        model_info_dict['module_list'].append(rv_backbone_module)
        model_info_dict['num_rv_features'] = rv_backbone_module.num_rv_features
        return rv_backbone_module, model_info_dict

    def build_pv_bridge(self, model_info_dict):
        if self.model_cfg.get('PV_BRIDGE', None) is None:
            return None, model_info_dict

        pv_bridge_module = pv_bridges.__all__[self.model_cfg.PV_BRIDGE.NAME](
            model_cfg=self.model_cfg.PV_BRIDGE,
            input_channels=model_info_dict['num_rv_features'] if 'num_rv_features' in model_info_dict else 0,
            point_cloud_range=model_info_dict['point_cloud_range'],
            training=model_info_dict['training'],
        )
        model_info_dict['module_list'].append(pv_bridge_module)
        model_info_dict['num_pv_features'] = pv_bridge_module.num_pv_features
        model_info_dict['grid_size'] = pv_bridge_module.grid_size
        return pv_bridge_module, model_info_dict

    def build_bev_backbone(self, model_info_dict):
        if self.model_cfg.get('BEV_BACKBONE', None) is None:
            return None, model_info_dict

        bev_backbone_module = bev_backbones.__all__[self.model_cfg.BEV_BACKBONE.NAME](
            model_cfg=self.model_cfg.BEV_BACKBONE,
            input_channels=model_info_dict['num_pv_features'],
        )
        model_info_dict['module_list'].append(bev_backbone_module)
        model_info_dict['num_bev_features'] = bev_backbone_module.num_bev_features
        return bev_backbone_module, model_info_dict

    def build_dense_head(self, model_info_dict):
        if self.model_cfg.get('DENSE_HEAD', None) is None:
            return None, model_info_dict
        dense_head_module = dense_heads.__all__[self.model_cfg.DENSE_HEAD.NAME](
            model_cfg=self.model_cfg.DENSE_HEAD,
            input_channels=model_info_dict['num_bev_features'],
            num_class=self.num_class,
            class_names=self.class_names,
            grid_size=model_info_dict['grid_size'],
            point_cloud_range=model_info_dict['point_cloud_range'],
        )
        model_info_dict['module_list'].append(dense_head_module)
        return dense_head_module, model_info_dict

    def _load_state_dict(self, model_state_disk, *, strict=True):
        state_dict = self.state_dict()  # local cache of state_dict

        spconv_keys = find_all_spconv_keys(self)

        update_model_state = {}
        for key, val in model_state_disk.items():
            if key in spconv_keys and key in state_dict and state_dict[key].shape != val.shape:
                # with different spconv versions, we need to adapt weight shapes for spconv blocks
                # adapt spconv weights from version 1.x to version 2.x if you used weights from spconv 1.x

                val_native = val.transpose(-1, -2)  # (k1, k2, k3, c_in, c_out) to (k1, k2, k3, c_out, c_in)
                if val_native.shape == state_dict[key].shape:
                    val = val_native.contiguous()
                else:
                    assert val.shape.__len__() == 5, 'currently only spconv 3D is supported'
                    val_implicit = val.permute(4, 0, 1, 2, 3)  # (k1, k2, k3, c_in, c_out) to (c_out, k1, k2, k3, c_in)
                    if val_implicit.shape == state_dict[key].shape:
                        val = val_implicit.contiguous()

            if key in state_dict and state_dict[key].shape == val.shape:
                update_model_state[key] = val
                # logger.info('Update weight %s: %s' % (key, str(val.shape)))

        if strict:
            self.load_state_dict(update_model_state)
        else:
            state_dict.update(update_model_state)
            self.load_state_dict(state_dict)
        return state_dict, update_model_state

    def load_params_from_file(self, filename, logger, to_cpu=False):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        model_state_disk = checkpoint['model_state']

        version = checkpoint.get("version", None)
        if version is not None:
            logger.info('==> Checkpoint trained from version: %s' % version)

        state_dict, update_model_state = self._load_state_dict(model_state_disk, strict=False)

        for key in state_dict:
            if key not in update_model_state:
                logger.info('Not updated weight %s: %s' % (key, str(state_dict[key].shape)))

        logger.info('==> Done (loaded %d/%d)' % (len(update_model_state), len(state_dict)))

    def load_params_with_optimizer(self, filename, to_cpu=False, optimizer=None, logger=None):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        epoch = checkpoint.get('epoch', -1)
        it = checkpoint.get('it', 0.0)

        self._load_state_dict(checkpoint['model_state'], strict=True)

        if optimizer is not None:
            if 'optimizer_state' in checkpoint and checkpoint['optimizer_state'] is not None:
                logger.info('==> Loading optimizer parameters from checkpoint %s to %s'
                            % (filename, 'CPU' if to_cpu else 'GPU'))
                optimizer.load_state_dict(checkpoint['optimizer_state'])
            else:
                assert filename[-4] == '.', filename
                src_file, ext = filename[:-4], filename[-3:]
                optimizer_filename = '%s_optim.%s' % (src_file, ext)
                if os.path.exists(optimizer_filename):
                    optimizer_ckpt = torch.load(optimizer_filename, map_location=loc_type)
                    optimizer.load_state_dict(optimizer_ckpt['optimizer_state'])

        if 'version' in checkpoint:
            print('==> Checkpoint trained from version: %s' % checkpoint['version'])
        logger.info('==> Done')

        return it, epoch

    @staticmethod
    def generate_recall_record(recall_dict, box_preds, gt_boxes, thresh_list):
        if recall_dict.__len__() == 0:
            recall_dict = {'gt': 0}
            for cur_thresh in thresh_list:
                recall_dict['roi_%s' % (str(cur_thresh))] = 0
                recall_dict['rcnn_%s' % (str(cur_thresh))] = 0
        
        cur_gt = gt_boxes
        k = cur_gt.__len__() - 1
        while k > 0 and cur_gt[k].sum() == 0:
            k -= 1
        cur_gt = cur_gt[:k + 1]
        
        if cur_gt.shape[0] > 0:
            recall_dict['gt'] += cur_gt.shape[0]
            if box_preds.shape[0] > 0:
                iou3d_rcnn = iou3d_nms_utils.boxes_iou3d_gpu(box_preds[:, 0:7], cur_gt[:, 0:7])
            else:
                iou3d_rcnn = torch.zeros((0, cur_gt.shape[0]))
            for cur_thresh in thresh_list:
                if iou3d_rcnn.shape[0] == 0:
                    recall_dict['rcnn_%s' % str(cur_thresh)] += 0
                else:
                    recall_dict['rcnn_%s' % str(cur_thresh)] += (iou3d_rcnn.max(dim=0)[0] > cur_thresh).sum().item()
        
        return recall_dict

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)
        
        if self.training:
            ret_dict, tb_dict, disp_dict = {}, {}, {}
            
            cls_loss = self.dense_head.get_cls_loss(batch_dict)
            loc_loss, dir_loss = self.dense_head.get_box_loss(batch_dict)
            tb_dict['cls_loss'] = cls_loss.item()
            tb_dict['loc_loss'] = loc_loss.item()
            tb_dict['dir_loss'] = dir_loss.item()
            
            all_loss = cls_loss + loc_loss + dir_loss
            
            ret_dict['all_loss'] = all_loss
            tb_dict['all_loss'] = all_loss.item()
            
            return ret_dict, tb_dict, disp_dict
            
        else:
            batch_size = batch_dict['batch_size']
            pred_dicts = []
            recall_dict = {}
            
            for batch_idx in range(batch_size):
                box_preds = batch_dict['box_preds_for_testing'][batch_idx] # (num_boxes, 7)
                cls_preds = batch_dict['cls_preds_for_testing'][batch_idx] # (num_boxes, num_class)
                
                score_preds, label_preds = torch.max(torch.sigmoid(cls_preds), dim=-1)
                label_preds += 1
                
                selected, selected_scores = model_nms_utils.class_agnostic_nms(
                    box_scores=score_preds, box_preds=box_preds,
                    nms_config=self.model_cfg.POST_PROCESSING.NMS_CONFIG,
                    score_thresh=self.model_cfg.POST_PROCESSING.SCORE_THRESH
                )
                
                box_preds = box_preds[selected]
                label_preds = label_preds[selected]
                score_preds = selected_scores
                
                pred_dicts.append({
                    'pred_boxes': box_preds,
                    'pred_labels': label_preds,
                    'pred_scores': score_preds,
                })
                
                if 'gt_boxes' in batch_dict:
                    gt_boxes = batch_dict['gt_boxes'][batch_idx]
                    recall_dict = self.generate_recall_record(recall_dict=recall_dict, box_preds=box_preds, gt_boxes=gt_boxes,
                        thresh_list=self.model_cfg.POST_PROCESSING.RECALL_THRESH_LIST)
            
            return pred_dicts, recall_dict


def build_network(model_cfg, num_class, dataset):
    model = MVMM(
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )
    return model


def load_data_to_gpu(batch_dict):
    for key, val in batch_dict.items():
        if not isinstance(val, np.ndarray):
            continue
        elif key in ['frame_id', 'metadata', 'calib']:
            continue
        elif key in ['images']:
            batch_dict[key] = torch.from_numpy(val).float().cuda().contiguous()
        elif key in ['image_shape']:
            batch_dict[key] = torch.from_numpy(val).int().cuda()
        else:
            batch_dict[key] = torch.from_numpy(val).float().cuda()


def model_fn_decorator():
    ModelReturn = namedtuple('ModelReturn', ['loss', 'tb_dict', 'disp_dict'])

    def model_func(model, batch_dict):
        load_data_to_gpu(batch_dict)
        ret_dict, tb_dict, disp_dict = model(batch_dict)

        loss = ret_dict['all_loss'].mean()
        if hasattr(model, 'update_global_step'):
            model.update_global_step()
        else:
            model.module.update_global_step()

        return ModelReturn(loss, tb_dict, disp_dict)

    return model_func
