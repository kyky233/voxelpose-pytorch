# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import time

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import argparse
import os
from tqdm import tqdm
from prettytable import PrettyTable
import pprint
import copy

import _init_paths
from core.config import config
from core.config import update_config
from utils.utils import create_logger, load_backbone_panoptic
import dataset
import models

from utils.vis import save_debug_images_multi
from utils.vis import save_debug_3d_images
from utils.vis import save_debug_3d_cubes
from utils.vis import save_batch_heatmaps_multi


save_pred_data = False
save_pred_vis = True


output_dir = './results'


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument(
        '--cfg', help='experiment configure file name', required=True, type=str)

    args, rest = parser.parse_known_args()
    update_config(args.cfg)

    return args


# by cpu version
def load_backbone_coco(model, pretrained_file):
    this_dir = os.path.dirname(__file__)
    # pretrained_file = os.path.abspath(os.path.join(this_dir, '../..', pretrained_file))
    pretrained_file = os.path.abspath(os.path.join(this_dir, '../', pretrained_file))
    pretrained_state_dict = torch.load(pretrained_file, map_location=torch.device('cpu'))
    model_state_dict = model.backbone.state_dict()

    prefix = "backbone."     # 'module'
    new_pretrained_state_dict = {}
    for k, v in pretrained_state_dict['state_dict'].items():
        if k.replace(prefix, "") in model_state_dict and v.shape == model_state_dict[k.replace(prefix, "")].shape:
            new_pretrained_state_dict[k.replace(prefix, "")] = v
        elif k.replace(prefix, "") == "final_layer.weight":  # TODO
            print("Reiniting final layer filters:", k)

            o = torch.zeros_like(model_state_dict[k.replace(prefix, "")][:, :, :, :])
            nn.init.xavier_uniform_(o)
            n_filters = min(o.shape[0], v.shape[0])
            o[:n_filters, :, :, :] = v[:n_filters, :, :, :]

            new_pretrained_state_dict[k.replace(prefix, "")] = o
        elif k.replace(prefix, "") == "final_layer.bias":
            print("Reiniting final layer biases:", k)
            o = torch.zeros_like(model_state_dict[k.replace(prefix, "")][:])
            nn.init.zeros_(o)
            n_filters = min(o.shape[0], v.shape[0])
            o[:n_filters] = v[:n_filters]

            new_pretrained_state_dict[k.replace(prefix, "")] = o
    print(f"load backbone statedict from {pretrained_file}")
    model.backbone.load_state_dict(new_pretrained_state_dict)

    return model


def main():
    args = parse_args()
    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'eval_map')
    cfg_name = os.path.basename(args.cfg).split('.')[0]

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    gpus = [int(i) for i in config.GPUS.split(',')]
    print('=> Loading data ..')
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    test_dataset = eval('dataset.' + config.DATASET.TEST_DATASET)(
        config, config.DATASET.TEST_SUBSET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.TEST.BATCH_SIZE * len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True)

    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    print('=> Constructing models ..')
    # load model with mapping to cpu
    model = eval('models.' + config.MODEL + '.get_multi_person_pose_net')(
        config, is_train=True)

    model = load_backbone_coco(model, config.NETWORK.PRETRAINED_BACKBONE)

    model.eval()
    with torch.no_grad():
        for i, (inputs, targets_2d, weights_2d, targets_3d, meta, input_heatmap) in enumerate(tqdm(test_loader)):
            if 'panoptic' or 'mvhw' in config.DATASET.TEST_DATASET:
                # pred, _, _, _, _, _ = model(views=inputs, meta=meta)
                pred, all_heatmaps, grid_centers, _, _, _ = model(views=inputs, meta=meta)
            elif 'campus' in config.DATASET.TEST_DATASET or 'shelf' in config.DATASET.TEST_DATASET:
                # pred, _, _, _, _, _ = model(meta=meta, input_heatmaps=input_heatmap)
                pred, all_heatmaps, grid_centers, _, _, _ = model(meta=meta, input_heatmaps=input_heatmap)
            else:
                raise Exception(f"use dataset {config.DATASET.TEST_DATASET} for test, which is not declared here...")

            prefix2 = '{}_{:08}'.format(os.path.join(output_dir, 'train'), i)
            save_batch_heatmaps_multi(input, targets_2d, '{}_hm_gt.jpg'.format(prefix2))
            save_batch_heatmaps_multi(input, all_heatmaps, '{}_hm_pred.jpg'.format(prefix2))
            save_debug_3d_cubes(config, meta[0], grid_centers, prefix2)
            save_debug_3d_images(config, meta[0], pred, prefix2)

            if i == 7:
                break

        # tb = PrettyTable()
        # if 'panoptic' or 'mvhw' in config.DATASET.TEST_DATASET:
        #     mpjpe_threshold = np.arange(25, 155, 25)
        #     aps, recs, mpjpe, _ = test_dataset.evaluate(preds)
        #     tb.field_names = ['Threshold/mm'] + [f'{i}' for i in mpjpe_threshold]
        #     tb.add_row(['AP'] + [f'{ap * 100:.2f}' for ap in aps])
        #     tb.add_row(['Recall'] + [f'{re * 100:.2f}' for re in recs])
        #     print(tb)
        #     print(f'MPJPE: {mpjpe:.2f}mm')
        # else:
        #     actor_pcp, avg_pcp, bone_person_pcp, _ = test_dataset.evaluate(preds)
        #     tb.field_names = ['Bone Group'] + [f'Actor {i+1}' for i in range(len(actor_pcp))] + ['Average']
        #     for k, v in bone_person_pcp.items():
        #         tb.add_row([k] + [f'{i*100:.1f}' for i in v] + [f'{np.mean(v)*100:.1f}'])
        #     tb.add_row(['Total'] + [f'{i*100:.1f}' for i in actor_pcp] + [f'{avg_pcp*100:.1f}'])
        #     print(tb)

        print(f"=> finished all...")


if __name__ == "__main__":
    main()
