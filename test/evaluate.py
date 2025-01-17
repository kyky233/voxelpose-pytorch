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


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument(
        '--cfg', help='experiment configure file name', required=True, type=str)

    args, rest = parser.parse_known_args()
    update_config(args.cfg)

    return args


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
    model = eval('models.' + config.MODEL + '.get_multi_person_pose_net')(
        config, is_train=True)
    with torch.no_grad():
        model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    test_model_file = os.path.join(final_output_dir, config.TEST.MODEL_FILE)
    if config.TEST.MODEL_FILE and os.path.isfile(test_model_file):
        logger.info('=> load models state {}'.format(test_model_file))
        model.module.load_state_dict(torch.load(test_model_file))
    else:
        raise ValueError('Check the model file for testing!')

    model.eval()
    preds = []
    heatmaps_all = []
    grid_centers_all = []
    inputs_all = []
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

            pred = pred.detach().cpu().numpy()
            all_heatmaps = all_heatmaps.detach().cpu().numpy()
            grid_centers = grid_centers.detach().cpu().numpy()
            inputs = inputs.detach().cpu().numpy()
            for b in range(pred.shape[0]):
                preds.append(pred[b])
                heatmaps_all.append(all_heatmaps[b])
                grid_centers_all.append(grid_centers[b])
                inputs_all.append(inputs[b])

        tb = PrettyTable()
        if 'panoptic' or 'mvhw' in config.DATASET.TEST_DATASET:
            mpjpe_threshold = np.arange(25, 155, 25)
            aps, recs, mpjpe, _ = test_dataset.evaluate(preds)
            tb.field_names = ['Threshold/mm'] + [f'{i}' for i in mpjpe_threshold]
            tb.add_row(['AP'] + [f'{ap * 100:.2f}' for ap in aps])
            tb.add_row(['Recall'] + [f'{re * 100:.2f}' for re in recs])
            print(tb)
            print(f'MPJPE: {mpjpe:.2f}mm')
        else:
            actor_pcp, avg_pcp, bone_person_pcp, _ = test_dataset.evaluate(preds)
            tb.field_names = ['Bone Group'] + [f'Actor {i+1}' for i in range(len(actor_pcp))] + ['Average']
            for k, v in bone_person_pcp.items():
                tb.add_row([k] + [f'{i*100:.1f}' for i in v] + [f'{np.mean(v)*100:.1f}'])
            tb.add_row(['Total'] + [f'{i*100:.1f}' for i in actor_pcp] + [f'{avg_pcp*100:.1f}'])
            print(tb)

    # save pred results
    results_all = {
        'inputs': inputs_all,
        'preds': preds,
        'heatmaps': heatmaps_all,
        'grid_centers': grid_centers_all
    }
    save_dir = './results'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    save_path = os.path.join(save_dir, config.DATASET.TEST_DATASET+time.strftime("_%Y-%m-%d-%H-%M", time.localtime())+'.pickle')
    with open(save_path, 'wb') as f:
        pickle.dump(results_all, f)
    print(f"=> pred results have been saved in {os.path.abspath(save_path)}...")


if __name__ == "__main__":
    main()
