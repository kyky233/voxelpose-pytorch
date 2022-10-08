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
    if gpus[0] != -1:
        with torch.no_grad():
            model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    test_model_file = os.path.join(final_output_dir, config.TEST.MODEL_FILE)
    logger.info('=> load models state {}'.format(test_model_file))
    if config.TEST.MODEL_FILE and os.path.isfile(test_model_file):
        if gpus[0] != -1:
            model.module.load_state_dict(torch.load(test_model_file))
        else:
            model.load_state_dict(torch.load(test_model_file))
    else:
        raise ValueError('Check the model file for testing!')

    model.eval()
    num_view = 5
    preds = []
    heatmaps_all = []
    grid_centers_all = []
    inputs_all = []
    meta_all = []
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

            pred = pred.detach().cpu().numpy()  #[num_view, batch_size, c, h, w]
            #all_heatmaps = all_heatmaps.detach().cpu().numpy()     #[num_view, batch_size, num_joints, h', w']
            grid_centers = grid_centers.detach().cpu().numpy()
            #inputs = inputs.detach().cpu().numpy()     #[num_view, batch_size, c, h, w]
            for b in range(pred.shape[0]):
                preds.append(pred[b])
                grid_centers_all.append(grid_centers[b])

                # heatmap & inputs
                heatmap_multi = []
                input_multi = []
                meta_multi = []
                for j in range(num_view):
                    heatmap_multi.append(all_heatmaps[j][b].detach().cpu().numpy())
                    input_multi.append(inputs[j][b].detach().cpu().numpy())
                    meta_single = dict()
                    for k, v in meta[j].items():
                        if k != 'camera':
                            meta_single[k] = v[b]
                        else:   # k == 'camera'
                            camera_single = dict()
                            for k_c, v_c in meta[j][k].items():
                                camera_single[k_c] = v_c[b]
                            meta_single[k] = camera_single
                    meta_multi.append(meta_single)

                heatmaps_all.append(heatmap_multi)
                inputs_all.append(input_multi)
                meta_all.append(meta_multi)

            if i == 1:
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

    # save pred results
    results_all = {
        'inputs': inputs_all,
        'preds': preds,
        'heatmaps': heatmaps_all,
        'grid_centers': grid_centers_all,
        'metas': meta_all,
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
