"""
visualize the dataset you create
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import time

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import argparse
import os
from tqdm import tqdm

import _init_paths
from core.config import config
from core.config import update_config
from utils.utils import create_logger, load_backbone_panoptic
import dataset
import models

from utils.vis import save_dataset_gts_multi

import pdb


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument(
        '--cfg', help='experiment configure file name', required=True, type=str)

    args, rest = parser.parse_known_args()
    update_config(args.cfg)

    return args


def main():
    args = parse_args()
    # logger, final_output_dir, tb_log_dir = create_logger(
    #     config, args.cfg, 'eval_map')
    # cfg_name = os.path.basename(args.cfg).split('.')[0]

    # logger.info(pprint.pformat(args))
    # logger.info(pprint.pformat(config))

    # gpus = [int(i) for i in config.GPUS.split(',')]
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
        batch_size=config.TEST.BATCH_SIZE,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True)

    ''' ---------------- Visualization ----------------'''
    output_dir = './vis_results'
    for i, (inputs, targets_2d, weights_2d, targets_3d, meta, input_heatmap) in enumerate(tqdm(test_loader)):

        for k in range(len(inputs)):
            view_name = 'view_{}'.format(k + 1)
            prefix = '{}_{:08}_{}'.format(
                os.path.join(output_dir, 'test'), i, view_name)

            save_dataset_gts_multi(config, inputs[k], meta[k], targets_2d[k], prefix)
        # prefix2 = '{}_{:08}'.format(
        #     os.path.join(output_dir, 'train'), i)

        # save_debug_3d_cubes(config, meta[0], grid_centers, prefix2)
        # save_debug_3d_images(config, meta[0], pred, prefix2)

        # vis_batch_size = 7
        # fig = plt.figure(figsize=(14, 10))
        # n_rows = len(inputs)   # num_views
        # n_cols = vis_batch_size
        #
        # for k in range(len(inputs)):   # inputs[k] ---torch.Size([32, 3, 512, 960])
        #     #ax = fig.add_subplot(1, len(inputs), k+1)
        #
        #     # convert image (unnormalized, dtype, channel)
        #     #pdb.set_trace()
        #     imgs = inputs[k][:vis_batch_size]
        #     imgs = util.invert_normalized_tensor(in_tensor=imgs, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        #     imgs = imgs.mul(255).clamp(0, 255).permute(0, 2, 3, 1).detach().cpu().numpy().astype(np.int)
        #
        #     for j in range(n_cols):
        #         ax = fig.add_subplot(n_rows, n_cols, k*n_cols+j+1)
        #         ax.imshow(imgs[j])
        #         print(f"k={k}, j={j}, k*n_cols+j+1={k*n_cols+j+1}")
        #
        # save_path = './vis_results.png'
        #
        # plt.savefig(save_path)
        # print(f"=> visualize results have been saved in {save_path}...")

        break


    print(f"=> Finished all...")



if __name__ == "__main__":
    main()
