# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os.path as osp
import numpy as np
import json_tricks as json
import pickle
import logging
import os
import copy
import cv2

from dataset.JointsDataset import JointsDataset
from utils.transforms import projectPoints

logger = logging.getLogger(__name__)

# TRAIN_LIST = [
#     '160422_ultimatum1',
#     '160224_haggling1',
#     '160226_haggling1',
#     '161202_haggling1',
#     '160906_ian1',
#     '160906_ian2',
#     '160906_ian3',
#     '160906_band1',
#     '160906_band2',
#     '160906_band3',
# ]
TRAIN_LIST = []

# VAL_LIST = ['b93c8262_o', 'd05eaeb3_o', 'fcb206cd_o']
if os.path.isdir('/mntnfs/med_data4/wangjiong/datasets/mvhuman'):
    VAL_LIST = os.listdir('/mntnfs/med_data4/wangjiong/datasets/mvhuman')
elif os.path.isdir('/home/yandanqi/0_data/MVHW'):
    VAL_LIST = os.listdir('/home/yandanqi/0_data/MVHW')
else:
    raise Exception(f'please check your VAL_LIST path')
VAL_LIST = [d for d in VAL_LIST if '_o' in d]   # leave dir with specific string '_o'

JOINTS_DEF = {
    'neck': 0,
    'nose': 1,
    'mid-hip': 2,
    'l-shoulder': 3,
    'l-elbow': 4,
    'l-wrist': 5,
    'l-hip': 6,
    'l-knee': 7,
    'l-ankle': 8,
    'r-shoulder': 9,
    'r-elbow': 10,
    'r-wrist': 11,
    'r-hip': 12,
    'r-knee': 13,
    'r-ankle': 14,
    # 'l-eye': 15,
    # 'l-ear': 16,
    # 'r-eye': 17,
    # 'r-ear': 18,
}

LIMBS = [[0, 1],
         [0, 2],
         [0, 3],
         [3, 4],
         [4, 5],
         [0, 9],
         [9, 10],
         [10, 11],
         [2, 6],
         [2, 12],
         [6, 7],
         [7, 8],
         [12, 13],
         [13, 14]]


class MVHW(JointsDataset):
    def __init__(self, cfg, image_set, is_train, transform=None):
        super().__init__(cfg, image_set, is_train, transform)
        self.pixel_std = 200.0
        self.joints_def = JOINTS_DEF
        self.limbs = LIMBS
        self.num_joints = len(JOINTS_DEF)

        if self.image_set == 'train':
            self.sequence_list = TRAIN_LIST
            self._interval = 3
            self.cam_list = [(0, 12), (0, 6), (0, 23), (0, 13), (0, 3)][:self.num_views]
            # self.cam_list = list(set([(0, n) for n in range(0, 31)]) - {(0, 12), (0, 6), (0, 23), (0, 13), (0, 3)})
            # self.cam_list.sort()
            self.num_views = len(self.cam_list)
        elif self.image_set == 'validation':
            self.sequence_list = VAL_LIST
            # self._interval = 12
            self._interval = 1
            self.cam_list = ['c0{}'.format(i) for i in range(1, 9, 1)][:self.num_views]   # [0, 1, 2, 3, 4, 5, 6, 7]
            self.num_views = len(self.cam_list)

        self.db_file = 'group_{}_cam{}.pkl'.format(self.image_set, self.num_views)
        self.db_file = os.path.join(self.dataset_root, self.db_file)

        # if osp.exists(self.db_file):
        #     info = pickle.load(open(self.db_file, 'rb'))
        #     assert info['sequence_list'] == self.sequence_list
        #     assert info['interval'] == self._interval
        #     assert info['cam_list'] == self.cam_list
        #     self.db = info['db']
        # else:
        #     self.db = self._get_db()
        #     info = {
        #         'sequence_list': self.sequence_list,
        #         'interval': self._interval,
        #         'cam_list': self.cam_list,
        #         'db': self.db
        #     }
        #     pickle.dump(info, open(self.db_file, 'wb'))
        self.db = self._get_db()
        self.db_size = len(self.db)

    @staticmethod
    def _get_img_name(idx):
        return str(idx+1).zfill(6)+'.jpg'

    def _get_db(self):
        width = 1920
        height = 1080
        db = []
        for seq in self.sequence_list:

            # get camera anno in this seq
            cameras = self._get_cam(seq)

            # get image dir
            img_dir = osp.join(self.dataset_root, seq, 'vframes')

            # get length of this seq
            seq_len = len(glob.glob(osp.join(img_dir, 'c01', '*.jpg')))

            # curr_anno = osp.join(self.dataset_root, seq, 'hdPose3d_stage1_coco19')
            # anno_files = sorted(glob.iglob('{:s}/*.json'.format(curr_anno)))

            for idx in range(seq_len):
                if idx % self._interval == 0:
                    # with open(file) as dfile:
                    #     bodies = json.load(dfile)['bodies']
                    # if len(bodies) == 0:
                    #     continue

                    for k, v in cameras.items():
                        # get each cam
                        our_cam = dict()
                        our_cam['R'] = v['R']
                        our_cam['T'] = -np.dot(v['R'].T, v['t']) * 10.0  # cm to mm
                        our_cam['fx'] = np.array(v['K'][0, 0])
                        our_cam['fy'] = np.array(v['K'][1, 1])
                        our_cam['cx'] = np.array(v['K'][0, 2])
                        our_cam['cy'] = np.array(v['K'][1, 2])
                        our_cam['k'] = v['distCoef'][[0, 1, 2]].reshape(3, 1)
                        our_cam['p'] = v['distCoef'][[3, 4]].reshape(2, 1)

                        # get image
                        img_path = osp.join(img_dir, k, self._get_img_name(idx))

                        # get pose
                        all_poses_3d = []
                        all_poses_vis_3d = []
                        all_poses = []
                        all_poses_vis = []
                        # fake pose
                        pose3d = np.zeros(shape=[self.num_joints, 3])
                        all_poses_3d.append(pose3d)
                        pose3d_vis = pose3d
                        all_poses_vis_3d.append(pose3d_vis)
                        pose2d = np.zeros(shape=[pose3d.shape[0], 2])
                        all_poses.append(pose2d)
                        pose2d_vis = pose2d
                        all_poses_vis.append(pose2d_vis)

                        db.append({
                            'key': "{}_{}-{}".format(seq, k, self._get_img_name(idx).split('.')[0]),
                            'image': img_path,
                            'joints_3d': all_poses_3d,
                            'joints_3d_vis': all_poses_vis_3d,
                            'joints_2d': all_poses,
                            'joints_2d_vis': all_poses_vis,
                            'camera': our_cam
                        })
        return db

    def _get_cam(self, seq):
        cam_file = osp.join(self.dataset_root, seq, 'cameras.json')
        with open(cam_file) as cfile:
            calib = json.load(cfile)

        M = np.array([[1.0, 0.0, 0.0],
                      [0.0, 0.0, -1.0],
                      [0.0, 1.0, 0.0]])
        cameras = {}
        for cam in calib:
            if cam['name'] in self.cam_list:    # # 'c01'-'c05' cams if num_list=5
                sel_cam = dict()
                sel_cam['K'] = np.array(cam['matrix'])
                sel_cam['distCoef'] = np.array(cam['distortions'])
                # sel_cam['R'] = np.array(cam['R']).dot(M)
                sel_cam['R'] = cv2.Rodrigues(np.array(cam['rotation']))[0].dot(M)
                sel_cam['t'] = np.array(cam['translation']).reshape((3, 1))
                cameras[cam['name']] = sel_cam
        return cameras

    def __getitem__(self, idx):
        input, target, weight, target_3d, meta, input_heatmap = [], [], [], [], [], []

        # if self.image_set == 'train':
        #     # camera_num = np.random.choice([5], size=1)
        #     select_cam = np.random.choice(self.num_views, size=5, replace=False)
        # elif self.image_set == 'validation':
        #     select_cam = list(range(self.num_views))

        for k in range(self.num_views):
            i, t, w, t3, m, ih = super().__getitem__(self.num_views * idx + k)
            if i is None:
                continue
            input.append(i)
            target.append(t)
            weight.append(w)
            target_3d.append(t3)
            meta.append(m)
            input_heatmap.append(ih)
        return input, target, weight, target_3d, meta, input_heatmap

    def __len__(self):
        return self.db_size // self.num_views

    def evaluate(self, preds):
        eval_list = []
        gt_num = self.db_size // self.num_views
        assert len(preds) == gt_num, 'number mismatch'

        total_gt = 0
        for i in range(gt_num):
            index = self.num_views * i
            db_rec = copy.deepcopy(self.db[index])
            joints_3d = db_rec['joints_3d']
            joints_3d_vis = db_rec['joints_3d_vis']

            if len(joints_3d) == 0:
                continue

            pred = preds[i].copy()
            pred = pred[pred[:, 0, 3] >= 0]
            for pose in pred:
                mpjpes = []
                for (gt, gt_vis) in zip(joints_3d, joints_3d_vis):
                    vis = gt_vis[:, 0] > 0
                    mpjpe = np.mean(np.sqrt(np.sum((pose[vis, 0:3] - gt[vis]) ** 2, axis=-1)))
                    mpjpes.append(mpjpe)
                min_gt = np.argmin(mpjpes)
                min_mpjpe = np.min(mpjpes)
                score = pose[0, 4]
                eval_list.append({
                    "mpjpe": float(min_mpjpe),
                    "score": float(score),
                    "gt_id": int(total_gt + min_gt)
                })

            total_gt += len(joints_3d)

        mpjpe_threshold = np.arange(25, 155, 25)
        aps = []
        recs = []
        for t in mpjpe_threshold:
            ap, rec = self._eval_list_to_ap(eval_list, total_gt, t)
            aps.append(ap)
            recs.append(rec)

        return aps, recs, self._eval_list_to_mpjpe(eval_list), self._eval_list_to_recall(eval_list, total_gt)

    @staticmethod
    def _eval_list_to_ap(eval_list, total_gt, threshold):
        eval_list.sort(key=lambda k: k["score"], reverse=True)
        total_num = len(eval_list)

        tp = np.zeros(total_num)
        fp = np.zeros(total_num)
        gt_det = []
        for i, item in enumerate(eval_list):
            if item["mpjpe"] < threshold and item["gt_id"] not in gt_det:
                tp[i] = 1
                gt_det.append(item["gt_id"])
            else:
                fp[i] = 1
        tp = np.cumsum(tp)
        fp = np.cumsum(fp)
        recall = tp / (total_gt + 1e-5)
        precise = tp / (tp + fp + 1e-5)
        for n in range(total_num - 2, -1, -1):
            precise[n] = max(precise[n], precise[n + 1])

        precise = np.concatenate(([0], precise, [0]))
        recall = np.concatenate(([0], recall, [1]))
        index = np.where(recall[1:] != recall[:-1])[0]
        ap = np.sum((recall[index + 1] - recall[index]) * precise[index + 1])

        return ap, recall[-2]

    @staticmethod
    def _eval_list_to_mpjpe(eval_list, threshold=500):
        eval_list.sort(key=lambda k: k["score"], reverse=True)
        gt_det = []

        mpjpes = []
        for i, item in enumerate(eval_list):
            if item["mpjpe"] < threshold and item["gt_id"] not in gt_det:
                mpjpes.append(item["mpjpe"])
                gt_det.append(item["gt_id"])

        return np.mean(mpjpes) if len(mpjpes) > 0 else np.inf

    @staticmethod
    def _eval_list_to_recall(eval_list, total_gt, threshold=500):
        gt_ids = [e["gt_id"] for e in eval_list if e["mpjpe"] < threshold]

        return len(np.unique(gt_ids)) / total_gt




