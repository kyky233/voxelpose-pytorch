"""
use this script to check if my dataset can load target data
by: ydq
"""
import copy
import os
import glob
import json
import numpy as np
import torch

import cv2
import matplotlib.pyplot as plt


dataset_root = '/home/yandanqi/0_data/MVHW'
sequence_list = os.listdir('/home/yandanqi/0_data/MVHW')
sequence_list = [d for d in sequence_list if '_o' in d]

num_views = 5
cam_list = ['c0{}'.format(i) for i in range(1, 9, 1)][:num_views]

_interval = 1
num_joints = 15


def get_3rd_point(a, b):
    direct = a - b
    return np.array(b) + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def get_scale(image_size, resized_size):
    w, h = image_size
    w_resized, h_resized = resized_size
    if w / w_resized < h / h_resized:
        w_pad = h / h_resized * w_resized
        h_pad = h
    else:
        w_pad = w
        h_pad = w / w_resized * h_resized
    scale = np.array([w_pad / 200.0, h_pad / 200.0], dtype=np.float32)

    return scale


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if isinstance(scale, torch.Tensor):
        scale = np.array(scale.cpu())
    if isinstance(center, torch.Tensor):
        center = np.array(center.cpu())
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale])

    scale_tmp = scale * 200.0
    src_w, src_h = scale_tmp[0], scale_tmp[1]
    dst_w, dst_h = output_size[0], output_size[1]

    rot_rad = np.pi * rot / 180
    if src_w >= src_h:
        src_dir = get_dir([0, src_w * -0.5], rot_rad)
        dst_dir = np.array([0, dst_w * -0.5], np.float32)
    else:
        src_dir = get_dir([src_h * -0.5, 0], rot_rad)
        dst_dir = np.array([dst_h * -0.5, 0], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift     # x,y
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def _get_img_name(idx):
    return str(idx+1).zfill(6)+'.jpg'


def _get_cam(seq):
    cam_file = os.path.join(dataset_root, seq, 'cameras.json')
    with open(cam_file) as cfile:
        calib = json.load(cfile)

    M = np.array([[1.0, 0.0, 0.0],
                  [0.0, 0.0, -1.0],
                  [0.0, 1.0, 0.0]])
    cameras = {}
    for cam in calib:
        if cam['name'] in cam_list:    # # 'c01'-'c05' cams if num_list=5
            sel_cam = dict()
            sel_cam['K'] = np.array(cam['matrix'])
            sel_cam['distCoef'] = np.array(cam['distortions'])
            # sel_cam['R'] = np.array(cam['R']).dot(M)
            sel_cam['R'] = cv2.Rodrigues(np.array(cam['rotation']))[0].dot(M)
            sel_cam['t'] = np.array(cam['translation']).reshape((3, 1))
            cameras[cam['name']] = sel_cam
    return cameras


def _get_db():
    width = 1920
    height = 1080
    db = []
    for seq in sequence_list:

        # get camera anno in this seq
        cameras = _get_cam(seq)

        # get image dir
        img_dir = os.path.join(dataset_root, seq, 'vframes')

        # get length of this seq
        seq_len = len(glob.glob(os.path.join(img_dir, 'c01', '*.jpg')))

        # curr_anno = osp.join(self.dataset_root, seq, 'hdPose3d_stage1_coco19')
        # anno_files = sorted(glob.iglob('{:s}/*.json'.format(curr_anno)))

        for idx in range(seq_len):
            if idx % _interval == 0:
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
                    img_path = os.path.join(img_dir, k, _get_img_name(idx))

                    # get pose
                    all_poses_3d = []
                    all_poses_vis_3d = []
                    all_poses = []
                    all_poses_vis = []
                    # fake pose
                    pose3d = np.zeros(shape=[num_joints, 3])
                    all_poses_3d.append(pose3d)
                    pose3d_vis = pose3d
                    all_poses_vis.append(pose3d_vis)
                    pose2d = np.zeros(shape=[pose3d.shape[0], 2])
                    all_poses.append(pose2d)
                    pose2d_vis = pose2d
                    all_poses_vis.append(pose2d_vis)

                    db.append({
                        'key': "{}_{}-{}".format(seq, k, _get_img_name(idx).split('.')[0]),
                        'image': img_path,
                        'joints_3d': all_poses_3d,
                        'joints_3d_vis': all_poses_vis_3d,
                        'joints_2d': all_poses,
                        'joints_2d_vis': all_poses_vis,
                        'camera': our_cam
                    })
    return db


def _get_group_item(db, idx):
    """
    db_rec:
        'key': "{}_{}-{}".format(seq, k, _get_img_name(idx).split('.')[0]),
        'image': img_path,
        'joints_3d': all_poses_3d,
        'joints_3d_vis': all_poses_vis_3d,
        'joints_2d': all_poses,
        'joints_2d_vis': all_poses_vis,
        'camera': our_cam
    """
    keys = []
    cameras = []
    images = []

    # collect group data
    for k in range(num_views):
        cur_idx = num_views * idx + k

        db_rec = copy.deepcopy(db[cur_idx])
        keys.append(db_rec['key'])
        cameras.append(db_rec['camera'])
        images.append(db_rec['image'])

    group_rec = dict()
    group_rec['key'] = keys
    group_rec['camera'] = cameras
    group_rec['image'] = images

    return group_rec


def preprocess_image(img_path, with_vis=False):
    image_size = np.array([960, 512])
    color_rgb = True

    # read image
    data_numpy = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if color_rgb:
        data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)

    # resize image to the given self.image_size
    height, width, _ = data_numpy.shape
    c = np.array([width / 2.0, height / 2.0])
    s = get_scale((width, height), image_size)
    r = 0

    trans = get_affine_transform(c, s, r, image_size)
    input = cv2.warpAffine(
        data_numpy,
        trans, (int(image_size[0]), int(image_size[1])),
        flags=cv2.INTER_LINEAR)

    # show image
    if with_vis:
        cv2.imshow("Before", data_numpy)
        cv2.imshow("After", input)
        cv2.waitKey(0)

        cv2.destroyAllWindows()

    return input


def show_db_item(db_rec, with_preprocess=True):
    """
    db_rec:
        'key': "{}_{}-{}".format(seq, k, _get_img_name(idx).split('.')[0]),
        'image': img_path,
        'joints_3d': all_poses_3d,
        'joints_3d_vis': all_poses_vis_3d,
        'joints_2d': all_poses,
        'joints_2d_vis': all_poses_vis,
        'camera': our_cam
    """
    print(f"key of this rec = {db_rec['key']}")
    print(f"camera = {db_rec['camera']}")
    print(f"shape of joints_3d = {len(db_rec['joints_3d'])}")

    fig = plt.figure(figsize=(1, 1))
    ax = fig.add_subplot(1, 1, 1)
    if with_preprocess:
        img = preprocess_image(img_path=db_rec['image'], with_vis=False)
    else:
        img = plt.imread(db_rec['image'])
    ax.imshow(img)

    plt.show()
    plt.close()


def show_db_single_group(db_group_rec, with_preprocess=True):
    """
    db_group_rec:
        'key': "{}_{}-{}".format(seq, k, _get_img_name(idx).split('.')[0]),
        'image': img_path,
        'camera': our_cam
    """
    keys = db_group_rec['key']
    cameras = db_group_rec['camera']
    images = db_group_rec['image']

    # show group data
    for i in range(num_views):
        print(f"key of this rec = {keys[i]}")
        print(f"camera = {cameras[i]}")
        print(f"image = {images[i]}")

    fig = plt.figure(figsize=(1, num_views))
    for i in range(num_views):
        ax = fig.add_subplot(1, num_views, i+1)
        if with_preprocess:
            img = preprocess_image(img_path=images[i], with_vis=False)
        else:
            img = plt.imread(images[i])
        ax.imshow(img)

    plt.show()
    plt.close()


def show_db_groups(db, num_groups, group_intervals, idx, with_preprocess=True):
    """
    db_group_rec:
        'key': "{}_{}-{}".format(seq, k, _get_img_name(idx).split('.')[0]),
        'image': img_path,
        'camera': our_cam
    """
    # collect data
    keys = []
    cameras = []
    images = []

    for i in range(num_groups):
        db_group_rec = _get_group_item(db=db, idx=idx + i*group_intervals)
        keys.append(db_group_rec['key'])
        cameras.append(db_group_rec['camera'])
        images.append(db_group_rec['image'])

    # show data
    for i in range(num_groups):
        print(f"key of db[{idx + i*group_intervals}] = {keys[i]}")
        print(f"image of db[{idx + i*group_intervals}]= {images[i]}")

    fig = plt.figure()
    for i in range(num_groups):
        for j in range(num_views):
            ax = fig.add_subplot(num_groups, num_views, i*num_views + j+1)
            if with_preprocess:
                img = preprocess_image(img_path=images[i][j], with_vis=False)
            else:
                img = plt.imread(images[i][j])
            ax.imshow(img)

    plt.show()
    plt.close()


def main():
    db = _get_db()
    print(f"length of dataset = {len(db)}")

    idx = 0

    vis_item = False
    vis_single_group = True
    vis_groups = False
    with_preprocess = True

    # # show processed image
    # db_rec = copy.deepcopy(db[idx])
    # preprocess_image(img_path=db_rec['image'], with_vis=True)

    # show db by idx
    if vis_item:
        db_rec = copy.deepcopy(db[idx])
        show_db_item(db_rec=db_rec, with_preprocess=with_preprocess)
        print(f"{idx}th rec has been showed!")

    # show group db by idx
    if vis_single_group:
        db_group_rec = _get_group_item(db=db, idx=idx)
        show_db_single_group(db_group_rec=db_group_rec, with_preprocess=with_preprocess)
        print(f"{idx}th group recs has been showed!")

    # show several groups db rec
    if vis_groups:
        num_groups = 7
        group_intervals = 500
        show_db_groups(db=db, num_groups=num_groups, group_intervals=group_intervals, idx=idx, with_preprocess=with_preprocess)
        print(f"{idx}th~{idx+num_groups}th group recs have been showed!")

    print(f"this is the end!")


if __name__ == '__main__':
    main()
