from torchvision import models
import numpy as np
import pickle


def camera_to_world_frame(x, R, T):
    """
    Args
        x: Nx3 points in camera coordinates
        R: 3x3 Camera rotation matrix
        T: 3x1 Camera translation parameters
    Returns
        xcam: Nx3 points in world coordinates
    """

    xcam = R.T.dot(x.T) + T  # rotate and translate
    return xcam.T

def unfold_camera_param(camera):
    R = camera['R']
    T = camera['T']
    fx = camera['fx']
    fy = camera['fy']
    f = np.array([[fx], [fy]]).reshape(-1, 1)
    c = np.array([[camera['cx']], [camera['cy']]]).reshape(-1, 1)
    k = camera['k']
    p = camera['p']
    return R, T, f, c, k, p


def project_point_radial(x, R, T, f, c, k, p):
    """
    Args
        x: Nx3 points in world coordinates
        R: 3x3 Camera rotation matrix
        T: 3x1 Camera translation parameters
        f: (scalar) Camera focal length
        c: 2x1 Camera center
        k: 3x1 Camera radial distortion coefficients
        p: 2x1 Camera tangential distortion coefficients
    Returns
        ypixel.T: Nx2 points in pixel space
    """
    n = x.shape[0]
    xcam = R.dot(x.T - T)
    y = xcam[:2] / (xcam[2]+1e-5)

    r2 = np.sum(y**2, axis=0)
    radial = 1 + np.einsum('ij,ij->j', np.tile(k, (1, n)),
                           np.array([r2, r2**2, r2**3]))
    tan = p[0] * y[1] + p[1] * y[0]
    y = y * np.tile(radial + 2 * tan,
                    (2, 1)) + np.outer(np.array([p[1], p[0]]).reshape(-1), r2)
    ypixel = np.multiply(f, y) + c
    return ypixel.T


def project_pose(x, camera):
    R, T, f, c, k, p = unfold_camera_param(camera)
    return project_point_radial(x, R, T, f, c, k, p)


def projectPoints(X, K, R, t, Kd):
    """
    Projects points X (3xN) using camera intrinsics K (3x3),
    extrinsics (R,t) and distortion parameters Kd=[k1,k2,p1,p2,k3].
    Roughly, x = K*(R*X + t) + distortion
    See http://docs.opencv.org/2.4/doc/tutorials/
    calib3d/camera_calibration/camera_calibration.html
    or cv2.projectPoints
    """
    x = np.dot(R, X) + t

    x[0:2, :] = x[0:2, :] / (x[2, :] + 1e-5)

    r = x[0, :] * x[0, :] + x[1, :] * x[1, :]

    x[0, :] = (
            x[0, :]
            * (1 + Kd[0] * r + Kd[1] * r * r + Kd[4] * r * r * r)
            + 2 * Kd[2] * x[0, :] * x[1, :]
            + Kd[3] * (r + 2 * x[0, :] * x[0, :]))
    x[1, :] = (
            x[1, :]
            * (1 + Kd[0] * r + Kd[1] * r * r + Kd[4] * r * r * r)
            + 2 * Kd[3] * x[0, :] * x[1, :]
            + Kd[2] * (r + 2 * x[1, :] * x[1, :]))

    x[0, :] = K[0, 0] * x[0, :] + K[0, 1] * x[1, :] + K[0, 2]
    x[1, :] = K[1, 0] * x[0, :] + K[1, 1] * x[1, :] + K[1, 2]

    return x


def _get_cam(camera):
    fx, fy = camera['fx'], camera['fy']
    cx, cy = camera['cx'], camera['cy']
    K = np.eye(3)
    K[0, 0] = fx
    K[1, 1] = fy
    K[0, 2] = cx
    K[1, 2] = cy
    camera['K'] = K
    return camera


def load_3d_kpts_sample():
    load_path = '/home/yandanqi/0_data/human36m/annot/h36m_validation.pkl'
    with open(load_path, 'rb') as f:
        annots = pickle.load(f)

    idx = 0
    joints_2d = annots[idx]['joints_2d']
    joints_3d = annots[idx]['joints_3d']
    joints_3d_cam = annots[idx]['joints_3d_camera']
    cam = annots[idx]['camera']

    R, T, f, c, k, p = unfold_camera_param(cam)
    cam = _get_cam(cam)


    print(f"kkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk")


def main():

    load_3d_kpts_sample()

    print(f"kkkkkkkkkkkkkkkkkkkkk")


if __name__ == '__main__':
    main()