import argparse
from torchinfo import summary

import _init_paths
from models.pose_resnet import get_pose_net
import models
from core.config import config
from core.config import update_config


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument(
        '--cfg', help='experiment configure file name', required=True, type=str)

    args, rest = parser.parse_known_args()
    update_config(args.cfg)

    return args


def main():
    # load cfg
    args = parse_args()         # update config
    # set gpu =false

    # load model with mapping to cpu
    backbone = eval('models.' + config.BACKBONE_MODEL + '.get_pose_net')(
        config, is_train=True
    )

    def get_pose_net(cfg, is_train, **kwargs):
        num_layers = cfg.POSE_RESNET.NUM_LAYERS

        block_class, layers = resnet_spec[num_layers]

        model = PoseResNet(block_class, layers, cfg, **kwargs)

        if is_train:
            model.init_weights(cfg.NETWORK.PRETRAINED)

        return model

    batch_size = 16
    # summary(backbone, input_size=(batch_size, 1, 28, 28))
    summary(backbone)

    print(f"this is the end!")


if __name__ == '__main__':
    main()