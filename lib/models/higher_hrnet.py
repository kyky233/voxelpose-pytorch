from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn
# from models.modules import BasicBlock, Bottleneck

import logging

logger = logging.getLogger(__name__)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, bn_momentum=0.1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=bn_momentum)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=bn_momentum)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, bn_momentum=0.1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=bn_momentum)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class StageModule(nn.Module):
    def __init__(self, stage, output_branches, c, bn_momentum):
        super(StageModule, self).__init__()
        self.stage = stage
        self.output_branches = output_branches

        self.branches = nn.ModuleList()
        for i in range(self.stage):
            w = c * (2 ** i)
            branch = nn.Sequential(
                BasicBlock(w, w, bn_momentum=bn_momentum),
                BasicBlock(w, w, bn_momentum=bn_momentum),
                BasicBlock(w, w, bn_momentum=bn_momentum),
                BasicBlock(w, w, bn_momentum=bn_momentum),
            )
            self.branches.append(branch)

        self.fuse_layers = nn.ModuleList()
        # for each output_branches (i.e. each branch in all cases but the very last one)
        for i in range(self.output_branches):
            self.fuse_layers.append(nn.ModuleList())
            for j in range(self.stage):  # for each branch
                if i == j:
                    self.fuse_layers[-1].append(nn.Sequential())  # Used in place of "None" because it is callable
                elif i < j:
                    self.fuse_layers[-1].append(nn.Sequential(
                        nn.Conv2d(c * (2 ** j), c * (2 ** i), kernel_size=(1, 1), stride=(1, 1), bias=False),
                        nn.BatchNorm2d(c * (2 ** i), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.Upsample(scale_factor=(2.0 ** (j - i)), mode='nearest'),
                    ))
                elif i > j:
                    ops = []
                    for k in range(i - j - 1):
                        ops.append(nn.Sequential(
                            nn.Conv2d(c * (2 ** j), c * (2 ** j), kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
                                      bias=False),
                            nn.BatchNorm2d(c * (2 ** j), eps=1e-05, momentum=0.1, affine=True,
                                           track_running_stats=True),
                            nn.ReLU(inplace=True),
                        ))
                    ops.append(nn.Sequential(
                        nn.Conv2d(c * (2 ** j), c * (2 ** i), kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
                                  bias=False),
                        nn.BatchNorm2d(c * (2 ** i), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    ))
                    self.fuse_layers[-1].append(nn.Sequential(*ops))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        assert len(self.branches) == len(x)

        x = [branch(b) for branch, b in zip(self.branches, x)]

        x_fused = []
        for i in range(len(self.fuse_layers)):
            for j in range(0, len(self.branches)):
                if j == 0:
                    x_fused.append(self.fuse_layers[i][0](x[0]))
                else:
                    x_fused[i] = x_fused[i] + self.fuse_layers[i][j](x[j])

        for i in range(len(x_fused)):
            x_fused[i] = self.relu(x_fused[i])

        return x_fused


class HigherHRNet(nn.Module):
    def __init__(self, c=48, nof_joints=17, bn_momentum=0.1):
        super(HigherHRNet, self).__init__()

        # init
        self.nof_joints = nof_joints

        # Input (stem net)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(64, eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)

        # Stage 1 (layer1) - First group of bottleneck (resnet) modules
        downsample = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True),
        )
        self.layer1 = nn.Sequential(
            Bottleneck(64, 64, downsample=downsample),
            Bottleneck(256, 64),
            Bottleneck(256, 64),
            Bottleneck(256, 64),
        )

        # Fusion layer 1 (transition1) - Creation of the first two branches (one full and one half resolution)
        self.transition1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, c, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                nn.BatchNorm2d(c, eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(nn.Sequential(  # Double Sequential to fit with official pretrained weights
                nn.Conv2d(256, c * (2 ** 1), kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
                nn.BatchNorm2d(c * (2 ** 1), eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            )),
        ])

        # Stage 2 (stage2) - Second module with 1 group of bottleneck (resnet) modules. This has 2 branches
        self.stage2 = nn.Sequential(
            StageModule(stage=2, output_branches=2, c=c, bn_momentum=bn_momentum),
        )

        # Fusion layer 2 (transition2) - Creation of the third branch (1/4 resolution)
        self.transition2 = nn.ModuleList([
            nn.Sequential(),  # None,   - Used in place of "None" because it is callable
            nn.Sequential(),  # None,   - Used in place of "None" because it is callable
            nn.Sequential(nn.Sequential(  # Double Sequential to fit with official pretrained weights
                nn.Conv2d(c * (2 ** 1), c * (2 ** 2), kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
                nn.BatchNorm2d(c * (2 ** 2), eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            )),  # ToDo Why the new branch derives from the "upper" branch only?
        ])

        # Stage 3 (stage3) - Third module with 4 groups of bottleneck (resnet) modules. This has 3 branches
        self.stage3 = nn.Sequential(
            StageModule(stage=3, output_branches=3, c=c, bn_momentum=bn_momentum),
            StageModule(stage=3, output_branches=3, c=c, bn_momentum=bn_momentum),
            StageModule(stage=3, output_branches=3, c=c, bn_momentum=bn_momentum),
            StageModule(stage=3, output_branches=3, c=c, bn_momentum=bn_momentum),
        )

        # Fusion layer 3 (transition3) - Creation of the fourth branch (1/8 resolution)
        self.transition3 = nn.ModuleList([
            nn.Sequential(),  # None,   - Used in place of "None" because it is callable
            nn.Sequential(),  # None,   - Used in place of "None" because it is callable
            nn.Sequential(),  # None,   - Used in place of "None" because it is callable
            nn.Sequential(nn.Sequential(  # Double Sequential to fit with official pretrained weights
                nn.Conv2d(c * (2 ** 2), c * (2 ** 3), kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
                nn.BatchNorm2d(c * (2 ** 3), eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            )),  # ToDo Why the new branch derives from the "upper" branch only?
        ])

        # Stage 4 (stage4) - Fourth module with 3 groups of bottleneck (resnet) modules. This has 4 branches
        self.stage4 = nn.Sequential(
            StageModule(stage=4, output_branches=4, c=c, bn_momentum=bn_momentum),
            StageModule(stage=4, output_branches=4, c=c, bn_momentum=bn_momentum),
            StageModule(stage=4, output_branches=1, c=c, bn_momentum=bn_momentum),
        )

        # New HigherHRNet section

        # Final blocks
        self.num_deconvs = 1
        self.final_layers = []

        # "We only predict tagmaps at the lowest resolution, instead of using all resolutions"
        # At the lower resolution, both heatmaps and tagmaps are predicted for every joint
        #   -> output channels are nof_joints * 2
        self.final_layers.append(nn.Conv2d(c, nof_joints * 2, kernel_size=(1, 1), stride=(1, 1)))
        for i in range(self.num_deconvs):
            self.final_layers.append(nn.Conv2d(c, nof_joints, kernel_size=(1, 1), stride=(1, 1)))

        self.final_layers = nn.ModuleList(self.final_layers)

        # Deconv layers
        self.deconv_layers = []
        input_channels = c
        for i in range(self.num_deconvs):
            if True:
                # See comment above about "nof_joints * 2" at lower resolution
                if i == 0:
                    input_channels += nof_joints * 2
                else:
                    input_channels += nof_joints
            output_channels = c

            deconv_kernel, padding, output_padding = 4, 1, 0

            layers = []
            layers.append(nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size=deconv_kernel, stride=2,
                                   padding=padding, output_padding=output_padding, bias=False),
                nn.BatchNorm2d(output_channels, momentum=bn_momentum),
                nn.ReLU(inplace=True)
            ))
            for _ in range(4):
                layers.append(nn.Sequential(
                    BasicBlock(output_channels, output_channels),
                ))
            self.deconv_layers.append(nn.Sequential(*layers))
            input_channels = output_channels

        self.deconv_layers = nn.ModuleList(self.deconv_layers)

    def forward(self, x):   # x---torch.Size([8, 3, 960, 512])
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = [trans(x) for trans in self.transition1]  # Since now, x is a list (# == nof branches)

        x = self.stage2(x)
        # x = [trans(x[-1]) for trans in self.transition2]  # New branch derives from the "upper" branch only
        x = [
            self.transition2[0](x[0]),
            self.transition2[1](x[1]),
            self.transition2[2](x[-1])
        ]  # New branch derives from the "upper" branch only

        x = self.stage3(x)
        # x = [trans(x) for trans in self.transition3]  # New branch derives from the "upper" branch only
        x = [
            self.transition3[0](x[0]),
            self.transition3[1](x[1]),
            self.transition3[2](x[2]),
            self.transition3[3](x[-1])
        ]  # New branch derives from the "upper" branch only

        x = self.stage4(x)

        final_outputs = []
        x = x[0]
        y = self.final_layers[0](x)
        final_outputs.append(y)

        for i in range(self.num_deconvs):
            if True:
                x = torch.cat((x, y), 1)

            x = self.deconv_layers[i](x)
            y = self.final_layers[i + 1](x)
            final_outputs.append(y)

        # recollect outputs to obtain heatmaps
        outputs, heatmaps, tags = self._get_multi_stage_outputs(outputs=final_outputs, project2image=True)

        # aggregate the multiple heatmaps and tags
        heatmaps_list = None
        tags_list = []
        heatmaps_list, tags_list = self._aggregate_results(
            heatmaps_list, tags_list, heatmaps, tags, with_flip=False, project2image=True
        )

        heatmaps = heatmaps_list.float()

        return heatmaps

    # derived from https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation
    def _get_multi_stage_outputs(self, outputs,
                                with_flip=False, project2image=False, size_projected=None,
                                nof_joints=17):
        heatmaps_avg = 0
        num_heatmaps = 0
        heatmaps = []
        tags = []

        # inference
        # outputs is a list with (default) shape
        #   [(batch, nof_joints*2, height//4, width//4), (batch, nof_joints, height//2, width//2)]
        # but it could also be (no checkpoints with this configuration)
        #   [(batch, nof_joints*2, height//4, width//4), (batch, nof_joints*2, height//2, width//2), (batch, nof_joints, height, width)]

        # get higher output resolution
        higher_resolution = (outputs[-1].shape[-2], outputs[-1].shape[-1])

        for i, output in enumerate(outputs):
            if i != len(outputs) - 1:
                output = torch.nn.functional.interpolate(
                    output,
                    size=higher_resolution,
                    mode='bilinear',
                    align_corners=False
                )

            heatmaps_avg += output[:, :nof_joints]
            num_heatmaps += 1

            if output.shape[1] > nof_joints:
                tags.append(output[:, nof_joints:])

        if num_heatmaps > 0:
            heatmaps.append(heatmaps_avg / num_heatmaps)

        if with_flip:  # ToDo
            raise NotImplementedError

        if project2image and size_projected:
            heatmaps = [
                torch.nn.functional.interpolate(
                    hms,
                    size=(size_projected[1], size_projected[0]),
                    mode='bilinear',
                    align_corners=False
                )
                for hms in heatmaps
            ]

            tags = [
                torch.nn.functional.interpolate(
                    tms,
                    size=(size_projected[1], size_projected[0]),
                    mode='bilinear',
                    align_corners=False
                )
                for tms in tags
            ]

        return outputs, heatmaps, tags

    def _aggregate_results(self, final_heatmaps, tags_list, heatmaps, tags, with_flip=False,
                          project2image=False):
        # if scale_factor == 1:
        if final_heatmaps is not None and not project2image:
            tags = [
                torch.nn.functional.interpolate(
                    tms,
                    size=(final_heatmaps.size(2), final_heatmaps.size(3)),
                    mode='bilinear',
                    align_corners=False
                )
                for tms in tags
            ]
        for tms in tags:
            tags_list.append(torch.unsqueeze(tms, dim=4))

        heatmaps_avg = (heatmaps[0] + heatmaps[1]) / 2.0 if with_flip else heatmaps[0]

        if final_heatmaps is None:
            final_heatmaps = heatmaps_avg
        elif project2image:
            final_heatmaps += heatmaps_avg
        else:
            final_heatmaps += torch.nn.functional.interpolate(
                heatmaps_avg,
                size=(final_heatmaps.size(2), final_heatmaps.size(3)),
                mode='bilinear',
                align_corners=False
            )

        return final_heatmaps, tags_list


def get_pose_net(cfg, is_train, **kwargs):
    model = HigherHRNet(c=32, nof_joints=17, bn_momentum=0.1)

    # load weight

    model.load_state_dict(
        torch.load(cfg.NETWORK.PRETRAINED_BACKBONE)
    )

    return model


if __name__ == '__main__':
    model = HigherHRNet(32, 17, 0.1)

    # print(model)

    model.load_state_dict(
        torch.load('./weights/pose_higher_hrnet_w32_512.pth')
    )
    # from collections import OrderedDict
    # weights_ = torch.load('./weights/pose_higher_hrnet_w32_640.pth')
    # weights = OrderedDict([(k[2:], v) for k, v in weights_.items()])
    # res = model.load_state_dict(weights)
    print('ok!!')

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    print(device)

    model = model.to(device)

    inp = torch.ones(2, 3, 384, 288).to(device)

    import cv2
    img = cv2.imread('./sample.jpg', cv2.IMREAD_ANYCOLOR)
    img = cv2.resize(img, (512, 512))
    cv2.imshow('', img)
    cv2.waitKey(1000)
    inp = torch.tensor(img.transpose(2, 0, 1)).unsqueeze(0).cuda().float()
    with torch.no_grad():
        ys = model(inp)
    for y in ys:
        print(y.shape)
        print(torch.min(y).item(), torch.mean(y).item(), torch.max(y).item())
