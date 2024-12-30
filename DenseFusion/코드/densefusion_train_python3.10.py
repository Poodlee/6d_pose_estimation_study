import argparse
import logging
import os
import random
import sys
import time
import math
from pathlib import Path

import numpy as np
import numpy.ma as ma
import scipy.io as scio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.autograd import Function
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


########################################
# Logger
########################################
def setup_logger(logger_name, log_file, level=logging.INFO):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)  
    logger = logging.getLogger(logger_name)

    if not logger.hasHandlers():  # 기존 핸들러가 없는 경우에만 추가
        formatter = logging.Formatter('%(asctime)s : %(message)s')

        fileHandler = logging.FileHandler(log_file, mode='w')
        fileHandler.setFormatter(formatter)

        streamHandler = logging.StreamHandler()
        streamHandler.setFormatter(formatter)

        logger.setLevel(level)
        logger.addHandler(fileHandler)
        logger.addHandler(streamHandler)
    
    return logger

########################################
# KNearestNeighbor Class
########################################
class KNearestNeighbor(Function):
    """
    Compute k nearest neighbors for each query point.
    Returns: top_ind (B, M, K) - 1-based index of the KNN.
    After calling this function, 
    the code will do something like: inds = knn(...) - 1 to get 0-based indexing.
    """

    @staticmethod
    def forward(ctx, ref, query=None):
        """
        ref: B,D,N
        query: B,D,M or None (if None, query = ref)
        Returns:
            top_ind: B,M,K (1-based indices of nearest neighbors)
        """
        k = 1 # 1개만 찾기
        if query is None:
            query = ref

        # dist: B,M,N  (M=#query_points, N=#ref_points)
        dist = torch.cdist(query.transpose(1, 2), ref.transpose(1, 2))
        _, top_ind = torch.topk(dist, k, dim=2, largest=False)
        top_ind = top_ind + 1
        return top_ind

    @staticmethod
    def backward(ctx, grad_output):
        return None, None

########################################
# Utility Functions
########################################
def get_bbox(label: np.ndarray, img_width: int = 480, img_length: int = 640):
    border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360,
                   400, 440, 480, 520, 560, 600, 640, 680]
    rows, cols = np.any(label, axis=1), np.any(label, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    rmax += 1
    cmax += 1

    r_b, c_b = rmax - rmin, cmax - cmin
    for tt in range(len(border_list) - 1):
        if border_list[tt] < r_b < border_list[tt + 1]:
            r_b = border_list[tt + 1]
            break
    for tt in range(len(border_list) - 1):
        if border_list[tt] < c_b < border_list[tt + 1]:
            c_b = border_list[tt + 1]
            break

    center = [(rmin + rmax) // 2, (cmin + cmax) // 2]
    rmin, rmax = center[0] - r_b // 2, center[0] + r_b // 2
    cmin, cmax = center[1] - c_b // 2, center[1] + c_b // 2

    if rmin < 0:
        delta = -rmin
        rmin, rmax = 0, rmax + delta
    if cmin < 0:
        delta = -cmin
        cmin, cmax = 0, cmax + delta
    if rmax > img_width:
        delta = rmax - img_width
        rmin, rmax = max(0, rmin - delta), img_width
    if cmax > img_length:
        delta = cmax - img_length
        cmin, cmax = max(0, cmin - delta), img_length

    return rmin, rmax, cmin, cmax

########################################
# PoseDataset
########################################
class PoseDataset(Dataset):
    def __init__(self, mode: str, num_pt: int, add_noise: bool, root: str, noise_trans: float, refine: bool):
        self.mode = mode
        self.num_pt = num_pt
        self.add_noise = add_noise
        self.noise_trans = noise_trans
        self.refine = refine
        self.root = Path(root)

        data_list_path = Path("data/ycb/dataset_config/") / f"{mode}_data_list.txt"
        self.list = data_list_path.read_text().splitlines()
        self.real = [line for line in self.list if line.startswith("data/")]
        self.syn = [line for line in self.list if line.startswith("data_syn/")]

        class_file_path = Path("data/ycb/dataset_config/classes.txt")
        self.cld = self._load_classes(class_file_path)

        self.cam_params = {
            "cam_1": {"cx": 312.9869, "cy": 241.3109, "fx": 1066.778, "fy": 1067.487},
            "cam_2": {"cx": 323.7872, "cy": 279.6921, "fx": 1077.836, "fy": 1078.189},
        }

        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        self.symmetry_obj_idx = [12, 15, 18, 19, 20]
        self.num_pt_mesh_small = 500
        self.num_pt_mesh_large = 2600
        self.minimum_num_pt = 50

        self.xmap = np.array([[j for i in range(640)] for j in range(480)])
        self.ymap = np.array([[i for i in range(640)] for j in range(480)])
        self.front_num = 2

    def _load_classes(self, class_file_path: Path):
        cld = {}
        with class_file_path.open() as class_file:
            for class_id, class_name in enumerate(class_file, start=1):
                model_dir = self.root / f"models/{class_name.strip()}/points.xyz"
                points = np.loadtxt(model_dir)
                cld[class_id] = points
        return cld

    def __getitem__(self, index):
        img_path = self.root / f"{self.list[index]}-color.jpg"
        depth_path = self.root / f"{self.list[index]}-depth.png"
        label_path = self.root / f"{self.list[index]}-label.png"
        meta_path = self.root / f"{self.list[index]}-meta.mat"

        img = Image.open(img_path).convert("RGB")
        depth = np.array(Image.open(depth_path))
        label = np.array(Image.open(label_path))
        meta = scio.loadmat(meta_path)

        # Camera parameters
        if self.list[index][:8] != "data_syn" and int(self.list[index][5:9]) >= 60:
            cam_params = self.cam_params["cam_2"]
        else:
            cam_params = self.cam_params["cam_1"]
        cam_cx, cam_cy, cam_fx, cam_fy = cam_params.values()

        mask_back = ma.getmaskarray(ma.masked_equal(label, 0))

        add_front = False
        if self.add_noise:
            for _ in range(5):
                seed = random.choice(self.syn)
                front_img = Image.open(f"{self.root}/{seed}-color.jpg").convert("RGB")
                front = np.array(self.trancolor(front_img))
                front = np.transpose(front, (2, 0, 1))

                f_label = np.array(Image.open(f"{self.root}/{seed}-label.png"))
                front_label = np.unique(f_label).tolist()[1:]
                if len(front_label) < self.front_num:
                    continue
                front_label = random.sample(front_label, self.front_num)
                for f_i in front_label:
                    mk = ma.getmaskarray(ma.masked_not_equal(f_label, f_i))
                    if f_i == front_label[0]:
                        mask_front = mk
                    else:
                        mask_front = mask_front * mk
                t_label = label * mask_front
                if len(t_label.nonzero()[0]) > 1000:
                    label = t_label
                    add_front = True
                    break

        obj = meta['cls_indexes'].flatten().astype(np.int32)

        # Select object with sufficient points
        while True:
            idx = np.random.randint(0, len(obj))
            mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
            mask_label = ma.getmaskarray(ma.masked_equal(label, obj[idx]))
            mask = mask_label * mask_depth
            if len(mask.nonzero()[0]) > self.minimum_num_pt:
                break

        if self.add_noise:
            img = self.trancolor(img)

        rmin, rmax, cmin, cmax = get_bbox(mask_label)
        img = np.transpose(np.array(img)[:, :, :3], (2, 0, 1))[:, rmin:rmax, cmin:cmax]

        if self.list[index][:8] == "data_syn":
            seed = random.choice(self.real)
            back_img = Image.open(f"{self.root}/{seed}-color.jpg").convert("RGB")
            back = np.array(self.trancolor(back_img))
            back = np.transpose(back, (2, 0, 1))[:, rmin:rmax, cmin:cmax]
            img_masked = back * mask_back[rmin:rmax, cmin:cmax] + img
        else:
            img_masked = img

        if self.add_noise and add_front:
            img_masked = (img_masked * mask_front[rmin:rmax, cmin:cmax] +
                          front[:, rmin:rmax, cmin:cmax] * ~(mask_front[rmin:rmax, cmin:cmax]))

        if self.list[index][:8] == "data_syn":
            img_masked = img_masked.astype(np.float32)
            img_masked += np.random.normal(loc=0.0, scale=7.0, size=img_masked.shape)

        choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
        if len(choose) > self.num_pt:
            c_mask = np.zeros(len(choose), dtype=int)
            c_mask[:self.num_pt] = 1
            np.random.shuffle(c_mask)
            choose = choose[c_mask.nonzero()]
        else:
            choose = np.pad(choose, (0, self.num_pt - len(choose)), 'wrap')
        

        depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, None].astype(np.float32)
        xmap_masked = self.xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, None].astype(np.float32)
        ymap_masked = self.ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, None].astype(np.float32)
        choose = np.array([choose])

        cam_scale = meta['factor_depth'][0][0]
        pt2 = depth_masked / cam_scale
        pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
        pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
        cloud = np.stack((pt0, pt1, pt2), axis=-1)

        if self.add_noise:
            add_t = np.random.uniform(-self.noise_trans, self.noise_trans, 3)
            cloud += add_t
        else:
            add_t = np.zeros(3)

        if self.refine:
            dellist = random.sample(range(len(self.cld[obj[idx]])),
                                    len(self.cld[obj[idx]]) - self.num_pt_mesh_large)
        else:
            dellist = random.sample(range(len(self.cld[obj[idx]])),
                                    len(self.cld[obj[idx]]) - self.num_pt_mesh_small)
        model_points = np.delete(self.cld[obj[idx]], dellist, axis=0)

        target_r = meta['poses'][:, :, idx][:, :3]
        target_t = meta['poses'][:, :, idx][:, 3:4].flatten()
        target = np.dot(model_points, target_r.T)
        if self.add_noise:
            target += (target_t + add_t)
        else:
            target += target_t

        return (torch.tensor(cloud.astype(np.float32)).to(device),
                torch.tensor(choose.astype(np.int32), dtype=torch.long).to(device),
                self.norm(torch.tensor(img_masked.astype(np.float32))).to(device),
                torch.tensor(target.astype(np.float32)).to(device),
                torch.tensor(model_points.astype(np.float32)).to(device),
                torch.tensor([obj[idx] - 1], dtype=torch.long).to(device))

    def __len__(self):
        return len(self.list)

    def get_sym_list(self):
        return self.symmetry_obj_idx

    def get_num_points_mesh(self):
        return self.num_pt_mesh_large if self.refine else self.num_pt_mesh_small

########################################
# ModifiedResnet: PSPNet 인스턴스 직접 생성
########################################
def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, dilation=dilation, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, stride=1, dilation=dilation)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, dilation=dilation,
                               padding=dilation, bias=False)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.relu(out)

        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers=(3, 4, 23, 3)):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False)
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x_3 = self.layer3(x)
        x = self.layer4(x_3)

        return x, x_3


def resnet18(pretrained=False):
    model = ResNet(BasicBlock, [2, 2, 2, 2])
    return model

def resnet34(pretrained=False):
    model = ResNet(BasicBlock, [3, 4, 6, 3])
    return model

def resnet50(pretrained=False):
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    return model

def resnet101(pretrained=False):
    model = ResNet(Bottleneck, [3, 4, 23, 3])
    return model

def resnet152(pretrained=False):
    model = ResNet(Bottleneck, [3, 8, 36, 3])
    return model


# PSPModule: Pyramid Scene Parsing Module
class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super(PSPModule, self).__init__()
        # Create a stage for each pooling size
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        # Bottleneck layer to combine all features (4 * stages + original)
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))  # Adaptive average pooling (channel 두고 w,h를 줄읾)
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)  # 1x1 convolution
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)  # Get the height and width of the input
        # Apply each stage and upsample to the original size
        priors = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
        # Concatenate original features and pyramid features
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)  # Apply ReLU activation
    
# PSPUpsample: Upsampling Module for PSPNet
class PSPUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PSPUpsample, self).__init__()
        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # Upsampling by a factor of 2
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),  # 3x3 convolution
            nn.PReLU()  # Parametric ReLU activation
        )

    def forward(self, x):
        return self.conv(x)


# PSPNet: Pyramid Scene Parsing Network
class PSPNet(nn.Module):
    """
    Implements the Pyramid Scene Parsing Network (PSPNet).
    Combines a feature extractor, a PSPModule, and an upsampling module
    to perform pixel-wise classification(segmentation).
    """
    def __init__(self, n_classes=21, sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024,
                    backend='resnet18', pretrained=False):
        """
        Args:
            n_classes: Number of output classes.
            sizes: Pooling sizes for the PSPModule.
            psp_size: Number of channels in the PSPModule's input.
            deep_features_size: Number of channels for the deep feature classifier.
            backend: Backbone model (e.g., 'resnet18', 'resnet50').
            pretrained: Whether to use pretrained weights for the backbone.
        """
        super(PSPNet, self).__init__()
        # Load the feature extractor backend (e.g., resnet18) from extractors
        self.feats = globals()[backend](pretrained)
        # PSP module for multi-scale feature aggregation
        self.psp = PSPModule(psp_size, 1024, sizes)
        self.drop_1 = nn.Dropout2d(p=0.3)  # Dropout after the PSP module

        # Upsampling modules for progressively refining the feature map
        self.up_1 = PSPUpsample(1024, 256)
        self.up_2 = PSPUpsample(256, 64)
        self.up_3 = PSPUpsample(64, 64)

        self.drop_2 = nn.Dropout2d(p=0.15)  # Dropout during upsampling
        # Final convolution layer to output class probabilities
        self.final = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1),  # Reduce to 32 channels
            nn.LogSoftmax(dim=1)  # Logarithm of softmax for multi-class probabilities
        )

        # Optional classifier for deep features (not used in the main segmentation pipeline)
        self.classifier = nn.Sequential(
            nn.Linear(deep_features_size, 256),  # Fully connected layer
            nn.ReLU(),  # Activation
            nn.Linear(256, n_classes)  # Output layer for classification
        )

    def forward(self, x):
        # Extract features using the backbone
        f, class_f = self.feats(x) 
        # Apply the PSP module
        p = self.psp(f)
        p = self.drop_1(p)  # Apply dropout

        # Upsample and refine the feature map
        p = self.up_1(p)
        p = self.drop_2(p)
        p = self.up_2(p)
        p = self.drop_2(p)
        p = self.up_3(p)

        # Compute final pixel-wise class probabilities
        return self.final(p)

class ModifiedResnet(nn.Module):
    def __init__(self):
        super().__init__()
        # PSPNet을 사용하는 부분. PSPNet이 사용 가능한 상태라고 가정.
        self.model = PSPNet(sizes=(1, 2, 3, 6), psp_size=512,
                            deep_features_size=256, backend='resnet18')
        self.model = self.model.to(device)

    def forward(self, x):
        return self.model(x)

########################################
# PoseNetFeat
########################################
class PoseNetFeat(nn.Module):
    def __init__(self, num_points):
        super().__init__()
        self.num_points = num_points

        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.e_conv1 = nn.Conv1d(32, 64, 1)
        self.e_conv2 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(256, 512, 1)
        self.conv6 = nn.Conv1d(512, 1024, 1)
        self.ap1 = nn.AvgPool1d(num_points)

    def forward(self, x, emb):
        x = F.relu(self.conv1(x))
        emb = F.relu(self.e_conv1(emb))
        pointfeat_1 = torch.cat((x, emb), dim=1)

        x = F.relu(self.conv2(x))
        emb = F.relu(self.e_conv2(emb))
        pointfeat_2 = torch.cat((x, emb), dim=1)

        x = F.relu(self.conv5(pointfeat_2))
        x = F.relu(self.conv6(x))

        ap_x = self.ap1(x).view(-1, 1024, 1).repeat(1, 1, self.num_points)
        return torch.cat([pointfeat_1, pointfeat_2, ap_x], 1)

########################################
# PoseNet
########################################
class PoseNet(nn.Module):
    def __init__(self, num_points, num_obj):
        super().__init__()
        self.num_points = num_points
        self.num_obj = num_obj

        self.cnn = ModifiedResnet()
        self.feat = PoseNetFeat(num_points)

        self.conv1_r = nn.Conv1d(1408, 640, 1)
        self.conv1_t = nn.Conv1d(1408, 640, 1)
        self.conv1_c = nn.Conv1d(1408, 640, 1)

        self.conv2_r = nn.Conv1d(640, 256, 1)
        self.conv2_t = nn.Conv1d(640, 256, 1)
        self.conv2_c = nn.Conv1d(640, 256, 1)

        self.conv3_r = nn.Conv1d(256, 128, 1)
        self.conv3_t = nn.Conv1d(256, 128, 1)
        self.conv3_c = nn.Conv1d(256, 128, 1)

        self.conv4_r = nn.Conv1d(128, num_obj * 4, 1)
        self.conv4_t = nn.Conv1d(128, num_obj * 3, 1)
        self.conv4_c = nn.Conv1d(128, num_obj * 1, 1)

    def forward(self, img, x, choose, obj):
        out_img = self.cnn(img)
        bs, di, _, _ = out_img.size()

        emb = out_img.view(bs, di, -1)
        choose = choose.repeat(1, di, 1)
        emb = torch.gather(emb, 2, choose).contiguous()

        x = x.squeeze(2).transpose(2, 1).contiguous()
        ap_x = self.feat(x, emb)

        rx = F.relu(self.conv1_r(ap_x))
        tx = F.relu(self.conv1_t(ap_x))
        cx = F.relu(self.conv1_c(ap_x))

        rx = F.relu(self.conv2_r(rx))
        tx = F.relu(self.conv2_t(tx))
        cx = F.relu(self.conv2_c(cx))

        rx = F.relu(self.conv3_r(rx))
        tx = F.relu(self.conv3_t(tx))
        cx = F.relu(self.conv3_c(cx))

        rx = self.conv4_r(rx).view(bs, self.num_obj, 4, self.num_points)
        tx = self.conv4_t(tx).view(bs, self.num_obj, 3, self.num_points)
        cx = torch.sigmoid(self.conv4_c(cx)).view(bs, self.num_obj, 1, self.num_points)

        b = 0
        out_rx = torch.index_select(rx[b], 0, obj[b])
        out_tx = torch.index_select(tx[b], 0, obj[b])
        out_cx = torch.index_select(cx[b], 0, obj[b])

        out_rx = out_rx.transpose(2, 1).contiguous()
        out_cx = out_cx.transpose(2, 1).contiguous()
        out_tx = out_tx.transpose(2, 1).contiguous()

        return out_rx, out_tx, out_cx, emb.detach()

########################################
# PoseRefineNet
########################################
class PoseRefineNetFeat(nn.Module):
    def __init__(self, num_points):
        super().__init__()
        self.num_points = num_points

        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.e_conv1 = nn.Conv1d(32, 64, 1)
        self.e_conv2 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(384, 512, 1)
        self.conv6 = nn.Conv1d(512, 1024, 1)
        self.ap1 = nn.AvgPool1d(num_points)

    def forward(self, x, emb):
        x = F.relu(self.conv1(x))
        emb = F.relu(self.e_conv1(emb))
        pointfeat_1 = torch.cat([x, emb], dim=1)

        x = F.relu(self.conv2(x))
        emb = F.relu(self.e_conv2(emb))
        pointfeat_2 = torch.cat([x, emb], dim=1)

        pointfeat_3 = torch.cat([pointfeat_1, pointfeat_2], dim=1)
        x = F.relu(self.conv5(pointfeat_3))
        x = F.relu(self.conv6(x))

        ap_x = self.ap1(x).view(-1, 1024)
        return ap_x

class PoseRefineNet(nn.Module):
    def __init__(self, num_points, num_obj):
        super().__init__()
        self.num_points = num_points
        self.num_obj = num_obj
        self.feat = PoseRefineNetFeat(num_points)

        self.conv1_r = nn.Linear(1024, 512)
        self.conv2_r = nn.Linear(512, 128)
        self.conv3_r = nn.Linear(128, num_obj * 4)

        self.conv1_t = nn.Linear(1024, 512)
        self.conv2_t = nn.Linear(512, 128)
        self.conv3_t = nn.Linear(128, num_obj * 3)

    def forward(self, x, emb, obj):
        bs = x.size(0)
        x = x.squeeze(2).transpose(2, 1).contiguous()
        ap_x = self.feat(x, emb)

        rx = F.relu(self.conv1_r(ap_x))
        rx = F.relu(self.conv2_r(rx))
        rx = self.conv3_r(rx).view(bs, self.num_obj, 4)

        tx = F.relu(self.conv1_t(ap_x))
        tx = F.relu(self.conv2_t(tx))
        tx = self.conv3_t(tx).view(bs, self.num_obj, 3)

        return rx, tx

########################################
# Loss Functions
########################################
def loss_calculation(pred_r, pred_t, pred_c, target, model_points, idx,
                     points, w, refine, num_point_mesh, sym_list):
    bs, num_p, _ = pred_c.size()

    pred_r = pred_r / torch.norm(pred_r, dim=2, keepdim=True)

    base = torch.cat([
        (1.0 - 2.0 * (pred_r[..., 2] ** 2 + pred_r[..., 3] ** 2)).unsqueeze(-1),
        (2.0 * pred_r[..., 1] * pred_r[..., 2] - 2.0 * pred_r[..., 0] * pred_r[..., 3]).unsqueeze(-1),
        (2.0 * pred_r[..., 0] * pred_r[..., 2] + 2.0 * pred_r[..., 1] * pred_r[..., 3]).unsqueeze(-1),
        (2.0 * pred_r[..., 1] * pred_r[..., 2] + 2.0 * pred_r[..., 0] * pred_r[..., 3]).unsqueeze(-1),
        (1.0 - 2.0 * (pred_r[..., 1] ** 2 + pred_r[..., 3] ** 2)).unsqueeze(-1),
        (-2.0 * pred_r[..., 0] * pred_r[..., 1] + 2.0 * pred_r[..., 2] * pred_r[..., 3]).unsqueeze(-1),
        (-2.0 * pred_r[..., 0] * pred_r[..., 2] + 2.0 * pred_r[..., 1] * pred_r[..., 3]).unsqueeze(-1),
        (2.0 * pred_r[..., 0] * pred_r[..., 1] + 2.0 * pred_r[..., 2] * pred_r[..., 3]).unsqueeze(-1),
        (1.0 - 2.0 * (pred_r[..., 1] ** 2 + pred_r[..., 2] ** 2)).unsqueeze(-1)
    ], dim=-1).view(bs * num_p, 3, 3)

    ori_base = base
    base = base.contiguous().transpose(2,1).contiguous()

    model_points = model_points.view(bs, 1, num_point_mesh, 3).expand(-1, num_p, -1, -1)
    model_points = model_points.reshape(bs * num_p, num_point_mesh, 3)

    target = target.view(bs, 1, num_point_mesh, 3).expand(-1, num_p, -1, -1)
    target = target.reshape(bs * num_p, num_point_mesh, 3)
    ori_target = target

    pred_t = pred_t.contiguous().view(bs * num_p, 1, 3)
    ori_t = pred_t
    points = points.contiguous().view(bs * num_p, 1, 3)
    pred_c = pred_c.contiguous().view(bs * num_p)

    pred = torch.bmm(model_points, base) + points + pred_t

    if not refine and idx[0].item() in sym_list:
        target = target[0].transpose(1, 0).reshape(3, -1)
        pred = pred.permute(2, 0, 1).reshape(3, -1)
        inds = KNearestNeighbor.apply(target.unsqueeze(0).to(device), pred.unsqueeze(0).to(device))
        target = torch.index_select(target, 1, inds.view(-1) - 1)
        target = target.view(3, bs * num_p, num_point_mesh).permute(1, 2, 0).contiguous()        
        pred = pred.view(3, bs * num_p, num_point_mesh).permute(1, 2, 0).contiguous()

    dis = torch.norm(pred - target, dim=2).mean(dim=1)
    loss = torch.mean(dis * pred_c - w * torch.log(pred_c + 1e-8), dim=0)

    pred_c = pred_c.view(bs, num_p)
    _, which_max = torch.max(pred_c, dim=1)
    dis = dis.view(bs, num_p)
    
    t = ori_t[which_max[0]] + points[which_max[0]]
    points = points.view(1, bs * num_p, 3)

    ori_base = ori_base[which_max[0]].view(1,3,3).contiguous()
    ori_t = t.repeat(bs*num_p,1).contiguous().view(1, bs*num_p,3)
    new_points = torch.bmm(points - ori_t, ori_base).contiguous()

    new_target = ori_target[0].view(1, num_point_mesh, 3).contiguous()
    ori_t = t.repeat(num_point_mesh, 1).contiguous().view(1, num_point_mesh, 3)
    new_target = torch.bmm(new_target - ori_t, ori_base).contiguous()

    return loss, dis[0][which_max[0]], new_points.detach(), new_target.detach()

class Loss(_Loss):
    def __init__(self, num_points_mesh, sym_list):
        super().__init__(reduction='mean')
        self.num_pt_mesh = num_points_mesh
        self.sym_list = sym_list

    def forward(self, pred_r, pred_t, pred_c, target, model_points, idx, points, w, refine):
        return loss_calculation(pred_r, pred_t, pred_c, target, model_points, idx,
                                points, w, refine, self.num_pt_mesh, self.sym_list)

def loss_calculation_refine(pred_r, pred_t, target, model_points, idx, points,
                            num_point_mesh, sym_list):
    pred_r = pred_r / torch.norm(pred_r, dim=2, keepdim=True)

    base = torch.cat([
        (1.0 - 2.0 * (pred_r[..., 2] ** 2 + pred_r[..., 3] ** 2)).unsqueeze(-1),
        (2.0 * pred_r[..., 1] * pred_r[..., 2] - 2.0 * pred_r[..., 0] * pred_r[..., 3]).unsqueeze(-1),
        (2.0 * pred_r[..., 0] * pred_r[..., 2] + 2.0 * pred_r[..., 1] * pred_r[..., 3]).unsqueeze(-1),
        (2.0 * pred_r[..., 1] * pred_r[..., 2] + 2.0 * pred_r[..., 0] * pred_r[..., 3]).unsqueeze(-1),
        (1.0 - 2.0 * (pred_r[..., 1] ** 2 + pred_r[..., 3] ** 2)).unsqueeze(-1),
        (-2.0 * pred_r[..., 0] * pred_r[..., 1] + 2.0 * pred_r[..., 2] * pred_r[..., 3]).unsqueeze(-1),
        (-2.0 * pred_r[..., 0] * pred_r[..., 2] + 2.0 * pred_r[..., 1] * pred_r[..., 3]).unsqueeze(-1),
        (2.0 * pred_r[..., 0] * pred_r[..., 1] + 2.0 * pred_r[..., 2] * pred_r[..., 3]).unsqueeze(-1),
        (1.0 - 2.0 * (pred_r[..., 1] ** 2 + pred_r[..., 2] ** 2)).unsqueeze(-1)
    ], dim=-1).view(-1, 3, 3)

    model_points = model_points.view(1, 1, num_point_mesh, 3).expand(-1, pred_r.size(1), -1, -1)
    model_points = model_points.reshape(-1, num_point_mesh, 3)

    target = target.view(1, 1, num_point_mesh, 3).expand(-1, pred_r.size(1), -1, -1)
    target = target.reshape(-1, num_point_mesh, 3)

    pred_t = pred_t.view(-1, 1, 3)
    pred = torch.bmm(model_points, base.transpose(2, 1)) + pred_t

    if idx[0].item() in sym_list:
        target_ = target[0].permute(1, 0).reshape(3, -1)
        pred_ = pred.permute(2, 0, 1).reshape(3, -1)
        inds = KNearestNeighbor.apply(target.unsqueeze(0).to(device), pred.unsqueeze(0).to(device)) 
        target_ = target_[:, inds].reshape(3, pred_r.size(0), num_point_mesh).permute(1, 2, 0)
        pred_ = pred_.reshape(3, pred_r.size(0), num_point_mesh).permute(1, 2, 0)
        target = target_
        pred = pred_

    dis = torch.mean(torch.norm(pred - target, dim=2), dim=1)
    t = pred_t[0]
    ori_base = base[0].unsqueeze(0)
    points = points.view(1, -1, 3)
    new_points = torch.bmm(points - t.unsqueeze(1), ori_base.transpose(2, 1))
    new_target = torch.bmm(target[0] - t.unsqueeze(1), ori_base.transpose(2, 1))

    return dis, new_points.detach(), new_target.detach()

class Loss_refine(_Loss):
    def __init__(self, num_points_mesh, sym_list):
        super().__init__(reduction='mean')
        self.num_pt_mesh = num_points_mesh
        self.sym_list = sym_list

    def forward(self, pred_r, pred_t, target, model_points, idx, points):
        dis, new_points, new_target = loss_calculation_refine(pred_r, pred_t, target,
                                                              model_points, idx, points,
                                                              self.num_pt_mesh, self.sym_list)
        return torch.mean(dis), new_points, new_target

########################################
# Argument Parsing & Main
########################################
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DenseFusion Training")
    parser.add_argument('--dataset', type=str, default='ycb', help='Dataset: ycb or linemod')
    parser.add_argument('--dataset_root', type=str, default='data/', help='Dataset root')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lr_rate', type=float, default=0.3)
    parser.add_argument('--w', type=float, default=0.015)
    parser.add_argument('--w_rate', type=float, default=0.3)
    parser.add_argument('--decay_margin', type=float, default=0.016)
    parser.add_argument('--refine_margin', type=float, default=0.013)
    parser.add_argument('--noise_trans', type=float, default=0.03)
    parser.add_argument('--iteration', type=int, default=2, help='Number of refinement iterations')
    parser.add_argument('--epoch', type=int, default=500)
    parser.add_argument('--resume_posenet', type=str, default='')
    parser.add_argument('--resume_refinenet', type=str, default='')
    parser.add_argument('--seed', type=int, default=7)
    return parser.parse_args()

def test_phase(arg, estimator, refiner, test_loader, criterion, criterion_refine, refine_start, device, epoch, start_time, start_time_str):
    estimator.eval()
    refiner.eval()
    total_dis = 0.0
    test_count = 0

    with torch.no_grad():
        for data in tqdm(test_loader, desc=f"Testing Epoch {epoch+1}"):
            points, choose, img, target, model_points, idx = [d.to(device) for d in data]
            pred_r, pred_t, pred_c, emb = estimator(img, points, choose, idx)
            _, dis, new_points, new_target = criterion(pred_r, pred_t, pred_c,
                                                       target, model_points, idx,
                                                       points, arg.w, refine_start)

            if refine_start:
                for _ in range(arg.iteration):
                    pred_r, pred_t = refiner(new_points, emb, idx)
                    dis_, new_points, new_target = criterion_refine(pred_r, pred_t,
                                                                    new_target, model_points,
                                                                    idx, new_points)
                    dis = dis_

            total_dis += dis.item()
            test_count += 1

    avg_dis = total_dis / test_count
    elapsed_time = time.time() - start_time
    BLUE = '\033[94m'  # Bright blue
    BOLD = '\033[1m'   # 볼드
    RESET = '\033[0m'  # Reset to default
    print(f'{BLUE}{BOLD}Test time {time.strftime("%Hh %Mm %Ss", time.gmtime(elapsed_time))} '
                f'Epoch {epoch+1} TEST FINISH Avg dis: {avg_dis}{RESET}')
    
    return avg_dis

def main():
    arg = parse_arguments()
    random.seed(arg.seed)
    torch.manual_seed(arg.seed)

    if arg.dataset == 'ycb':
        arg.num_objects = 21
        arg.num_points = 1000
        arg.outf = 'trained_models/ycb'
        arg.log_dir = 'experiments/logs'
        arg.repeat_epoch = 1
    else:
        raise ValueError(f"Unknown dataset: {arg.dataset}")

    # 경과 시간 계산을 위한 시작 시간 설정
    start_time = time.time()
    start_time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())

    estimator = PoseNet(num_points=arg.num_points, num_obj=arg.num_objects).to(device)
    refiner = PoseRefineNet(num_points=arg.num_points, num_obj=arg.num_objects).to(device)

    if arg.resume_posenet and os.path.isfile(arg.resume_posenet):
        estimator.load_state_dict(torch.load(arg.resume_posenet, map_location=device))
    if arg.resume_refinenet and os.path.isfile(arg.resume_refinenet):
        refiner.load_state_dict(torch.load(arg.resume_refinenet, map_location=device))
        arg.refine_start = True
        arg.decay_start = True
        arg.lr *= arg.lr_rate
        arg.w *= arg.w_rate
        arg.batch_size = int(arg.batch_size / arg.iteration)
    else:
        arg.refine_start = False
        arg.decay_start = False
        if os.path.exists(arg.log_dir): 
            for log in os.listdir(arg.log_dir):  
                log_path = os.path.join(arg.log_dir, log)
                if os.path.isfile(log_path): 
                    os.remove(log_path)


    trainset = PoseDataset('train', arg.num_points, True, arg.dataset_root, arg.noise_trans, arg.refine_start)
    testset = PoseDataset('test', arg.num_points, False, arg.dataset_root, 0.0, arg.refine_start)

    train_loader = DataLoader(trainset, batch_size=arg.batch_size, shuffle=True, num_workers=arg.workers, multiprocessing_context='spawn')
    test_loader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=arg.workers, multiprocessing_context='spawn')

    sym_list = trainset.get_sym_list()
    num_points_mesh = trainset.get_num_points_mesh()

    criterion = Loss(num_points_mesh, sym_list)
    criterion_refine = Loss_refine(num_points_mesh, sym_list)

    print(f'>>>>>>>>----------Dataset loaded!---------<<<<<<<<\nlength of the training set: {len(trainset)}\nlength of the testing set: {len(testset)}\nnumber of sample points on mesh: {num_points_mesh}\nsymmetry object list: {sym_list}')

    optimizer = optim.Adam(refiner.parameters() if arg.resume_refinenet else estimator.parameters(), lr=arg.lr)

    best_test = float('inf')

    for epoch in range(arg.epoch):
        logger = setup_logger(f'epoch{epoch+1}', os.path.join(arg.log_dir, f'epoch_{epoch+1}_log.txt'))
        elapsed_time = time.time() - start_time
        logger.info(f'Train time {time.strftime("%Hh %Mm %Ss", time.gmtime(elapsed_time))},Training started')

        estimator.train()
        if arg.refine_start:
            estimator.eval()
            refiner.train()

        for data in tqdm(train_loader, desc=f"Training Epoch {epoch} / {arg.epoch}"):
            points, choose, img, target, model_points, idx = [d.to(device) for d in data]
            pred_r, pred_t, pred_c, emb = estimator(img, points, choose, idx)
            loss, dis, new_points, new_target = criterion(pred_r, pred_t, pred_c,
                                                          target, model_points, idx,
                                                          points, arg.w, arg.refine_start)

            if arg.refine_start:
                for _ in range(arg.iteration):
                    pred_r, pred_t = refiner(new_points, emb, idx)
                    dis_, new_points, new_target = criterion_refine(pred_r, pred_t,
                                                                    new_target, model_points,
                                                                    idx, new_points)
                    dis_.backward()
            else:
                loss.backward()

            optimizer.step()
            optimizer.zero_grad()

        RED = '\033[31m'  # RED
        RESET = '\033[0m'  # RESET

        print(f'>>>>>>>>----------epoch {epoch + 1} {RED}train finish loss: {loss}{RESET}---------<<<<<<<<')

        if (epoch + 1) % 1 == 0:
            avg_dis = test_phase(arg, estimator, refiner, test_loader, criterion, criterion_refine, arg.refine_start, device, epoch, start_time, start_time_str)
            if avg_dis <= best_test:
                best_test = avg_dis
                os.makedirs(arg.outf,exist_ok=True)
                if arg.refine_start:
                    torch.save(refiner.state_dict(), f'{arg.outf}/pose_refine_model_{epoch+1}_{avg_dis}.pth')
                else:
                    torch.save(estimator.state_dict(), f'{arg.outf}/pose_model_{epoch+1}_{avg_dis}')
                print(epoch, '>>>>>>>>----------BEST TEST MODEL SAVED---------<<<<<<<<')

            if best_test < arg.decay_margin and not arg.decay_start:
                arg.decay_start = True
                arg.lr *= arg.lr_rate
                arg.w *= arg.w_rate
                optimizer = optim.Adam(estimator.parameters(), lr=arg.lr)

            if best_test < arg.refine_margin and not arg.refine_start:
                arg.refine_start = True
                arg.batcj_size = int(arg.batch_size / arg.iteration)
                optimizer = optim.Adam(refiner.parameters(), lr=arg.lr)

                trainset = PoseDataset('train', arg.num_points, True, arg.dataset_root, arg.noise_trans, arg.refine_start)
                testset = PoseDataset('test', arg.num_points, False, arg.dataset_root, 0.0, arg.refine_start)

                train_loader = DataLoader(trainset, batch_size=arg.batch_size, shuffle=True, num_workers=arg.workers, multiprocessing_context='spawn')
                test_loader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=arg.workers, multiprocessing_context='spawn')

                print(f'>>>>>>>>----------Dataset(refine) loaded!---------<<<<<<<<\nlength of the training set: {len(trainset)}\nlength of the testing set: {len(testset)}\nnumber of sample points on mesh: {num_points_mesh}\nsymmetry object list: {sym_list}')


                sym_list = trainset.get_sym_list()
                num_points_mesh = trainset.get_num_points_mesh()

                criterion = Loss(num_points_mesh, sym_list)
                criterion_refine = Loss_refine(num_points_mesh, sym_list)

    print(">>>>>>>>----------Training and testing complete.---------<<<<<<<<")

if __name__ == "__main__":
    main()
