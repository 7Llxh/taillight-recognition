# -*- coding: utf-8 -*-
"""尾灯/整车嵌入网络（F7/F8/F17）。ResNet-50 + ArcFace 度量学习。

Embedder 前向返回 512 维 L2 归一化向量（入库/检索用）。
训练时配合 pytorch_metric_learning.ArcFaceLoss（见 train_embedder.py）。
尾灯/整车同架构、分别训练、权重独立。详见 特征提取网络-技术设计.md。
"""
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

EMBED_DIM = 512
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

_TFM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class Embedder(nn.Module):
    """ResNet-50 backbone（去 avgpool/fc）+ Linear(2048,512) + BatchNorm 嵌入头。

    forward 返回 L2 归一化的 512 维向量。
    """
    def __init__(self, embed_dim=EMBED_DIM, pretrained=True):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet50(weights=weights)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # conv1..layer4
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(2048, embed_dim),
            nn.BatchNorm1d(embed_dim),
        )

    def forward(self, x):
        f = self.backbone(x)
        f = self.pool(f).flatten(1)
        f = self.head(f)
        return F.normalize(f, p=2, dim=1)  # L2 归一化（cosine 检索用）


def load_embedder(weights_path, device="cuda"):
    """加载训练好的 Embedder 权重。"""
    model = Embedder(pretrained=False)
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval().to(device)
    return model


def preprocess(crop):
    """BGR ndarray (H,W,3) uint8 -> ImageNet 归一化 tensor (3,224,224)。"""
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return _TFM(rgb)


@torch.no_grad()
def extract_features(model, crops, device="cuda", batch_size=32):
    """批量提取特征。crops: list[BGR ndarray uint8]。返回 (N,512) ndarray（已 L2 归一化）。"""
    model.eval().to(device)
    feats = []
    for i in range(0, len(crops), batch_size):
        batch = crops[i:i + batch_size]
        imgs = torch.stack([preprocess(c) for c in batch]).to(device)
        f = model(imgs).cpu().numpy()
        feats.append(f)
    return np.vstack(feats) if feats else np.zeros((0, EMBED_DIM), dtype=np.float32)
