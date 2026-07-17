# -*- coding: utf-8 -*-
"""训练尾灯嵌入网络（F7）。ResNet-50 + ArcFace 度量学习。

用 data/taillight_dataset（ImageFolder，按车型）训练 512 维嵌入，
使同车型尾灯向量靠近、异车型远离。权重存 runs/embedder/taillight/best.pt。

详见 特征提取网络-技术设计.md。
"""
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pytorch_metric_learning.losses import ArcFaceLoss

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from embedder import Embedder, EMBED_DIM

NAME = sys.argv[1] if len(sys.argv) > 1 else "taillight"  # taillight/vehicle
DATA = os.path.join(HERE, "data", f"{NAME}_dataset")
OUT = os.path.join(HERE, "runs", "embedder", NAME)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 50
BATCH = 32          # 数据少(118)，batch 32 比设计 64 更合适
LR = 1e-4
WD = 1e-4
MARGIN = 0.5
SCALE = 64

TRAIN_TFM = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
EVAL_TFM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def recall_at_k(feat, labels, k=1):
    """Recall@K：查询的 Top-K 近邻中含同类的比例。feat (N,512) 已归一化。"""
    sim = feat @ feat.T  # cosine
    np.fill_diagonal(sim, -1)
    order = np.argsort(-sim, axis=1)[:, :k]
    correct = (labels[order] == labels[:, None]).any(axis=1)
    return float(correct.mean()) if len(labels) else 0.0


def evaluate(embedder, loader):
    embedder.eval()
    feats, labs = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            emb = embedder(imgs.to(DEVICE)).cpu().numpy()
            feats.append(emb); labs.append(labels.numpy())
    if not feats:
        return 0.0, 0.0
    f = np.vstack(feats); l = np.concatenate(labs)
    return recall_at_k(f, l, 1), recall_at_k(f, l, 5)


def main():
    train_ds = datasets.ImageFolder(os.path.join(DATA, "train"), TRAIN_TFM)
    val_ds = datasets.ImageFolder(os.path.join(DATA, "val"), EVAL_TFM)
    num_classes = len(train_ds.classes)
    print(f"[device] {DEVICE}  车型数: {num_classes} {train_ds.classes}")
    print(f"train: {len(train_ds)}  val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0)

    embedder = Embedder(pretrained=True).to(DEVICE)
    arcface = ArcFaceLoss(embedding_size=EMBED_DIM, num_classes=num_classes,
                          margin=MARGIN, scale=SCALE).to(DEVICE)
    optimizer = torch.optim.AdamW(list(embedder.parameters()) + list(arcface.parameters()),
                                  lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    os.makedirs(OUT, exist_ok=True)
    best_r1 = 0.0
    for epoch in range(EPOCHS):
        embedder.train()
        total_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            emb = embedder(imgs)
            loss = arcface(emb, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        r1, r5 = evaluate(embedder, val_loader)
        print(f"epoch {epoch+1:2d}/{EPOCHS}  loss={total_loss/len(train_loader):.4f}  "
              f"val R@1={r1:.3f} R@5={r5:.3f}", flush=True)
        if r1 > best_r1:
            best_r1 = r1
            torch.save(embedder.state_dict(), os.path.join(OUT, "best.pt"))
    torch.save(embedder.state_dict(), os.path.join(OUT, "last.pt"))
    print(f"训练完成。best val R@1={best_r1:.3f}。权重: {OUT}/best.pt")


if __name__ == "__main__":
    main()
