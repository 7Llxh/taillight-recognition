# -*- coding: utf-8 -*-
"""从标注 JSON 生成部件检测数据集（YOLO detection 格式）。

复制原图 + 生成部件框 labels（YOLO xywh 归一化），划分 train/val（8:2），
生成 data.yaml。
"""
import glob
import json
import os
import random
import shutil
from collections import Counter

import cv2
import numpy as np

ANN = r"e:/车辆识别/-/data/annotations"
OUT = r"e:/车辆识别/-/data/parts_dataset"
PARTS = ["taillight", "headlight", "mirror", "window", "wheel",
         "plate", "grille", "bumper", "exhaust"]
SEED = 42


def imread_unicode(p):
    return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)


def imwrite_unicode(p, img):
    cv2.imencode(os.path.splitext(p)[1], img)[1].tofile(p)


def main():
    items = []
    for f in glob.glob(os.path.join(ANN, "*.json")):
        with open(f, encoding="utf-8") as fp:
            d = json.load(fp)
        if not d.get("parts"):
            continue
        items.append((d["image"], d["parts"], os.path.splitext(os.path.basename(f))[0]))
    print(f"有部件框的标注: {len(items)}")

    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    for sp in ("train", "val"):
        os.makedirs(os.path.join(OUT, "images", sp), exist_ok=True)
        os.makedirs(os.path.join(OUT, "labels", sp), exist_ok=True)

    random.seed(SEED)
    random.shuffle(items)
    n = len(items)
    n_train = int(n * 0.8)
    splits = ["train"] * n_train + ["val"] * (n - n_train)

    counts = {"train": 0, "val": 0}
    pc = Counter()
    for (img_path, parts, fname), sp in zip(items, splits):
        img = imread_unicode(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        lines = []
        for p in parts:
            if p["label"] not in PARTS:
                continue
            cls = PARTS.index(p["label"])
            x1, y1, x2, y2 = p["box"]
            xc = ((x1 + x2) / 2) / w
            yc = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            lines.append(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
            pc[p["label"]] += 1
        if not lines:
            continue
        imwrite_unicode(os.path.join(OUT, "images", sp, f"{fname}.jpg"), img)
        with open(os.path.join(OUT, "labels", sp, f"{fname}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        counts[sp] += 1

    with open(os.path.join(OUT, "data.yaml"), "w", encoding="utf-8") as f:
        f.write(f"path: {OUT}\ntrain: images/train\nval: images/val\n"
                f"nc: {len(PARTS)}\nnames: {PARTS}\n")
    print("数据集已生成:", OUT)
    for sp in ("train", "val"):
        print(f"  {sp}: {counts[sp]} 张")
    print("部件框:", dict(pc))


if __name__ == "__main__":
    main()
