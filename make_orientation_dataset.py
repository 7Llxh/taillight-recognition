# -*- coding: utf-8 -*-
"""从标注 JSON 生成朝向分类数据集（ImageFolder）。

读 data/annotations/*.json，按 vehicle_box 裁剪原图车辆区域，
按 view 分到 data/orientation_dataset/{train,val,test}/{front,rear,side}/。
按车型分层 8:1:1 划分。

注: cv2.imread/imwrite 在 Windows 不支持中文路径，用 imdecode/imencode 替代。
"""
import glob
import json
import os
import random
import shutil
from collections import defaultdict

import cv2
import numpy as np

ANN = r"e:/车辆识别/-/data/annotations"
OUT = r"e:/车辆识别/-/data/orientation_dataset"
VIEWS = ["front", "rear", "side"]
SEED = 42


def imread_unicode(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def imwrite_unicode(path, img):
    ext = os.path.splitext(path)[1]
    cv2.imencode(ext, img)[1].tofile(path)


def main():
    by_model = defaultdict(list)
    for f in glob.glob(os.path.join(ANN, "*.json")):
        with open(f, encoding="utf-8") as fp:
            d = json.load(fp)
        vb = d.get("vehicle_box")
        view = d.get("view", "")
        if not vb or view not in VIEWS:
            continue
        by_model[d["model"]].append((d["image"], vb, view,
                                    os.path.splitext(os.path.basename(f))[0]))

    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    for sp in ("train", "val", "test"):
        for v in VIEWS:
            os.makedirs(os.path.join(OUT, sp, v), exist_ok=True)

    random.seed(SEED)
    total = {"train": 0, "val": 0, "test": 0}
    for model, items in by_model.items():
        random.shuffle(items)
        n = len(items)
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)
        splits = (["train"] * n_train) + (["val"] * n_val) + (["test"] * (n - n_train - n_val))
        for (img_path, vb, view, fname), sp in zip(items, splits):
            img = imread_unicode(img_path)
            if img is None:
                continue
            x1, y1, x2, y2 = vb
            crop = img[y1:y2, x1:x2]
            out_name = f"{model}_{fname}.jpg"
            imwrite_unicode(os.path.join(OUT, sp, view, out_name), crop)
            total[sp] += 1

    print("数据集已生成:", OUT)
    for sp in ("train", "val", "test"):
        parts = []
        for v in VIEWS:
            c = len(glob.glob(os.path.join(OUT, sp, v, "*.jpg")))
            parts.append(f"{v}={c}")
        print(f"  {sp} ({total[sp]}): " + ", ".join(parts))


if __name__ == "__main__":
    main()
