# -*- coding: utf-8 -*-
"""生成尾灯特征训练集（ImageFolder，按车型系列）。

遍历 data/raw/{车型}/，对每张图：
  车辆检测 -> 朝向分类(筛 rear) -> 部件检测器裁尾灯 -> 归一化 224
按车型存到 data/taillight_dataset/{train,val,test}/{车型}/。
8:1:1 按车型分层（测试集不参与建库/训练）。

注: cv2 在 Windows 不支持中文路径，用 imdecode/imencode。
"""
import glob
import os
import random
import shutil
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from detect_vehicle import (_get_model, MODEL_PATH, _filter_vehicle_boxes,
                            detect_taillights, crop_and_normalize)
from orientation import detect_orientation

RAW = os.path.join(HERE, "data", "raw")
OUT = os.path.join(HERE, "data", "taillight_dataset")
SEED = 42
SPLIT = (0.8, 0.1, 0.1)
IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def imread_unicode(p):
    return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)


def imwrite_unicode(p, img):
    cv2.imencode(os.path.splitext(p)[1], img)[1].tofile(p)


def extract_taillight_crops(img):
    """对一张图提取所有 rear 车辆的尾灯裁剪（归一化 224）。返回 [crop, ...]。"""
    h, w = img.shape[:2]
    res = _get_model(MODEL_PATH)(img, classes=[2, 5, 7], conf=0.4)[0]
    crops = []
    for box in _filter_vehicle_boxes(res.boxes, h, w):
        x1, y1, x2, y2 = box.xyxy[0].int().tolist()
        vcrop = img[y1:y2, x1:x2]
        ori = detect_orientation(vcrop)
        if ori["view"] != "rear":
            continue  # 只取车尾视图
        for t in detect_taillights(vcrop):
            tx1, ty1, tx2, ty2 = t["box"]
            tc = vcrop[ty1:ty2, tx1:tx2]
            if tc.size == 0:
                continue
            crops.append(crop_and_normalize(tc, 224))
    return crops


def main():
    by_model = {}
    for model_dir in sorted(glob.glob(os.path.join(RAW, "*"))):
        if not os.path.isdir(model_dir):
            continue
        model = os.path.basename(model_dir)
        crops = []
        for p in sorted(glob.glob(os.path.join(model_dir, "*"))):
            if not p.lower().endswith(IMG_EXTS):
                continue
            img = imread_unicode(p)
            if img is None:
                continue
            crops.extend(extract_taillight_crops(img))
        if crops:
            by_model[model] = crops
            print(f"{model}: {len(crops)} 个尾灯裁剪", flush=True)

    if not by_model:
        print("未生成任何尾灯裁剪（检查 data/raw 与朝向/部件检测）")
        return

    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    for sp in ("train", "val", "test"):
        os.makedirs(os.path.join(OUT, sp), exist_ok=True)

    random.seed(SEED)
    total = {"train": 0, "val": 0, "test": 0}
    for model, crops in by_model.items():
        random.shuffle(crops)
        n = len(crops)
        n_tr = int(n * SPLIT[0])
        n_va = int(n * SPLIT[1])
        splits = ["train"] * n_tr + ["val"] * n_va + ["test"] * (n - n_tr - n_va)
        for i, (c, sp) in enumerate(zip(crops, splits)):
            d = os.path.join(OUT, sp, model)
            os.makedirs(d, exist_ok=True)
            imwrite_unicode(os.path.join(d, f"{model}_{i:04d}.jpg"), c)
            total[sp] += 1

    print(f"\n尾灯数据集已生成: {OUT}")
    for sp in ("train", "val", "test"):
        print(f"  {sp}: {total[sp]}")


if __name__ == "__main__":
    main()
