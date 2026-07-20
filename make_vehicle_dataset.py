# -*- coding: utf-8 -*-
"""生成整车特征训练集（ImageFolder，按 make_model 车型系列）。

数据源 VMMRdb（读 survey 产的 vmmr_series.json）。**全朝向**（front/side/rear）整车裁剪入库
（整车库为主力，不依赖尾灯定位）。按 make_model 存 data/vehicle_dataset/{train,val,test}/{make_model}/，
文件名带 year：{make_model}__{year}__{idx}.jpg（供 build_library 解析年份）。

用法：
    python make_vehicle_dataset.py                                   # 全量（清单内所有系列）
    python make_vehicle_dataset.py --limit 50 --max-per-series 30    # 验证：图片最多50系列，每系列最多30图
    python make_vehicle_dataset.py --subset                          # 仅 >=SUBSET_MIN_IMGS 图系列
"""
import argparse
import json
import os
import random
import shutil
import sys

import cv2
import numpy as np
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from config import RAW_DIR, SERIES_FILE, SUBSET_MIN_IMGS
from detect_vehicle import _get_model, MODEL_PATH, _filter_vehicle_boxes, crop_and_normalize
from orientation import detect_orientation

OUT = os.path.join(HERE, "data", "vehicle_dataset")
SEED = 42


def imread_unicode(p):
    return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)


def imwrite_unicode(p, img):
    cv2.imencode(os.path.splitext(p)[1], img)[1].tofile(p)


def parse_year(rel_path):
    """从 'make_model_year/file.jpg' 解析 year。非数字（如 1900 占位）返回 None。"""
    dirname = rel_path.replace("\\", "/").split("/")[0]
    _, _, year = dirname.rpartition("_")
    return year if year.isdigit() else None


def extract_vehicle_crops(img):
    """全朝向整车裁剪（整车库主力）。返回 [crop, ...]。"""
    h, w = img.shape[:2]
    res = _get_model(MODEL_PATH)(img, classes=[2, 5, 7], conf=0.4)[0]
    crops = []
    for box in _filter_vehicle_boxes(res.boxes, h, w):
        x1, y1, x2, y2 = box.xyxy[0].int().tolist()
        vcrop = img[y1:y2, x1:x2]
        crops.append(crop_and_normalize(vcrop, 224))  # 全朝向都入（主力）
    return crops


def main():
    ap = argparse.ArgumentParser(description="生成整车特征训练集（VMMRdb）")
    ap.add_argument("--limit", type=int, default=None, help="只取图片最多的 N 系列（验证用）")
    ap.add_argument("--max-per-series", type=int, default=None, help="每系列最多取 N 图（验证用）")
    ap.add_argument("--subset", action="store_true", help=f"仅 >={SUBSET_MIN_IMGS} 图系列")
    args = ap.parse_args()

    with open(SERIES_FILE, encoding="utf-8") as f:
        series = json.load(f)
    items = sorted(series.items(), key=lambda kv: kv[1]["img_count"], reverse=True)
    if args.subset:
        items = [(m, e) for m, e in items if e["img_count"] >= SUBSET_MIN_IMGS]
    if args.limit:
        items = items[:args.limit]
    print(f"系列数: {len(items)}  (limit={args.limit}, max_per_series={args.max_per_series}, subset={args.subset})")

    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    for sp in ("train", "val", "test"):
        os.makedirs(os.path.join(OUT, sp), exist_ok=True)

    random.seed(SEED)
    total = {"train": 0, "val": 0, "test": 0}
    for make_model, e in tqdm(items, desc="系列"):
        paths = e["img_paths"]
        if args.max_per_series:
            random.shuffle(paths)
            paths = paths[:args.max_per_series]
        crops = []  # [(crop, year)]
        for rel in paths:
            p = os.path.join(RAW_DIR, rel)
            img = imread_unicode(p)
            if img is None:
                continue
            year = parse_year(rel)
            for crop in extract_vehicle_crops(img):
                crops.append((crop, year))
        if not crops:
            continue
        random.shuffle(crops)
        n = len(crops)
        n_tr = int(n * 0.8)
        n_va = int(n * 0.1)
        splits = ["train"] * n_tr + ["val"] * n_va + ["test"] * (n - n_tr - n_va)
        for i, ((crop, year), sp) in enumerate(zip(crops, splits)):
            d = os.path.join(OUT, sp, make_model)
            os.makedirs(d, exist_ok=True)
            yr = year if year else "unknown"
            imwrite_unicode(os.path.join(d, f"{make_model}__{yr}__{i:04d}.jpg"), crop)
            total[sp] += 1

    print(f"\n整车数据集: {OUT}")
    for sp in ("train", "val", "test"):
        print(f"  {sp}: {total[sp]}")


if __name__ == "__main__":
    main()
