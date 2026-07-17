# -*- coding: utf-8 -*-
"""建库程序（F7/F8/F9）。特征库构建（尾灯/整车）。

用训练好的嵌入网络对训练集提取 512 维特征，构建 FAISS 索引 + 元数据。
用法:
    python build_library.py taillight   # 尾灯库（主）
    python build_library.py vehicle      # 整车库（兜底）
输出 data/features/{name}_index.faiss + {name}_meta.json。
"""
import glob
import os
import sys

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from embedder import load_embedder, extract_features
from faiss_index import build_faiss_index, save_index, save_meta

NAME = sys.argv[1] if len(sys.argv) > 1 else "taillight"
WEIGHTS = os.path.join(HERE, "runs", "embedder", NAME, "best.pt")
DATA = os.path.join(HERE, "data", f"{NAME}_dataset", "train")
OUT_DIR = os.path.join(HERE, "data", "features")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def imread_unicode(p):
    return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)


def main():
    if not os.path.exists(WEIGHTS):
        print(f"权重不存在: {WEIGHTS}，先训练: python train_embedder.py {NAME}")
        return
    os.makedirs(OUT_DIR, exist_ok=True)
    model = load_embedder(WEIGHTS, device=DEVICE)

    items = []
    for model_dir in sorted(glob.glob(os.path.join(DATA, "*"))):
        if not os.path.isdir(model_dir):
            continue
        model_series = os.path.basename(model_dir)
        for p in sorted(glob.glob(os.path.join(model_dir, "*.jpg"))):
            img = imread_unicode(p)
            if img is None:
                continue
            items.append((img, model_series, p))

    print(f"[{NAME}] device={DEVICE}  入库样本: {len(items)}")
    if not items:
        return

    crops = [it[0] for it in items]
    feats = extract_features(model, crops, device=DEVICE)

    meta = []
    for i, (crop, model_series, p) in enumerate(items):
        entry = {"id": i, "model_series": model_series, "sample_path": p}
        if NAME == "taillight":
            from detect_vehicle import _judge_lit
            is_lit, v_mean = _judge_lit(crop)
            entry["view"] = "rear"
            entry["lit_status"] = bool(is_lit)
            entry["v_mean"] = round(v_mean, 1)
        else:
            entry["view"] = "vehicle"
            entry["lit_status"] = None  # 整车不分亮灭
        meta.append(entry)

    index = build_faiss_index(feats)
    save_index(index, os.path.join(OUT_DIR, f"{NAME}_index.faiss"))
    save_meta(meta, os.path.join(OUT_DIR, f"{NAME}_meta.json"))
    msg = f"{NAME} 特征库已构建: {OUT_DIR}/{NAME}_index.faiss  样本: {len(meta)}"
    if NAME == "taillight":
        lit_n = sum(1 for m in meta if m["lit_status"])
        msg += f"  亮灯: {lit_n}  灭灯: {len(meta) - lit_n}"
    print(msg)


if __name__ == "__main__":
    main()
