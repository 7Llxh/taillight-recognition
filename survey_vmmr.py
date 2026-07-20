# -*- coding: utf-8 -*-
"""阶段 0：扫描 VMMRdb，按 make_model 归并，剔除图片过少的系列，产系列清单。

输出 data/vmmr_series.json：{make_model: {years:[...], img_count, img_paths:[...]}}
仅清单内系列参与后续数据集生成/建库；<MIN_IMGS_PER_SERIES 的系列剔除。

用法：
    python survey_vmmr.py             # 全量扫描，产清单
"""
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from config import RAW_DIR, MIN_IMGS_PER_SERIES, SUBSET_MIN_IMGS, SERIES_FILE

EXTS = (".jpg", ".jpeg", ".png", ".webp")


def survey():
    if not os.path.isdir(RAW_DIR):
        print(f"VMMRdb 目录不存在: {RAW_DIR}")
        return

    mm = defaultdict(lambda: {"years": set(), "img_paths": [], "img_count": 0})
    ndirs = 0
    for d in os.listdir(RAW_DIR):
        p = os.path.join(RAW_DIR, d)
        if not os.path.isdir(p):
            continue
        ndirs += 1
        make_model, _, year = d.rpartition("_")
        make_model = make_model or d
        imgs = [f for f in os.listdir(p) if f.lower().endswith(EXTS)]
        if not imgs:
            continue
        entry = mm[make_model]
        entry["years"].add(year)
        entry["img_paths"].extend(os.path.join(d, f) for f in imgs)  # 相对 RAW_DIR
        entry["img_count"] += len(imgs)

    # 剔除 < MIN_IMGS_PER_SERIES
    kept, dropped = {}, []
    for make_model, e in mm.items():
        e["years"] = sorted(e["years"])
        if e["img_count"] < MIN_IMGS_PER_SERIES:
            dropped.append((make_model, e["img_count"]))
        else:
            kept[make_model] = e

    subset_n = sum(1 for e in kept.values() if e["img_count"] >= SUBSET_MIN_IMGS)
    total_imgs = sum(e["img_count"] for e in kept.values())

    os.makedirs(os.path.dirname(SERIES_FILE), exist_ok=True)
    with open(SERIES_FILE, "w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False, indent=2)

    print(f"VMMRdb: {RAW_DIR}")
    print(f"make_model_year 目录: {ndirs}")
    print(f"归并系列(make_model): {len(mm)}")
    print(f"剔除 <{MIN_IMGS_PER_SERIES} 图: {len(dropped)} 系列")
    print(f"保留系列: {len(kept)}  (图片 {total_imgs:,})")
    print(f"子集验证(>={SUBSET_MIN_IMGS} 图): {subset_n} 系列")
    print(f"清单已写入: {SERIES_FILE}")
    print(f"\n剔除样例(图片最少的 10 个):")
    for m, c in sorted(dropped, key=lambda x: x[1])[:10]:
        print(f"  {m}: {c} 图")


if __name__ == "__main__":
    survey()
