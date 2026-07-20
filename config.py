# -*- coding: utf-8 -*-
"""VMMRdb 车型年份分离识别 - 全局配置。

各 make_*/build_library/survey/recognize 脚本 import 此处常量，避免硬编码。
VMMRdb 数据已复制入 data/raw，所有脚本统一读 data/raw。
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# === 数据源 ===
# 原始图目录（VMMRdb 已复制入 data/raw，统一数据源）
RAW_DIR = os.path.join(HERE, "data", "raw")

# === 系列图片数阈值 ===
# <此值的系列剔除（无训练价值），不生成数据集/不入库；识别时标"未知(样本不足)"
MIN_IMGS_PER_SERIES = 20
# 子集验证模式：仅处理 >=此值的系列（先子集验证再扩全量）
SUBSET_MIN_IMGS = 30

# === 输出 ===
SERIES_FILE = os.path.join(HERE, "data", "vmmr_series.json")  # survey 产的系列清单
DATA_DIR = os.path.join(HERE, "data")

# === 嵌入维度（与 embedder.py 一致）===
EMBED_DIM = 512
