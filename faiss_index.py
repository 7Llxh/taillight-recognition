# -*- coding: utf-8 -*-
"""FAISS 特征库构建与检索（F9/F18）。IndexFlatIP（cosine）。

向量 L2 归一化后内积 = cosine 相似度。尾灯库/整车库各一份。
元数据（车型、朝向、亮灭、样本路径、尾灯框）存 JSON，向量 id 与元数据 id 对齐。
检索时按 lit_status 过滤候选集。详见 特征提取网络-技术设计.md。
"""
import json

import faiss
import numpy as np

DIM = 512


def build_faiss_index(features):
    """构建 IndexFlatIP（内积；配 L2 归一化向量 = cosine）。features (N,512)。"""
    feats = np.ascontiguousarray(features, dtype=np.float32)
    faiss.normalize_L2(feats)  # 确保归一化
    index = faiss.IndexFlatIP(DIM)
    index.add(feats)
    return index


def save_index(index, path):
    # faiss.write_index 用 C fopen，不支持中文路径；改用 serialize + numpy tofile
    faiss.serialize_index(index).tofile(path)


def load_index(path):
    return faiss.deserialize_index(np.fromfile(path, dtype=np.uint8))


def save_meta(meta, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def load_meta(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def search(index, query_feat, meta, k=10, lit_status_filter=None):
    """检索 Top-K。query_feat (512,)。

    lit_status_filter: True/False 时只在该状态子集检索（亮灯查亮灯，灭灯查灭灯）；
                      None 全库。
    返回 [{id, model_series, lit_status, similarity, ...}]（按相似度降序）。
    """
    q = np.ascontiguousarray(query_feat.reshape(1, -1), dtype=np.float32)
    faiss.normalize_L2(q)
    # 状态过滤：先全库检索，再按 lit_status 筛
    search_k = len(meta) if lit_status_filter is not None else k
    search_k = max(search_k, k)
    sims, idxs = index.search(q, search_k)
    results = []
    for sim, idx in zip(sims[0], idxs[0]):
        if idx < 0:
            continue
        m = meta[idx]
        if lit_status_filter is not None and m.get("lit_status") != lit_status_filter:
            continue
        results.append({**m, "similarity": float(sim)})
        if len(results) >= k:
            break
    return results
