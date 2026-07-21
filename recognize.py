# -*- coding: utf-8 -*-
"""识别程序（F11-F21）。输入图像 -> 检测/朝向 -> 尾灯或整车特征检索 -> 判别车型 + 可视化。

车尾(rear):尾灯特征 -> 尾灯库检索（主路径，精度高）。
正/侧(front/side)或尾灯置信度低:整车特征 -> 整车库检索（兜底）。

输出: {图名}_recognize/{图名}_recognize.jpg + {图名}_result.json。
"""
import glob
import json
import os
import sys

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from detect_vehicle import (_get_model, MODEL_PATH, _filter_vehicle_boxes,
                            detect_taillights, _judge_lit, crop_and_normalize)
from orientation import detect_orientation
from embedder import load_embedder, extract_features
from faiss_index import load_index, load_meta, search

TL_WEIGHTS = os.path.join(HERE, "runs", "embedder", "taillight", "best.pt")
VH_WEIGHTS = os.path.join(HERE, "runs", "embedder", "vehicle", "best.pt")
TL_INDEX = os.path.join(HERE, "data", "features", "taillight_index.faiss")
TL_META = os.path.join(HERE, "data", "features", "taillight_meta.json")
VH_INDEX = os.path.join(HERE, "data", "features", "vehicle_index.faiss")
VH_META = os.path.join(HERE, "data", "features", "vehicle_meta.json")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K = 5
MIN_CONF = 0.5  # 低于此置信度标"未知"

_tl_embedder = _tl_index = _tl_meta = None
_vh_embedder = _vh_index = _vh_meta = None


def _load():
    global _tl_embedder, _tl_index, _tl_meta, _vh_embedder, _vh_index, _vh_meta
    if _tl_embedder is None:
        _tl_embedder = load_embedder(TL_WEIGHTS, device=DEVICE)
        _tl_index = load_index(TL_INDEX)
        _tl_meta = load_meta(TL_META)
        # 整车兜底库（可能未构建，容错）
        if os.path.exists(VH_WEIGHTS) and os.path.exists(VH_INDEX):
            _vh_embedder = load_embedder(VH_WEIGHTS, device=DEVICE)
            _vh_index = load_index(VH_INDEX)
            _vh_meta = load_meta(VH_META)


def _search_library(embedder, index, meta, crop, k=TOP_K, lit_filter=None):
    norm = crop_and_normalize(crop, 224)
    feat = extract_features(embedder, [norm], device=DEVICE)[0]
    matches = search(index, feat, meta, k=k, lit_status_filter=lit_filter)
    if not matches and lit_filter is not None:  # 无同状态，回退全库
        matches = search(index, feat, meta, k=k)
    return matches


def _judge_model(matches):
    """车型 = 候选 model_series 众数（按计数）。置信度 = 该车型候选最高相似度。"""
    if not matches:
        return "未知(置信度低)", 0.0
    from collections import Counter
    c = Counter(m["model_series"] for m in matches)
    model, _ = c.most_common(1)[0]
    conf = max(m["similarity"] for m in matches if m["model_series"] == model)
    if conf < MIN_CONF:
        return "未知(置信度低)", conf
    return model, conf


def _judge_year(matches, model):
    """该车型候选 year 统计：集中给范围，分散标不确定。无年份标未知。"""
    years = [m.get("year") for m in matches if m["model_series"] == model and m.get("year")]
    if not years:
        return "年份未知"
    from collections import Counter
    yrs = sorted(int(y) for y in years)
    span = yrs[-1] - yrs[0]
    if len(yrs) >= 3 and span <= 2:
        return f"约{yrs[0]}-{yrs[-1]}"
    if len(yrs) >= 3:
        return f"年份不确定({yrs[0]}-{yrs[-1]})"
    if span <= 2:
        return f"约{yrs[0]}-{yrs[-1]}" if yrs[0] != yrs[-1] else f"约{yrs[0]}"
    mode = Counter(yrs).most_common(1)[0][0]
    return f"约{mode}({yrs[0]}-{yrs[-1]})"


def _draw(img, box, label, conf, view, color):
    x1, y1, x2, y2 = box
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, f"{label} {conf:.2f} [{view}]", (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def recognize(image_path):
    _load()
    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print("读图失败:", image_path)
        return []
    h, w = img.shape[:2]
    res = _get_model(MODEL_PATH)(img, classes=[2, 5, 7], conf=0.4)[0]
    result_img = img.copy()
    results = []

    for box in _filter_vehicle_boxes(res.boxes, h, w):
        x1, y1, x2, y2 = box.xyxy[0].int().tolist()
        vcrop = img[y1:y2, x1:x2]
        view = detect_orientation(vcrop)["view"]

        if view == "rear":
            # 车尾:尾灯检索（主路径）
            tls = detect_taillights(vcrop)
            if not tls:
                results.append({"box": [x1, y1, x2, y2], "view": view, "result": "未检测到尾灯"})
                _draw(result_img, (x1, y1, x2, y2), "无尾灯", 0, view, (255, 255, 0))
                continue
            # 画所有尾灯框(映射回原图:红=亮灯,绿=灭灯)
            for t in tls:
                ttx1, tty1, ttx2, tty2 = t["box"]
                ttc = vcrop[tty1:tty2, ttx1:ttx2]
                if ttc.size == 0:
                    continue
                t_lit, _ = _judge_lit(ttc)
                tcolor = (0, 0, 255) if t_lit else (0, 255, 0)
                cv2.rectangle(result_img, (x1+ttx1, y1+tty1), (x1+ttx2, y1+tty2), tcolor, 2)
            tl = max(tls, key=lambda t: t["conf"])
            tx1, ty1, tx2, ty2 = tl["box"]
            tc = vcrop[ty1:ty2, tx1:tx2]
            is_lit, _ = _judge_lit(tc)
            matches = _search_library(_tl_embedder, _tl_index, _tl_meta, tc, lit_filter=is_lit)
            model, conf = _judge_model(matches)
            year_range = _judge_year(matches, model) if model != "未知(置信度低)" else "年份未知"
            color = (0, 0, 255) if is_lit else (0, 255, 0)
            results.append({"box": [x1, y1, x2, y2], "view": view, "lit": bool(is_lit),
                            "make_model": model, "year_range": year_range,
                            "confidence": round(conf, 3),
                            "topk": [{"model": m["model_series"], "sim": round(m["similarity"], 3)}
                                     for m in matches[:3]]})
            _draw(result_img, (x1, y1, x2, y2), f"{model} {year_range}", conf, view, color)
        else:
            # 正/侧:整车兜底
            if _vh_index is None:
                results.append({"box": [x1, y1, x2, y2], "view": view, "result": "整车兜底库未构建"})
                _draw(result_img, (x1, y1, x2, y2), "兜底库未构建", 0, view, (255, 255, 0))
                continue
            matches = _search_library(_vh_embedder, _vh_index, _vh_meta, vcrop)
            model, conf = _judge_model(matches)
            year_range = _judge_year(matches, model) if model != "未知(置信度低)" else "年份未知"
            results.append({"box": [x1, y1, x2, y2], "view": view,
                            "make_model": model, "year_range": year_range,
                            "confidence": round(conf, 3),
                            "topk": [{"model": m["model_series"], "sim": round(m["similarity"], 3)}
                                     for m in matches[:3]]})
            _draw(result_img, (x1, y1, x2, y2), f"{model} {year_range}", conf, f"{view}兜底", (255, 200, 0))

    stem = os.path.splitext(os.path.basename(image_path))[0]
    od = f"{os.path.splitext(image_path)[0]}_recognize"
    os.makedirs(od, exist_ok=True)
    cv2.imencode(".jpg", result_img)[1].tofile(os.path.join(od, f"{stem}_recognize.jpg"))
    with open(os.path.join(od, f"{stem}_result.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"识别完成。结果: {od}/")
    for r in results:
        print(" ", r)
    return results


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else input("图片路径: ").strip().strip('"')
    if os.path.isdir(path):
        imgs = [p for p in glob.glob(os.path.join(path, "*"))
                if p.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]
        print(f"目录模式: {path} 下 {len(imgs)} 张图，逐张识别...")
        for p in imgs:
            recognize(p)
    else:
        recognize(path)
