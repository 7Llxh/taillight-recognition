# -*- coding: utf-8 -*-
"""检查部件检测器在各车型上的尾灯检测情况。

统计每车型: 图数、朝向分布、rear 图尾灯数分布（0/1/2+）、rear图无尾灯数。
判读:
  - 0尾灯/rear图 比例应低;1尾灯(后侧方单侧)/2+尾灯(正后双侧)均正常
  - 0尾灯比例高 -> 部件检测器漏检，需补标注 + 重训部件
  - rear 图少 -> 不是漏检，是 rear 图少，补车尾图即可

用法:
    python check_taillight_detection.py              # 检查所有车型
    python check_taillight_detection.py 新车型名      # 只检查某车型
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from detect_vehicle import _get_model, MODEL_PATH, _pick_main_vehicle, detect_taillights
from orientation import detect_orientation

RAW = os.path.join(HERE, "data", "raw")
IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def imread_unicode(p):
    return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)


def check_model(model_dir):
    model = os.path.basename(model_dir)
    imgs = [p for p in glob.glob(os.path.join(model_dir, "*")) if p.lower().endswith(IMG_EXTS)]
    views = {"front": 0, "rear": 0, "side": 0, "uncertain": 0}
    rear_tail = 0
    rear_0 = rear_1 = rear_2 = 0  # 尾灯数分布：0=漏检, 1=后侧方(单侧), 2+=正后(双侧)
    queue = []
    for p in imgs:
        img = imread_unicode(p)
        if img is None:
            continue
        h, w = img.shape[:2]
        res = _get_model(MODEL_PATH)(img, classes=[2, 5, 7], conf=0.4, verbose=False)[0]
        box = _pick_main_vehicle(res.boxes, h, w)
        if box is None:
            continue
        x1, y1, x2, y2 = box.xyxy[0].int().tolist()
        vc = img[y1:y2, x1:x2]
        view = detect_orientation(vc)["view"]
        views[view] = views.get(view, 0) + 1
        if view == "rear":
            tls = detect_taillights(vc)
            n = len(tls)
            if n == 0:
                rear_0 += 1
                queue.append({
                    "path": os.path.relpath(p, HERE).replace("\\", "/"),
                    "model": model,
                    "view": "rear",
                    "vehicle_box": [x1, y1, x2, y2],
                    "reason": "rear无尾灯",
                })
            elif n == 1:
                rear_1 += 1  # 后侧方单尾灯（正常）
            else:
                rear_2 += 1  # 正后双尾灯（正常）
            rear_tail += n
    rear = views["rear"]
    avg = rear_tail / rear if rear else 0
    flag = ""
    if rear > 0 and rear_0 / rear > 0.3:
        flag += f"  ⚠️ {rear_0}/{rear} 张rear图无尾灯（漏检）"
    print(f"{model}: 图{len(imgs)} | 朝向{views} | "
          f"rear图{rear}(0灯{rear_0}/1灯{rear_1}/2+灯{rear_2}) 尾灯{rear_tail} 平均{avg:.1f}{flag}")
    return queue


def main():
    ap = argparse.ArgumentParser(description="检查部件检测器在各车型的尾灯检测")
    ap.add_argument("model", nargs="?", default=None, help="只检查某车型（data/raw 下目录名）")
    ap.add_argument("--limit", type=int, default=None,
                    help="只检查图片最多的 N 系列（top N make_model，从 vmmr_series.json 取）")
    args = ap.parse_args()

    if args.limit:
        series = json.load(open(os.path.join(HERE, "data", "vmmr_series.json"), encoding="utf-8"))
        top = [m for m, _ in sorted(series.items(), key=lambda kv: kv[1]["img_count"], reverse=True)[:args.limit]]
        dirs = []
        for m in top:
            dirs += sorted(glob.glob(os.path.join(RAW, m + "_*")))
        dirs = [d for d in dirs if os.path.isdir(d)]
        print(f"--limit {args.limit}: top {len(top)} 系列, {len(dirs)} 个年款目录")
    else:
        dirs = sorted(d for d in glob.glob(os.path.join(RAW, "*")) if os.path.isdir(d))
        if args.model:
            dirs = [d for d in dirs if os.path.basename(d) == args.model]
    if not dirs:
        print("未找到车型目录:", args.model or RAW)
        return
    print(f"检查 {len(dirs)} 个车型（跑车辆+朝向+部件检测，稍等）...\n")
    all_queue = []
    for d in dirs:
        all_queue.extend(check_model(d))
    queue_path = os.path.join(HERE, "data", "annotate_queue.json")
    with open(queue_path, "w", encoding="utf-8") as f:
        json.dump(all_queue, f, ensure_ascii=False, indent=2)
    print(f"\n标注队列已写入: {queue_path}")
    print(f"需增强标注: {len(all_queue)} 张（rear 无尾灯）")
    print("用 annotate_tool.py 按 n 逐张补标尾灯框。")
    print("\n判读: 0尾灯/rear图 比例应低;1尾灯(后侧方单侧)/2+尾灯(正后双侧)均正常。"
          "0尾灯比例高 = 漏检(补标注+重训部件);rear图少 = 数据少(补车尾图)。")


if __name__ == "__main__":
    main()
