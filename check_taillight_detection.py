# -*- coding: utf-8 -*-
"""检查部件检测器在各车型上的尾灯检测情况。

统计每车型: 图数、朝向分布、rear 图数、尾灯检测数、平均尾灯/rear图、rear图无尾灯数。
判读:
  - 平均尾灯/rear图 应 ≈ 2（每张车尾图左右各一尾灯）
  - 明显偏低(<1.5)或大量 rear 图无尾灯 -> 部件检测器漏检，需补标注 + 重训部件
  - 尾灯少但 rear 图也少 -> 不是漏检，是 rear 图少，补车尾图即可

用法:
    python check_taillight_detection.py              # 检查所有车型
    python check_taillight_detection.py 新车型名      # 只检查某车型
"""
import glob
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from detect_vehicle import _get_model, MODEL_PATH, _filter_vehicle_boxes, detect_taillights
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
    rear_no_tail = 0
    for p in imgs:
        img = imread_unicode(p)
        if img is None:
            continue
        h, w = img.shape[:2]
        res = _get_model(MODEL_PATH)(img, classes=[2, 5, 7], conf=0.4)[0]
        for box in _filter_vehicle_boxes(res.boxes, h, w):
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            vc = img[y1:y2, x1:x2]
            view = detect_orientation(vc)["view"]
            views[view] = views.get(view, 0) + 1
            if view == "rear":
                tls = detect_taillights(vc)
                if not tls:
                    rear_no_tail += 1
                rear_tail += len(tls)
    rear = views["rear"]
    avg = rear_tail / rear if rear else 0
    flag = ""
    if rear > 0 and avg < 1.5:
        flag += "  ⚠️ 平均<1.5，可能漏检"
    if rear > 0 and rear_no_tail / rear > 0.3:
        flag += f"  ⚠️ {rear_no_tail}/{rear} 张rear图无尾灯"
    print(f"{model}: 图{len(imgs)} | 朝向{views} | "
          f"rear图{rear} 尾灯{rear_tail} 平均{avg:.1f}/rear图{flag}")


def main():
    model_filter = sys.argv[1] if len(sys.argv) > 1 else None
    dirs = sorted(d for d in glob.glob(os.path.join(RAW, "*")) if os.path.isdir(d))
    if model_filter:
        dirs = [d for d in dirs if os.path.basename(d) == model_filter]
    if not dirs:
        print("未找到车型目录:", model_filter or RAW)
        return
    print(f"检查 {len(dirs)} 个车型（跑车辆+朝向+部件检测，稍等）...\n")
    for d in dirs:
        check_model(d)
    print("\n判读: 平均尾灯/rear图 应 ≈2。明显偏低或大量rear图无尾灯 = 漏检(补标注+重训部件)；"
          "尾灯少但rear图也少 = 数据少(补车尾图即可)。")


if __name__ == "__main__":
    main()
