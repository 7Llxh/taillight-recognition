# -*- coding: utf-8 -*-
"""评估朝向分类器（test 集，准确率 + 混淆矩阵）。"""
import glob
import os

import cv2
import numpy as np
from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "orientation_dataset")
WEIGHTS = os.path.join(HERE, "runs", "cls", "orientation", "weights", "best.pt")
CLASSES = ["front", "rear", "side"]


def imread_unicode(p):
    return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)


def main():
    model = YOLO(WEIGHTS)
    confusion = [[0] * 3 for _ in range(3)]
    correct = 0
    total = 0
    for i, cls in enumerate(CLASSES):
        for img in sorted(glob.glob(os.path.join(DATA, "test", cls, "*.jpg"))):
            arr = imread_unicode(img)
            r = model.predict(arr, verbose=False)[0]
            pred = CLASSES[int(r.probs.top1)]
            confusion[i][CLASSES.index(pred)] += 1
            if pred == cls:
                correct += 1
            total += 1
    acc = correct / total if total else 0
    print(f"Test 准确率: {correct}/{total} = {acc:.1%}")
    print("混淆矩阵 (行=真实, 列=预测):")
    print(f"{'':8}" + "".join(f"{c:>8}" for c in CLASSES))
    for i, c in enumerate(CLASSES):
        print(f"{c:8}" + "".join(f"{confusion[i][j]:>8}" for j in range(3)))


if __name__ == "__main__":
    main()
