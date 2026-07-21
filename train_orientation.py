# -*- coding: utf-8 -*-
"""训练 YOLOv8n-cls 朝向分类器（front/rear/side 三分类）。

用法（carident 环境）:
    python train_orientation.py

需先:
    1. 运行 make_orientation_dataset.py 生成数据集
    2. 下载 yolov8n-cls.pt 放到项目根（同 yolov8s.pt 一样手动下）
"""
import os
from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "orientation_dataset")


def main():
    model = YOLO("weights/yolov8n-cls.pt")
    model.train(
        data=DATA,
        epochs=50,
        imgsz=224,
        batch=64,
        project=os.path.join(HERE, "runs", "cls"),
        name="orientation",
    )
    print("训练完成。权重: runs/cls/orientation/weights/best.pt")


if __name__ == "__main__":
    main()
