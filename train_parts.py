# -*- coding: utf-8 -*-
"""训练 YOLOv8s 部件检测器（8 类部件）。

用法（carident 环境）:
    python train_parts.py

需先:
    1. 运行 make_parts_dataset.py 生成数据集
    2. yolov8s.pt 已在项目根（之前下过）
    3. 安装 GPU 版 torch（见下方说明）

GPU 说明（RTX 5060 为 Blackwell 架构 sm_120，必须用 cu128，装 cu121/cu124 会报
"no kernel image is available"）:
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
若官方源拉不动，用阿里云镜像直装指定 wheel（cp312 = carident 环境 Python 3.12）:
    pip install \
      https://mirrors.aliyun.com/pytorch-wheels/cu128/torch-2.11.0+cu128-cp312-cp312-win_amd64.whl \
      https://mirrors.aliyun.com/pytorch-wheels/cu128/torchvision-0.26.0+cu128-cp312-cp312-win_amd64.whl \
      -i https://mirrors.aliyun.com/pypi/simple/
验证: python -c "import torch; print(torch.cuda.is_available())"  # 应为 True
"""
import os

import torch
from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "parts_dataset", "data.yaml")


def pick_device():
    """优先 GPU；不可用时回退 CPU 并打印提示。"""
    if torch.cuda.is_available():
        print(f"[device] 使用 GPU: {torch.cuda.get_device_name(0)}")
        return 0
    print("[device] 未检测到 CUDA，回退 CPU（训练会很慢，请按文件头说明安装 GPU 版 torch）。")
    return "cpu"


def main():
    device = pick_device()
    model = YOLO("weights/yolov8s.pt")
    model.train(
        data=DATA,
        epochs=50,
        imgsz=640,
        batch=8,   # 临时降（numpy 内存碎片），跑通后调回 16
        device=device,
        workers=8,
        project=os.path.join(HERE, "runs", "detect"),
        name="parts_v2",          # exist_ok 直接覆盖 parts_v2，供 detect_vehicle 用
        exist_ok=True,
        # 小样本增强(103 图)：mixup 正则、轻微旋转、末段关 mosaic 提精度
        mixup=0.0,   # 临时关 mixup（float64 双图触发 numpy._ArrayMemoryError），跑通后调回 0.15
        degrees=5.0,
        close_mosaic=10,
    )
    print("训练完成。权重: runs/detect/parts_v2/weights/best.pt")


if __name__ == "__main__":
    main()
