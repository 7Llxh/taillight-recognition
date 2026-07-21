# -*- coding: utf-8 -*-
"""一键训练 + 建库流程（整合 make_*/train_*/build_library）。

按顺序：数据集生成 -> 训练各模型 -> 构建特征库。
所有子脚本用当前 Python 解释器（carident）调用。

用例：
    # 全流程（从零或重训全部）
    python build_all.py

    # 加新车型时（朝向/部件检测器通用，可跳过；只重训特征网络 + 重建库）
    python build_all.py --only-embedder

    # 子集验证特征网络（快速跑通：跳过朝向/部件，make_* 取图片最多的 50 系列每 30 图）
    python build_all.py --only-embedder --limit 50 --max-per-series 30

    # 自定义跳过某组
    python build_all.py --skip-orientation --skip-parts

前置：原始图已放 data/raw/{车型}/（车尾/正面/侧面多角度）。
后续：python recognize.py <图片>
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# (显示名, 脚本, [参数], 组)
STEPS = [
    ("1. 朝向数据集",    "make_orientation_dataset.py", [],            "orientation"),
    ("2. 训练朝向分类器", "train_orientation.py",         [],            "orientation"),
    ("3. 部件数据集",    "make_parts_dataset.py",       [],            "parts"),
    ("4. 训练部件检测器", "train_parts.py",               [],            "parts"),
    ("5. 尾灯训练集",    "make_taillight_dataset.py",   [],            "embedder"),
    ("6. 整车训练集",    "make_vehicle_dataset.py",     [],            "embedder"),
    ("7. 训练尾灯网络",  "train_embedder.py",           ["taillight"], "embedder"),
    ("8. 训练整车网络",  "train_embedder.py",           ["vehicle"],   "embedder"),
    ("9. 构建尾灯库",    "build_library.py",            ["taillight"], "library"),
    ("10.构建整车库",    "build_library.py",            ["vehicle"],   "library"),
]


def run(name, script, args):
    cmd = [sys.executable, script, *args]
    print(f"\n{'='*60}\n[运行] {name}\n  {' '.join(cmd)}\n{'='*60}", flush=True)
    rc = subprocess.run(cmd, cwd=HERE).returncode
    if rc != 0:
        print(f"\n[失败] {name}（返回码 {rc}），流程中止。", flush=True)
        sys.exit(1)
    print(f"[完成] {name}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="一键训练+建库流程")
    ap.add_argument("--skip-orientation", action="store_true", help="跳过朝向数据集+训练")
    ap.add_argument("--skip-parts", action="store_true", help="跳过部件数据集+训练")
    ap.add_argument("--skip-embedder", action="store_true", help="跳过特征网络数据集+训练")
    ap.add_argument("--skip-library", action="store_true", help="跳过特征库构建")
    ap.add_argument("--only-embedder", action="store_true",
                    help="只重训特征网络+建库（加新车型常用，等价于 --skip-orientation --skip-parts）")
    ap.add_argument("--limit", type=int, default=None,
                    help="子集验证：make_taillight/vehicle_dataset 只取图片最多的 N 系列")
    ap.add_argument("--max-per-series", type=int, default=None,
                    help="子集验证：每系列最多 N 图（配合 --limit）")
    args = ap.parse_args()

    skip = set()
    if args.only_embedder:
        skip = {"orientation", "parts"}
    if args.skip_orientation:
        skip.add("orientation")
    if args.skip_parts:
        skip.add("parts")
    if args.skip_embedder:
        skip.add("embedder")
    if args.skip_library:
        skip.add("library")

    print(f"项目目录: {HERE}")
    print(f"跳过组: {sorted(skip) if skip else '无（全流程）'}")
    print(f"Python: {sys.executable}")

    for name, script, sargs, group in STEPS:
        if group in skip:
            print(f"[跳过] {name}")
            continue
        # 子集参数传给 make_taillight/vehicle_dataset
        extra = []
        if script in ("make_taillight_dataset.py", "make_vehicle_dataset.py"):
            if args.limit:
                extra += ["--limit", str(args.limit)]
            if args.max_per_series:
                extra += ["--max-per-series", str(args.max_per_series)]
        run(name, script, sargs + extra)

    print("\n全部完成 ✅  现在可识别: python recognize.py <图片>")


if __name__ == "__main__":
    main()
