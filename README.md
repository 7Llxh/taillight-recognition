# 汽车尾灯车型识别系统

基于图像检索的车型判别系统。构建尾灯/整车特征向量库(FAISS),以"特征检索"方式判别车型系列。

## 简介

不同品牌/车型的尾灯在形状、轮廓、颜色上具有差异化特征,尤其夜间亮灯时清晰可靠。系统按车辆朝向分流检索:

- **车尾视图**:尾灯特征 → 检索【尾灯库】(主路径,精度高)
- **正面/侧面**:整车特征 → 检索【整车库】(兜底)

## 整体流程

```
输入图 -> 车辆检测(YOLOv8s) -> 朝向识别(YOLOv8n-cls)
   ├─ 车尾 rear -> 部件检测器定位尾灯 -> 尾灯特征(ResNet50+ArcFace) -> 检索尾灯库(主)
   └─ 正面/侧面   -> 整车特征 -> 检索整车库(兜底)
-> 判别车型 + 置信度 + Top-K + 可视化(结果图 + JSON)
```

## 环境配置

Python 3.12 | conda 环境 `carident` | GPU: RTX 5060(Blackwell,需 cu128)

```bash
conda create -n carident python=3.12 -y
conda activate carident
```

### 安装依赖

完整依赖见 [requirements.txt](requirements.txt)。关键点:

**GPU torch(RTX 5060 Blackwell sm_120,必须 cu128)**:官方 index 在国内被 SSL 间歇拦截,用阿里云直装 wheel(绕过 index 页):

```bash
pip install \
  https://mirrors.aliyun.com/pytorch-wheels/cu128/torch-2.11.0+cu128-cp312-cp312-win_amd64.whl \
  https://mirrors.aliyun.com/pytorch-wheels/cu128/torchvision-0.26.0+cu128-cp312-cp312-win_amd64.whl
```

> 单连接下载被限速(~0.45MB/s),用多线程(aria2 `-x16` 或 Python 分块 Range)可达 ~7MB/s。
> conda/pip 镜像走清华(用户目录 `.condarc` + `AppData/Roaming/pip/pip.ini`)。

**其他依赖**:

```bash
pip install ultralytics opencv-python numpy Pillow pandas scipy scikit-learn tqdm \
            faiss-cpu pytorch-metric-learning requests selenium
```

### 验证

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# torch 2.11.0+cu128 True
```

## 项目结构

```
├── detect_vehicle.py        # 车辆检测 + 部件检测器尾灯定位 + 亮灭判定
├── orientation.py            # 朝向识别(front/rear/side)
├── annotate_tool.py          # 交互式标注工具(部件框 + 朝向)
├── make_orientation_dataset.py / train_orientation.py / eval_orientation.py
├── make_parts_dataset.py / train_parts.py
├── make_taillight_dataset.py / make_vehicle_dataset.py
├── embedder.py / train_embedder.py          # ResNet50 + ArcFace 嵌入网络
├── faiss_index.py            # FAISS 构建与检索
├── build_library.py          # 建库程序(尾灯/整车)
├── recognize.py              # 识别程序(双路径 + 可视化)
├── build_all.py              # 一键训练+建库
├── check_taillight_detection.py  # 检查尾灯漏检 + 产标注队列
├── visualize_parts.py        # 部件检测可视化(画8类部件框,看准确度)
├── config.py                 # VMMRdb 配置(数据路径/阈值)
├── survey_vmmr.py            # 扫 VMMRdb 归并系列 + 剔除稀疏
├── requirements.txt
├── 需求分析.md / 原理.md / *-技术设计.md
├── docs/实施计划/            # 实施计划文档
├── data/
│   ├── raw/{车型}/           # 原始图(VMMRdb 全量 9170 车型目录)
│   ├── annotations/          # 标注 JSON
│   ├── annotate_queue.json   # 标注增强队列(check 产,gitignore)
│   ├── vmmr_series.json      # VMMRdb 系列清单(survey 产,gitignore)
│   ├── orientation_dataset/  # 朝向数据集
│   ├── parts_dataset/        # 部件数据集
│   ├── taillight_dataset/    # 尾灯特征训练集
│   ├── vehicle_dataset/      # 整车特征训练集
│   └── features/             # FAISS 库 + 元数据(含 year)
└── runs/
    ├── cls/orientation/                      # 朝向分类器
    ├── detect/parts_v2/                      # 部件检测器
    └── embedder/{taillight,vehicle}/         # 嵌入网络
```

## 快速开始

库已构建好,直接识别:

```bash
conda activate carident
python recognize.py data/raw/acura_mdx_2007/00101_59bqNeCJGMt_600x450.jpg
```

输出到 `{图名}_recognize/`:结果图(标注车型 + 置信度)+ `{图名}_result.json`。

## 完整流程(从零构建)

### 1. 数据准备

原始图放 `data/raw/{车型}/`(车尾/正面/侧面多角度,含亮灯+灭灯)。

### 2. 标注(产出部件/朝向数据集)

```bash
python annotate_tool.py   # 拖拽标部件框 + 选朝向,存 data/annotations/*.json
```

### 3. 训练 + 建库（一键）

```bash
python build_all.py                  # 全流程（数据集->训练->建库）
```

加新车型时（朝向/部件检测器通用，可跳过，只重训特征网络 + 重建库）：

```bash
python build_all.py --only-embedder
```

自定义跳过某组：

```bash
python build_all.py --skip-orientation --skip-parts
```

手动命令（参考，等价于 `build_all.py`）：

```bash
python make_orientation_dataset.py && python train_orientation.py
python make_parts_dataset.py && python train_parts.py
python make_taillight_dataset.py && python make_vehicle_dataset.py
python train_embedder.py taillight
python train_embedder.py vehicle
python build_library.py taillight
python build_library.py vehicle
```

### 4. 加新车型（先 check 再决定是否标注）

加新车型**不一定需要标注**。先用 check 检查部件检测器在新车型上是否漏检尾灯：

```bash
python check_taillight_detection.py 新车型名   # 只查新车型（也可不传参数查全部）
```

check 同时把漏检图（rear 无尾灯）写入 `data/annotate_queue.json`，供标注工具按 `n` 逐张跳转补标。

看输出「平均尾灯/rear图」（应 ≈2，每张车尾图左右各一尾灯）：

| 平均尾灯/rear图 | 含义 | 操作 |
|---|---|---|
| ≈2 | 部件检测器正常 | `python build_all.py --only-embedder`（重训特征网络+建库） |
| <1.5 或大量rear图无尾灯 | 部件检测器**漏检**（尾灯形状没见过） | 标注部件框 + `python build_all.py --skip-orientation`（重训部件+特征网络+建库） |

> 标注：`python annotate_tool.py`，按 `n` 跳到 check 产的漏检图队列，画尾灯框保存自动出队（每车型 20-30 张 rear 图）。重训后再跑 check 验证平均升到 ~2。
> 判读依据：尾灯少但 rear 图也少 = 数据少（补车尾图）；rear 图多但尾灯少 = 漏检（补标注+重训部件）。

### 5. 识别

```bash
python recognize.py <图片路径>
```

## 模型与指标

| 模型 | 产出 | 指标 |
|---|---|---|
| 朝向分类器 | runs/cls/orientation | front/rear/side 三类 |
| 部件检测器 | runs/detect/parts_v2 | **taillight mAP50=0.99, recall=0.967** |
| 尾灯嵌入网络 | runs/embedder/taillight | **val R@1=0.917**(目标 ≥0.70) |
| 整车嵌入网络 | runs/embedder/vehicle | val R@1=0.50, R@5=0.67-0.75(目标 ≥0.60) |

特征库:`data/features/{taillight,vehicle}_index.faiss` + `{name}_meta.json`。

端到端识别验证(库内车型):acura(尾灯主路径)**0.997**、bmw(整车兜底)**0.972**。

## 关键设计决策

- **部件检测器替代 HSV 红色分割**:HSV 尾灯定位在灭灯/红色干扰下不准,改用训练好的部件检测器(不依赖颜色,灭灯鲁棒)。
- **ArcFace 度量学习(非三元组)**:小数据下收敛更稳、无需难样本挖掘。
- **双特征库**:尾灯库(主,车尾)+ 整车库(兜底,正/侧),按朝向分流。
- **统一库 + lit_status 过滤**:亮/灭灯样本共库,检索时按查询状态过滤。

## 数据局限

- data/raw 现含 VMMRdb 全量(9170 车型目录),特征库已用 VMMRdb 50 系列子集验证(见 [docs/实施计划/](docs/实施计划/))。
- 部件检测器漏检:buick/cadillac 尾灯样本少(buick 5、cadillac 2),**不是 rear 图少**,是部件检测器只训练了 acura+audi、在这两车型上漏检尾灯(check 显示 buick 58/62、cadillac 43/45 张 rear 图无尾灯),需补标注+重训部件检测器。
- 库外图置信度低(合理,不匹配)。

## 后续优化

- 扩车型到 20-50,补 rear 图(尤其 cadillac/buick,含灭灯样本)。
- 系统评估脚本(批量 Recall@K,目前为单图验证)。
- 增量入库(新车型不重训)。
- 视频流识别(Phase 4 扩展)。

## 文档

- [需求分析.md](docs/需求分析.md) — 完整需求与架构
- [原理.md](docs/原理.md) — 检测原理(部件检测器定位尾灯)
- [朝向识别-技术设计.md](docs/朝向识别-技术设计.md)
- [特征提取网络-技术设计.md](docs/特征提取网络-技术设计.md)
- [交互式标注工具-技术设计.md](docs/交互式标注工具-技术设计.md)
- [部件识别-技术设计.md](docs/部件识别-技术设计.md) - 8 类部件检测器（taillight mAP50=0.99）
- [尾灯定位-技术设计.md](docs/尾灯定位-技术设计.md) - 部件检测器定位尾灯 + 亮灭判定
- [识别程序-技术设计.md](docs/识别程序-技术设计.md) - 双路径融合 + 可视化（recognize.py）
- [数据标注规范.md](docs/数据标注规范.md) - 部件类别/朝向/数据组织/划分
