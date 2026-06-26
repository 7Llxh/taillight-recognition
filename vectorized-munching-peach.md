# 汽车尾灯品牌车型识别项目

## Context

通过汽车尾灯识别汽车品牌和年代车型。尾灯特点：
- 不同品牌/车型的尾灯形状不同
- 灭灯时与车身颜色可区分
- 晚上亮灯时无需补光即可清晰识别，且无前大灯干扰

技术选型：Python + PyTorch，学习研究性质项目，需从头收集数据。

---

## 项目目录结构

```
car_taillight_recognition/
├── config/                    # 配置文件
├── data/
│   ├── raw/                   # 原始图像/视频
│   ├── processed/             # 处理后数据
│   ├── annotations/           # 标注文件
│   └── features/              # 特征向量库
├── src/
│   ├── data_collection/       # 数据收集模块
│   ├── preprocessing/         # 预处理模块
│   ├── feature_extraction/    # 特征提取模块
│   ├── models/                # 模型定义
│   ├── inference/             # 推理模块
│   ├── training/              # 训练模块
│   └── utils/                 # 工具函数
├── scripts/                   # 运行脚本
├── notebooks/                 # 实验笔记本
├── checkpoints/               # 模型权重
└── tests/                     # 单元测试
```

---

## 实现步骤

### 步骤1：数据收集
- **来源**：汽车之家/懂车帝（静态图）、YouTube/B站（视频）、停车场自采集
- **技术**：requests + BeautifulSoup 爬虫、yt-dlp 视频下载
- **目标规模**：50-100品牌，每品牌10-30车型，每车型亮灯+灭灯各30张

### 步骤2：亮灯/灭灯分类
- **方法**：基于HSV颜色空间的红色区域检测（尾灯典型颜色）
- **阈值**：红色像素占比 > 15% 视为亮灯

### 步骤3：车辆后轮廓分割
- **模型**：YOLOv8-seg 或 SAM (Segment Anything)
- **输出**：车辆后视图的分割mask

### 步骤4：尾灯检测与裁剪
- **检测**：YOLOv8 微调（类别：left_taillight, right_taillight）
- **归一化**：裁剪后缩放至 224×224，保持宽高比并填充

### 步骤5：特征向量库构建
- **模型**：ResNet-50 / EfficientNet-B4 骨干网络
- **嵌入维度**：512维
- **训练**：三元组损失 (Triplet Loss) 度量学习
- **检索**：FAISS 向量索引，支持快速相似度搜索

### 步骤6：视频车型识别
- **管道**：车辆检测 → 车辆跟踪 → 尾灯检测 → 特征提取 → 检索匹配
- **跟踪**：ByteTracker 跨帧关联

---

## 关键依赖

```text
torch>=2.0.0
ultralytics>=8.0.0          # YOLOv8
segment-anything            # SAM分割
albumentations>=1.3.0       # 数据增强
faiss-cpu>=1.7.4            # 向量检索
pytorch-metric-learning     # 度量学习
opencv-python>=4.8.0
```

---

## 关键文件

| 文件 | 作用 |
|------|------|
| `src/preprocessing/taillight_detector.py` | 尾灯检测核心逻辑 |
| `src/feature_extraction/embedding.py` | 特征嵌入网络 |
| `src/inference/pipeline.py` | 完整推理管道 |
| `scripts/build_feature_db.py` | 特征库构建脚本 |

---

## 验证方法

1. **数据阶段**：人工检查亮灯/灭灯分类准确率
2. **检测阶段**：在测试集上评估尾灯检测 mAP@0.5
3. **识别阶段**：计算 Recall@K（K=1,5,10）评估检索准确率
4. **视频阶段**：在测试视频上演示完整流程，可视化识别结果

---

## 实施时间线

| 阶段 | 时间 | 内容 |
|------|------|------|
| Phase 1 | 第1-2周 | 环境搭建、数据收集、标注 |
| Phase 2 | 第3-4周 | 车辆分割、尾灯检测模型训练 |
| Phase 3 | 第5-6周 | 特征网络训练、特征库构建 |
| Phase 4 | 第7-8周 | 视频推理管道开发、测试 |
| Phase 5 | 第9-10周 | 优化、文档 |