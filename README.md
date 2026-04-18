# 梨表面缺陷检测融合项目

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-Research-green.svg)](LICENSE)

本项目实现了基于**拓扑感知先验**的梨表面缺陷检测融合方案，通过结合图像分割先验与 YOLO26 目标检测，提升瘀伤和划伤的检测精度（mAP50）。

## 🎯 项目概述

### 核心思路

本项目融合两个独立系统：

- **Project A (Image Segmentation)**: 基于拓扑感知不确定性的分割模型，提供像素级软概率图（likelihood）和拓扑遮罩（topology_mask）
- **Project B (YOLO26-pear)**: 基于 YOLO26 的目标检测 baseline

### E1 实验：Global + Local Detector

**两阶段检测流程**：

1. **全局检测器**：使用 YOLO26 进行全图检测
2. **局部检测器**：基于 topology_mask 的 ROI 区域进行高分辨率检测
3. **智能融合**：优先级 NMS 合并两个检测器的结果（local > global）

### 工程原则

✅ **低侵入** - 不修改源项目代码  
✅ **可回退** - 保持 baseline 可运行  
✅ **可消融** - 每个组件可独立评估  
✅ **可解释** - 明确每个改进的来源

## 📦 环境准备

### 系统要求

- **操作系统**: Linux (推荐 Ubuntu 20.04+)
- **GPU**: NVIDIA RTX 4090 或同等性能 GPU
- **CUDA**: 11.8+
- **Python**: 3.10+

### 部署方式

本项目支持两种部署方式：

#### 方式 1：独立部署（推荐）⭐

**无需外部项目代码**，只需要准备：
1. **Priors 数据**（预先生成的 likelihood.npy, topology_mask.npy）
2. **预训练模型**（YOLO baseline 模型 .pt 文件）
3. **数据集**（YOLO 格式）

详见：[独立部署指南](STANDALONE_DEPLOYMENT.md)

#### 方式 2：完整部署

需要依赖两个外部项目：
1. **Image Segmentation** - 用于生成 priors
2. **YOLO26-pear** - 提供 baseline 模型

详见：[服务器部署指南](SERVER_DEPLOYMENT.md)

### 环境安装

#### 方案 1：使用现有环境（快速开始）

如果你已经有 `pear-topo` 和 `yolo` 环境：

```bash
# 激活 yolo 环境
conda activate yolo

# 安装额外依赖
pip install opencv-python pyyaml tqdm
```

#### 方案 2：创建新环境（推荐）

```bash
# 克隆仓库
git clone git@github.com:colorful-tju/pear-defect-detection-fusion.git
cd pear-defect-detection-fusion

# 创建环境
conda create -n pear-fusion python=3.10
conda activate pear-fusion

# 安装依赖
pip install -r requirements.txt
```

## 🚀 快速开始

### 独立部署（推荐）

如果你已经有 priors 数据和模型文件：

#### 1. 克隆仓库

```bash
git clone git@github.com:colorful-tju/pear-defect-detection-fusion.git
cd pear-defect-detection-fusion
```

#### 2. 创建环境

```bash
conda create -n pear-fusion python=3.10
conda activate pear-fusion
pip install -r requirements.txt
```

#### 3. 准备文件

```bash
# 准备 priors 数据（从其他机器复制或生成）
cp -r /path/to/priors outputs/

# 准备模型文件
mkdir -p models
cp /path/to/baseline_model.pt models/global_detector.pt
```

#### 4. 配置路径

编辑 `configs/e1_config_standalone.yaml`：

```yaml
dataset:
  root: /home/robot/yolo/datasets/PearSurfaceDefects
  data_yaml: /home/robot/yolo/datasets/PearSurfaceDefects/li_data.yaml

priors:
  root_dir: outputs/priors

models:
  global_detector: models/global_detector.pt
```

#### 5. 运行训练

```bash
python tools/train_e1_standalone.py --config configs/e1_config_standalone.yaml
```

#### 6. 运行推理

```bash
python scripts/infer_e1_fusion.py \
  --config configs/e1_config_standalone.yaml \
  --source test_images/ \
  --visualize
```

### 完整部署

如果你需要从头生成 priors，参考 [服务器部署指南](SERVER_DEPLOYMENT.md)。

## 📖 详细使用指南

### 分步训练流程

如果你想分步执行，可以按以下步骤：

#### 步骤 1：生成 Priors

使用 Project A 生成 likelihood 和 topology_mask：

```bash
# 方法 1：使用脚本
bash scripts/prepare_priors.sh

# 方法 2：手动调用
conda run -n pear-topo pear-topo infer-dataset \
  --config /path/to/Image_Segmentation/configs/pear_topology_4090.yaml \
  --data-yaml /home/robot/yolo/datasets/PearSurfaceDefects/li_data.yaml \
  --seg-ckpt /path/to/Image_Segmentation/outputs/checkpoints/unet_best.pt \
  --uq-ckpt /path/to/Image_Segmentation/outputs/checkpoints/uq_best.pt \
  --splits train val test \
  --out outputs/priors
```

**输出**：`outputs/priors/` 包含每张图的 likelihood.npy, topology_mask.npy, metadata.json

#### 步骤 2：构建 ROI 训练数据集

从 topology_mask 提取 ROI patches 并构建 YOLO 格式数据集：

```bash
python tools/build_roi_dataset.py --config configs/e1_config.yaml
```

**输出**：`outputs/roi_dataset/` 包含 YOLO 格式的 ROI patch 数据集

**数据集结构**：
```
outputs/roi_dataset/
├── images/
│   ├── train/
│   │   ├── cam0_1_557_pear_2_roi_0.jpg
│   │   └── ...
│   └── val/
├── labels/
│   ├── train/
│   │   ├── cam0_1_557_pear_2_roi_0.txt  # YOLO format
│   │   └── ...
│   └── val/
├── mapping.json  # patch_id -> 原图映射
└── data.yaml     # YOLO 数据集配置
```

#### 步骤 3：训练局部检测器

在 ROI patches 上训练新的 YOLO 模型：

```bash
python tools/train_e1_pipeline.py \
  --config configs/e1_config.yaml \
  --skip-priors \
  --skip-dataset
```

**输出**：`outputs/e1_models/local_detector/weights/best.pt`

### 推理选项

```bash
# 单张图像推理
python scripts/infer_e1_fusion.py \
  --config configs/e1_config.yaml \
  --source image.jpg \
  --visualize

# 批量推理
python scripts/infer_e1_fusion.py \
  --config configs/e1_config.yaml \
  --source test_images/ \
  --output outputs/e1_detections

# 指定 split（用于加载 priors）
python scripts/infer_e1_fusion.py \
  --config configs/e1_config.yaml \
  --source test_images/ \
  --split test
```

## 🏗️ 项目结构

```
pear-defect-detection-fusion/
├── configs/
│   ├── e1_config.yaml           # E1 实验配置
│   └── fusion_config.yaml       # 基础融合配置
├── fusion/
│   └── roi_proposal/
│       ├── roi_mapper.py        # 坐标转换工具
│       ├── roi_generator.py     # ROI 生成器
│       ├── roi_dataset_builder.py  # 数据集构建器
│       ├── roi_infer.py         # ROI 推理
│       └── roi_fusion.py        # 检测融合
├── src/
│   ├── priors_loader.py         # 先验数据加载器
│   └── roi_proposal.py          # ROI 提取（基础）
├── tools/
│   ├── build_roi_dataset.py     # 构建 ROI 数据集脚本
│   ├── train_e1_pipeline.py     # 统一训练流程
│   └── test_e1_modules.py       # 模块测试
├── scripts/
│   ├── infer_e1_fusion.py       # E1 推理脚本
│   └── prepare_priors.sh        # 生成 priors 脚本
├── outputs/                     # 输出目录
│   ├── priors/                  # 先验数据
│   ├── roi_dataset/             # ROI 训练数据集
│   ├── e1_models/               # 训练模型
│   └── e1_detections/           # 推理结果
├── requirements.txt             # Python 依赖
├── README.md                    # 本文档
├── QUICKSTART_E1.md            # 快速开始指南
├── E1_IMPLEMENTATION_REPORT.md # 实施报告
└── TRAINING_GUIDE.md           # 训练指南
```

## 🔧 配置说明

### 关键配置项

编辑 `configs/e1_config.yaml`：

```yaml
# ROI 生成参数
roi:
  min_area: 100          # 最小 ROI 面积
  max_area: 50000        # 最大 ROI 面积
  expansion_ratio: 0.2   # ROI 外扩比例（20%）
  morphology:
    kernel_size: 5       # 形态学核大小
    operation: closing   # 形态学操作

# ROI 数据集构建
roi_dataset:
  label_assignment:
    iou_threshold: 0.3        # IoU 阈值
    overlap_ratio: 0.5        # 重叠率阈值
  hard_negative:
    enabled: true
    likelihood_threshold: 0.5  # 困难负样本阈值
    max_ratio: 0.3            # 最大困难负样本比例

# 训练参数
training:
  epochs: 100
  imgsz: 640
  batch: 32
  device: '0'

# 融合策略
fusion:
  method: priority_nms    # priority_nms 或 confidence_nms
  iou_threshold: 0.5      # NMS IoU 阈值
  priority: local         # local, global, 或 confidence
```

## 📊 实验结果

### E1 实验设计

| 实验组 | 全局检测 | 局部检测 | 融合策略 | mAP50 | mAP50-95 |
|--------|----------|----------|----------|-------|----------|
| E0 (Baseline) | ✓ | ✗ | - | TBD | TBD |
| E1 | ✓ | ✓ | Priority NMS | TBD | TBD |

### 评估指标

- **主要指标**: mAP50（目标：提升 > 2%）
- **次要指标**: mAP50-95, Precision, Recall
- **按类别分析**: bruise (瘀伤), twig (划伤), rot (腐烂)

## 🧪 测试与验证

### 运行测试

```bash
# 测试所有模块
python tools/test_e1_modules.py

# 测试单个模块
python -c "
from fusion.roi_proposal.roi_mapper import compute_iou
print(compute_iou((0, 0, 100, 100), (50, 50, 150, 150)))
"
```

### 快速验证（小数据集）

```bash
# 1. 生成少量 priors（验证集 10 张图）
bash scripts/prepare_priors.sh --splits val --limit 10

# 2. 构建 ROI 数据集（只用验证集）
python tools/build_roi_dataset.py --config configs/e1_config.yaml --splits val

# 3. 快速训练测试（10 epochs）
# 编辑 configs/e1_config.yaml，设置 training.epochs: 10
python tools/train_e1_pipeline.py \
  --config configs/e1_config.yaml \
  --skip-priors \
  --skip-dataset
```

## 🐛 常见问题

### Q1: CUDA 不可用错误

**错误**: `RuntimeError: CUDA was requested but torch.cuda.is_available() is false.`

**解决**: 
- 在 Linux 服务器上，使用 `pear_topology_4090.yaml`（已配置）
- 在 Mac 上，配置会自动使用 `pear_topology.yaml`（自动设备选择）

### Q2: 模块导入失败

**错误**: `ModuleNotFoundError: No module named 'cv2'`

**解决**: 
```bash
pip install opencv-python pyyaml tqdm ultralytics
```

### Q3: Priors 不存在

**错误**: 推理时找不到 priors

**解决**: 先生成 priors
```bash
bash scripts/prepare_priors.sh
```

### Q4: 内存不足

**解决**: 
- 减小 batch size：编辑 `configs/e1_config.yaml`，设置 `training.batch: 16`
- 使用更小的图像尺寸：设置 `training.imgsz: 512`

## 📚 文档

- [QUICKSTART_E1.md](QUICKSTART_E1.md) - 快速开始指南
- [E1_IMPLEMENTATION_REPORT.md](E1_IMPLEMENTATION_REPORT.md) - 详细实施报告
- [TRAINING_GUIDE.md](TRAINING_GUIDE.md) - 训练流程说明
- [CLAUDE.md](CLAUDE.md) - Claude Code 工作指导

## 🤝 贡献

本项目为研究性质，欢迎提出问题和建议。

## 📄 License

本项目遵循源项目的许可协议，仅用于学术研究。

## 🙏 致谢

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) - YOLO26 实现
- [Topology-Aware Uncertainty for Image Segmentation](https://arxiv.org/abs/2306.05671) - 拓扑感知不确定性方法

## 📧 联系方式

如有问题，请通过 GitHub Issues 联系。

---

**开发状态**: ✅ E1 实验已完成实施，可以开始训练和评估
