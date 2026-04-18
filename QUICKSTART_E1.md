# E1 快速开始指南

## 环境准备

### 选项 1：使用现有环境（临时验证）

```bash
# 激活 YOLO 环境
conda activate yolo

# 安装额外依赖
pip install opencv-python pyyaml tqdm
```

### 选项 2：创建新环境（推荐）

```bash
# 创建环境
conda create -n pear-fusion python=3.10
conda activate pear-fusion

# 安装依赖
cd /Users/renxd/code/pear-defect-detection-fusion
pip install -r requirements.txt
```

## 快速验证

### 1. 测试模块

```bash
cd /Users/renxd/code/pear-defect-detection-fusion
python tools/test_e1_modules.py
```

### 2. 测试 ROI 生成（需要先有 priors）

```python
import yaml
from fusion.roi_proposal.roi_generator import ROIGenerator

# 加载配置
config = yaml.safe_load(open('configs/e1_config.yaml'))

# 初始化生成器
generator = ROIGenerator(config)

# 测试单张图像
test_image = "/home/robot/yolo/datasets/PearSurfaceDefects/images/val/cam0_1_557_pear_2.jpg"
patches = generator.generate_rois(test_image, split='val')

print(f'生成了 {len(patches)} 个 ROI patches')
for i, patch_info in enumerate(patches):
    print(f'  Patch {i}: coords={patch_info["roi_coords"]}, size={patch_info["patch_size"]}')
```

## 完整训练流程

### 一键训练（推荐）

```bash
# 运行完整流程：生成 priors → 构建数据集 → 训练局部检测器
python tools/train_e1_pipeline.py --config configs/e1_config.yaml
```

### 分步训练

#### 步骤 1：生成 Priors

```bash
# 使用 Project A 生成 priors
bash scripts/prepare_priors.sh

# 或指定 splits
bash scripts/prepare_priors.sh --splits train val
```

#### 步骤 2：构建 ROI 数据集

```bash
# 构建完整数据集
python tools/build_roi_dataset.py --config configs/e1_config.yaml

# 或只构建验证集（快速测试）
python tools/build_roi_dataset.py --config configs/e1_config.yaml --splits val
```

#### 步骤 3：训练局部检测器

```bash
# 跳过前两步，只训练
python tools/train_e1_pipeline.py \
  --config configs/e1_config.yaml \
  --skip-priors \
  --skip-dataset
```

## 推理

### 单张图像推理

```bash
python scripts/infer_e1_fusion.py \
  --config configs/e1_config.yaml \
  --source /path/to/image.jpg \
  --visualize
```

### 批量推理

```bash
python scripts/infer_e1_fusion.py \
  --config configs/e1_config.yaml \
  --source /home/robot/yolo/datasets/PearSurfaceDefects/images/test \
  --output outputs/e1_detections \
  --visualize
```

## 常见问题

### Q1: 模块导入失败（ModuleNotFoundError）

**A**: 安装缺失的依赖：
```bash
pip install opencv-python pyyaml tqdm ultralytics
```

### Q2: Priors 不存在

**A**: 先生成 priors：
```bash
bash scripts/prepare_priors.sh
```

### Q3: 局部检测器模型不存在

**A**: 先训练局部检测器：
```bash
python tools/train_e1_pipeline.py --config configs/e1_config.yaml
```

### Q4: 环境切换问题

**A**: 统一训练脚本会自动使用 `conda run -n pear-topo` 调用 Project A，无需手动切换。

## 输出目录结构

```
outputs/
├── priors/                      # Project A 生成的先验数据
│   ├── manifests/
│   ├── train/
│   ├── val/
│   └── test/
├── roi_dataset/                 # ROI 训练数据集
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   ├── labels/
│   │   ├── train/
│   │   └── val/
│   ├── mapping.json
│   └── data.yaml
├── e1_models/                   # 训练输出
│   └── local_detector/
│       └── weights/
│           └── best.pt          # 局部检测器最佳权重
└── e1_detections/               # 推理结果
    ├── e1_results.json
    └── visualizations/
```

## 配置调整

编辑 `configs/e1_config.yaml` 来调整参数：

```yaml
# ROI 生成
roi:
  expansion_ratio: 0.2    # 调整外扩比例（10%-30%）

# 融合策略
fusion:
  iou_threshold: 0.5      # 调整 NMS IoU 阈值
  priority: local         # local/global/confidence

# 训练
training:
  epochs: 100             # 训练轮数
  batch: 32               # 批大小
```

## 下一步

1. ✅ 验证模块导入
2. ✅ 生成 priors（如果还没有）
3. ✅ 构建 ROI 数据集
4. ✅ 训练局部检测器
5. ✅ 运行 E1 推理
6. ⏳ 评估结果（与 E0 对比）
7. ⏳ 消融实验
8. ⏳ 参数调优

## 获取帮助

- 查看详细报告：`E1_IMPLEMENTATION_REPORT.md`
- 查看计划文档：`/Users/renxd/.claude/plans/proud-herding-planet.md`
- 查看项目文档：`CLAUDE.md`, `README.md`
