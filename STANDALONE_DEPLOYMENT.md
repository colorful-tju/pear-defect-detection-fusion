# 独立部署指南（无需外部项目依赖）

本指南说明如何在服务器上部署和运行 E1 融合检测项目，**无需依赖外部项目代码**。

## 📋 前置准备

### 需要准备的文件

在服务器上运行本项目，你需要准备以下文件：

1. **数据集**（YOLO 格式）
   - 路径示例: `/home/robot/yolo/datasets/PearSurfaceDefects/`
   - 包含: `images/`, `labels/`, `data.yaml`

2. **Priors 数据**（预先生成）
   - 可以在任何有 Project A 的机器上生成
   - 包含: `likelihood.npy`, `topology_mask.npy`, `metadata.json`
   - 目录结构见下文

3. **预训练模型**
   - Global detector: YOLO baseline 模型 (`.pt` 文件)
   - 可选: YOLO 预训练权重 (`yolo26s.pt`)

### Priors 数据结构

Priors 数据应该有以下结构：

```
priors/
├── manifests/
│   ├── train.json
│   ├── val.json
│   └── test.json
├── train/
│   └── <image_key>/
│       ├── likelihood.npy
│       ├── topology_mask.npy
│       └── metadata.json
├── val/
└── test/
```

## 🚀 部署步骤

### 1. 克隆仓库

```bash
# 在服务器上
git clone git@github.com:colorful-tju/pear-defect-detection-fusion.git
cd pear-defect-detection-fusion
```

### 2. 创建环境

```bash
# 创建 conda 环境
conda create -n pear-fusion python=3.10
conda activate pear-fusion

# 安装 PyTorch (CUDA 11.8)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 安装其他依赖
pip install -r requirements.txt
```

### 3. 准备 Priors 数据

#### 方案 A：在其他机器上生成，然后传输

```bash
# 在有 Project A 的机器上生成 priors
cd /path/to/Image_Segmentation
conda activate pear-topo

pear-topo infer-dataset \
  --config configs/pear_topology_4090.yaml \
  --data-yaml /path/to/data.yaml \
  --seg-ckpt outputs/checkpoints/unet_best.pt \
  --uq-ckpt outputs/checkpoints/uq_best.pt \
  --splits train val test \
  --out /tmp/priors

# 传输到服务器
rsync -avz /tmp/priors/ \
  server:/path/to/pear-defect-detection-fusion/outputs/priors/
```

#### 方案 B：直接复制已有的 priors

```bash
# 如果已经有生成好的 priors
cp -r /path/to/existing/priors outputs/
```

### 4. 准备模型文件

```bash
# 创建 models 目录
mkdir -p models

# 复制或链接 global detector
cp /path/to/yolo26-pear/best.pt models/global_detector.pt

# 或创建软链接
ln -s /path/to/yolo26-pear/best.pt models/global_detector.pt

# 下载 YOLO 预训练权重（如果需要）
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo26s.pt
```

### 5. 配置路径

编辑 `configs/e1_config_standalone.yaml`：

```yaml
# 数据集路径
dataset:
  root: /home/robot/yolo/datasets/PearSurfaceDefects
  data_yaml: /home/robot/yolo/datasets/PearSurfaceDefects/li_data.yaml

# Priors 路径
priors:
  root_dir: outputs/priors

# 模型路径
models:
  global_detector: models/global_detector.pt
  local_detector: outputs/e1_models/local_detector/weights/best.pt

# 训练参数
training:
  epochs: 100
  batch: 32
  device: '0'
```

### 6. 验证环境

```bash
# 测试模块
python tools/test_e1_modules.py

# 检查 priors 是否存在
ls -lh outputs/priors/manifests/
cat outputs/priors/manifests/train.json | head -20
```

## 🎯 运行训练

### 完整训练流程

```bash
# 激活环境
conda activate pear-fusion

# 运行训练（使用独立配置）
python tools/train_e1_standalone.py --config configs/e1_config_standalone.yaml
```

这会自动完成：
1. ✅ 检查 priors 是否存在
2. ✅ 构建 ROI 训练数据集
3. ✅ 训练局部检测器

### 分步执行

```bash
# 步骤 1: 只构建 ROI 数据集
python tools/build_roi_dataset.py --config configs/e1_config_standalone.yaml

# 步骤 2: 只训练模型
python tools/train_e1_standalone.py \
  --config configs/e1_config_standalone.yaml \
  --skip-dataset
```

## 🔍 运行推理

```bash
# 测试集推理
python scripts/infer_e1_fusion.py \
  --config configs/e1_config_standalone.yaml \
  --source /home/robot/yolo/datasets/PearSurfaceDefects/images/test \
  --output outputs/e1_detections \
  --visualize

# 单张图像推理
python scripts/infer_e1_fusion.py \
  --config configs/e1_config_standalone.yaml \
  --source image.jpg \
  --visualize
```

## 📁 项目文件结构

部署后的完整结构：

```
pear-defect-detection-fusion/
├── configs/
│   └── e1_config_standalone.yaml    # 独立配置（推荐使用）
├── fusion/                           # 核心模块
├── tools/
│   └── train_e1_standalone.py       # 独立训练脚本（推荐使用）
├── scripts/
│   └── infer_e1_fusion.py           # 推理脚本
├── models/                           # 模型文件（需要准备）
│   ├── global_detector.pt           # Global detector
│   └── yolo26s.pt                   # YOLO 预训练权重
├── outputs/
│   ├── priors/                      # Priors 数据（需要准备）
│   │   ├── manifests/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── roi_dataset/                 # ROI 数据集（自动生成）
│   └── e1_models/                   # 训练输出（自动生成）
└── requirements.txt
```

## ✅ 检查清单

部署前确认：

- [ ] 已克隆仓库
- [ ] 已创建 conda 环境并安装依赖
- [ ] 数据集已准备好（YOLO 格式）
- [ ] Priors 数据已准备好（在 `outputs/priors/`）
- [ ] Global detector 模型已准备好（在 `models/`）
- [ ] 已配置 `configs/e1_config_standalone.yaml`
- [ ] 已运行 `python tools/test_e1_modules.py` 验证环境

## 🎓 最佳实践

### 1. 使用独立配置和脚本

推荐使用：
- 配置: `configs/e1_config_standalone.yaml`
- 训练: `tools/train_e1_standalone.py`

这些文件不依赖外部项目，只需要准备好数据文件即可。

### 2. Priors 数据管理

建议：
- 在一台机器上生成 priors，然后分发到多台服务器
- 使用 rsync 或 scp 传输 priors 数据
- 可以将 priors 数据打包成 tar.gz 便于传输

```bash
# 打包 priors
tar -czf priors.tar.gz outputs/priors/

# 传输到服务器
scp priors.tar.gz server:/path/to/fusion/

# 在服务器上解压
tar -xzf priors.tar.gz
```

### 3. 模型文件管理

建议：
- 将模型文件统一放在 `models/` 目录
- 使用软链接而不是复制，节省空间
- 记录模型版本和训练配置

### 4. 使用 tmux 进行长时间训练

```bash
# 创建 tmux 会话
tmux new -s e1-training

# 运行训练
python tools/train_e1_standalone.py --config configs/e1_config_standalone.yaml

# 分离会话: Ctrl+B, D
# 重新连接: tmux attach -t e1-training
```

## 🐛 故障排查

### 问题 1: Priors 不存在

**错误**: `ERROR: Priors not found!`

**解决**: 
1. 检查 `outputs/priors/manifests/` 是否存在
2. 检查 manifest 文件是否有内容
3. 按照上面的步骤生成或复制 priors

### 问题 2: 模型文件不存在

**错误**: `FileNotFoundError: models/global_detector.pt`

**解决**:
```bash
# 检查模型文件
ls -lh models/

# 复制或链接模型
cp /path/to/best.pt models/global_detector.pt
```

### 问题 3: CUDA 内存不足

**解决**:
```yaml
# 编辑 configs/e1_config_standalone.yaml
training:
  batch: 16  # 减小 batch size
  imgsz: 512  # 减小图像尺寸
```

## 📞 获取帮助

- **GitHub Issues**: https://github.com/colorful-tju/pear-defect-detection-fusion/issues
- **查看文档**: `README.md`, `SERVER_DEPLOYMENT.md`
- **运行测试**: `python tools/test_e1_modules.py`

---

**关键优势**: 
- ✅ 无需安装 Project A
- ✅ 无需 pear-topo 环境
- ✅ 只需要数据文件和模型文件
- ✅ 完全独立运行
