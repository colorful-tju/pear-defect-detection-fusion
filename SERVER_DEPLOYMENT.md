# Linux 服务器部署指南

本指南专门针对在 Linux 服务器（RTX 4090）上部署和运行 E1 融合检测项目。

## 📋 前置条件

### 硬件要求
- GPU: NVIDIA RTX 4090 或同等性能
- 内存: 32GB+ RAM
- 存储: 100GB+ 可用空间

### 软件要求
- OS: Ubuntu 20.04+ / CentOS 7+
- CUDA: 11.8+
- cuDNN: 8.6+
- Conda/Miniconda

### 依赖项目

确保以下项目已经准备好：

1. **Image Segmentation** - 已训练的分割模型
   - 路径: `/path/to/Image_Segmentation`
   - 模型: `outputs/checkpoints/unet_best.pt`, `uq_best.pt`
   - 环境: `pear-topo`

2. **YOLO26-pear** - 已训练的 baseline 模型
   - 路径: `/path/to/yolo26-pear`
   - 模型: `best.pt`

3. **数据集** - YOLO 格式数据集
   - 路径: `/home/robot/yolo/datasets/PearSurfaceDefects`
   - 结构: `images/`, `labels/`, `li_data.yaml`

## 🚀 部署步骤

### 1. 克隆仓库

```bash
# SSH 方式（推荐）
git clone git@github.com:colorful-tju/pear-defect-detection-fusion.git

# 或 HTTPS 方式
git clone https://github.com/colorful-tju/pear-defect-detection-fusion.git

cd pear-defect-detection-fusion
```

### 2. 创建 Conda 环境

```bash
# 创建环境
conda create -n pear-fusion python=3.10
conda activate pear-fusion

# 安装 PyTorch（CUDA 11.8）
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 安装其他依赖
pip install -r requirements.txt
```

### 3. 验证 CUDA

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

预期输出：
```
CUDA available: True
GPU: NVIDIA GeForce RTX 4090
```

### 4. 配置路径

编辑 `configs/e1_config.yaml`：

```bash
vim configs/e1_config.yaml
```

修改以下路径：

```yaml
# 项目路径（根据实际情况修改）
project_a_root: /home/robot/yolo/Image_Segmentation
project_a_config: pear_topology_4090.yaml  # Linux 使用 GPU 配置
project_b_root: /home/robot/yolo/yolo26-pear

# 数据集路径
dataset:
  root: /home/robot/yolo/datasets/PearSurfaceDefects
  data_yaml: /home/robot/yolo/datasets/PearSurfaceDefects/li_data.yaml

# 模型路径
models:
  global_detector: /home/robot/yolo/yolo26-pear/best.pt
  local_detector: outputs/e1_models/local_detector/weights/best.pt
```

### 5. 测试模块

```bash
# 测试所有模块
python tools/test_e1_modules.py
```

如果所有测试通过，说明环境配置正确。

## 🎯 运行训练

### 方案 1：一键训练（推荐）

```bash
# 激活环境
conda activate pear-fusion

# 运行完整训练流程
python tools/train_e1_pipeline.py --config configs/e1_config.yaml
```

这会自动完成：
1. 生成 priors（调用 pear-topo 环境）
2. 构建 ROI 训练数据集
3. 训练局部检测器

**预计时间**：
- Priors 生成: 2-4 小时（取决于数据集大小）
- ROI 数据集构建: 30-60 分钟
- 局部检测器训练: 4-8 小时（100 epochs）

### 方案 2：分步训练

#### 步骤 1：生成 Priors

```bash
# 确保 pear-topo 环境存在
conda activate pear-topo
conda list | grep torch  # 验证环境

# 返回 pear-fusion 环境
conda activate pear-fusion

# 生成 priors
bash scripts/prepare_priors.sh
```

**输出检查**：
```bash
ls -lh outputs/priors/manifests/
# 应该看到 train.json, val.json, test.json

# 检查生成的文件数量
cat outputs/priors/manifests/train.json | jq '. | length'
```

#### 步骤 2：构建 ROI 数据集

```bash
python tools/build_roi_dataset.py --config configs/e1_config.yaml
```

**输出检查**：
```bash
# 检查数据集结构
ls -lh outputs/roi_dataset/images/train/ | head
ls -lh outputs/roi_dataset/labels/train/ | head

# 检查 patch 数量
find outputs/roi_dataset/images/train/ -name "*.jpg" | wc -l
find outputs/roi_dataset/labels/train/ -name "*.txt" | wc -l

# 查看 mapping 文件
head outputs/roi_dataset/mapping.json
```

#### 步骤 3：训练局部检测器

```bash
python tools/train_e1_pipeline.py \
  --config configs/e1_config.yaml \
  --skip-priors \
  --skip-dataset
```

**训练监控**：
```bash
# 查看训练日志
tail -f outputs/e1_models/local_detector/train/results.csv

# 使用 TensorBoard（如果启用）
tensorboard --logdir outputs/e1_models/local_detector
```

## 🔍 运行推理

### 测试集推理

```bash
python scripts/infer_e1_fusion.py \
  --config configs/e1_config.yaml \
  --source /home/robot/yolo/datasets/PearSurfaceDefects/images/test \
  --output outputs/e1_detections \
  --visualize
```

### 单张图像推理

```bash
python scripts/infer_e1_fusion.py \
  --config configs/e1_config.yaml \
  --source /path/to/image.jpg \
  --visualize
```

### 查看结果

```bash
# 查看检测结果
cat outputs/e1_detections/e1_results.json | jq '.' | head -50

# 查看可视化
ls -lh outputs/e1_detections/visualizations/
```

## 📊 性能优化

### GPU 利用率监控

```bash
# 实时监控 GPU
watch -n 1 nvidia-smi

# 或使用 gpustat
pip install gpustat
gpustat -i 1
```

### 调整 Batch Size

如果遇到 OOM（内存不足）：

```bash
# 编辑配置文件
vim configs/e1_config.yaml

# 修改 batch size
training:
  batch: 16  # 从 32 降到 16
```

### 使用混合精度训练

配置文件中已默认启用 AMP（Automatic Mixed Precision）：

```yaml
training:
  amp: true  # 已启用
```

### 多 GPU 训练

如果有多个 GPU：

```yaml
training:
  device: '0,1,2,3'  # 使用 4 个 GPU
```

## 🐛 故障排查

### 问题 1：CUDA 不可用

```bash
# 检查 CUDA
nvcc --version
nvidia-smi

# 检查 PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# 重新安装 PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

### 问题 2：pear-topo 环境不存在

```bash
# 检查环境
conda env list

# 如果不存在，需要先设置 Image Segmentation 项目
cd /path/to/Image_Segmentation
conda env create -f environment-linux-cuda.yml
conda activate pear-topo
pip install -e .
```

### 问题 3：内存不足

```bash
# 减小 batch size
vim configs/e1_config.yaml
# training.batch: 16 或 8

# 减小图像尺寸
# training.imgsz: 512

# 减少 workers
# training.workers: 4
```

### 问题 4：训练速度慢

```bash
# 启用缓存
vim configs/e1_config.yaml
# training.cache: true

# 增加 workers
# training.workers: 16

# 使用 SSD 存储数据集
```

## 📈 监控和日志

### 训练日志

```bash
# 实时查看训练日志
tail -f outputs/e1.log

# 查看训练结果
cat outputs/e1_models/local_detector/train/results.csv
```

### 系统资源监控

```bash
# GPU 监控
nvidia-smi dmon -i 0 -s pucvmet

# CPU 和内存监控
htop

# 磁盘 I/O 监控
iotop
```

## 🔄 更新代码

```bash
# 拉取最新代码
git pull origin main

# 如果有依赖更新
pip install -r requirements.txt --upgrade
```

## 📦 备份和迁移

### 备份重要文件

```bash
# 备份训练模型
tar -czf e1_models_backup.tar.gz outputs/e1_models/

# 备份 priors
tar -czf priors_backup.tar.gz outputs/priors/

# 备份配置
tar -czf configs_backup.tar.gz configs/
```

### 迁移到其他服务器

```bash
# 在源服务器
tar -czf pear-fusion-full.tar.gz \
  pear-defect-detection-fusion/ \
  --exclude='outputs/priors' \
  --exclude='outputs/roi_dataset'

# 传输到目标服务器
scp pear-fusion-full.tar.gz user@target-server:/path/to/

# 在目标服务器
tar -xzf pear-fusion-full.tar.gz
cd pear-defect-detection-fusion
# 重新配置路径和环境
```

## 🎓 最佳实践

1. **使用 tmux/screen**：长时间训练建议使用 tmux 或 screen
   ```bash
   tmux new -s e1-training
   python tools/train_e1_pipeline.py --config configs/e1_config.yaml
   # Ctrl+B, D 分离会话
   ```

2. **定期保存检查点**：训练配置已自动保存最佳模型

3. **记录实验**：每次实验记录配置和结果
   ```bash
   cp configs/e1_config.yaml experiments/e1_exp_$(date +%Y%m%d_%H%M%S).yaml
   ```

4. **版本控制**：提交重要的配置更改
   ```bash
   git add configs/e1_config.yaml
   git commit -m "Update training config: batch size 32->16"
   git push
   ```

## 📞 获取帮助

如果遇到问题：

1. 查看文档：`README.md`, `TRAINING_GUIDE.md`
2. 运行测试：`python tools/test_e1_modules.py`
3. 查看日志：`outputs/e1.log`
4. 提交 Issue：https://github.com/colorful-tju/pear-defect-detection-fusion/issues

---

**部署完成后，你应该能够**：
- ✅ 成功运行模块测试
- ✅ 生成 priors
- ✅ 构建 ROI 数据集
- ✅ 训练局部检测器
- ✅ 运行 E1 推理
