# E1 训练流程说明

## 问题诊断

你遇到的错误是因为 `pear_topology_4090.yaml` 配置文件指定了使用 CUDA，但 Mac 上没有 CUDA 支持。

## 解决方案

已经修改了以下文件来使用 `pear_topology.yaml`（自动设备选择）：

1. ✅ `configs/e1_config.yaml` - 添加了 `project_a_config: pear_topology.yaml`
2. ✅ `tools/train_e1_pipeline.py` - 使用配置中的 config 文件
3. ✅ `scripts/prepare_priors.sh` - 使用 `pear_topology.yaml`

## 推荐工作流程

### 选项 1：在服务器上生成 priors（推荐）

如果你有 Linux 服务器（带 GPU），建议在服务器上生成 priors：

```bash
# 在服务器上
cd /home/robot/yolo/pear-defect-detection-fusion
conda activate pear-topo

# 生成 priors（使用 GPU 配置）
pear-topo infer-dataset \
  --config /path/to/Image\ Segmentation/configs/pear_topology_4090.yaml \
  --data-yaml /home/robot/yolo/datasets/PearSurfaceDefects/li_data.yaml \
  --seg-ckpt /path/to/Image\ Segmentation/outputs/checkpoints/unet_best.pt \
  --uq-ckpt /path/to/Image\ Segmentation/outputs/checkpoints/uq_best.pt \
  --splits train val test \
  --out outputs/priors
```

然后将 `outputs/priors/` 目录同步到 Mac：

```bash
# 在 Mac 上
rsync -avz server:/home/robot/yolo/pear-defect-detection-fusion/outputs/priors/ \
  /Users/renxd/code/pear-defect-detection-fusion/outputs/priors/
```

然后跳过 priors 生成步骤：

```bash
# 在 Mac 上
python tools/train_e1_pipeline.py --config configs/e1_config.yaml --skip-priors
```

### 选项 2：在 Mac 上生成 priors（慢但可行）

如果你想在 Mac 上生成 priors（会使用 MPS 或 CPU，比较慢）：

```bash
# 在 Mac 上
python tools/train_e1_pipeline.py --config configs/e1_config.yaml
```

这会自动使用 `pear_topology.yaml`，它会自动选择可用设备（MPS 或 CPU）。

### 选项 3：只构建 ROI 数据集和训练（假设 priors 已存在）

如果你已经有 priors 了：

```bash
# 跳过 priors 生成
python tools/train_e1_pipeline.py --config configs/e1_config.yaml --skip-priors
```

## 快速测试（小数据集）

如果你想快速测试流程，可以先用少量数据：

```bash
# 1. 生成少量 priors（验证集的 10 张图）
bash scripts/prepare_priors.sh --splits val --limit 10

# 2. 构建 ROI 数据集（只用验证集）
python tools/build_roi_dataset.py --config configs/e1_config.yaml --splits val

# 3. 训练局部检测器（少量 epochs）
# 编辑 configs/e1_config.yaml，将 training.epochs 改为 10
python tools/train_e1_pipeline.py \
  --config configs/e1_config.yaml \
  --skip-priors \
  --skip-dataset
```

## 环境说明

- **pear-topo 环境**：用于运行 Project A（生成 priors）
- **yolo 环境**：用于 ROI 数据集构建、训练和推理

统一训练脚本会自动使用 `conda run -n pear-topo` 调用 Project A，无需手动切换环境。

## 下一步

现在你可以重新运行：

```bash
python tools/train_e1_pipeline.py --config configs/e1_config.yaml
```

如果还有问题，请告诉我具体的错误信息。
