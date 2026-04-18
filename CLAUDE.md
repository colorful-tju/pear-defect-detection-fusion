# CLAUDE.md

本文档为 Claude Code 提供梨表面缺陷检测融合项目的工作指导。

## 项目概述

本项目融合两个独立工程以提升梨表面缺陷（瘀伤和划伤）的目标检测精度（mAP50）：

- **项目 A (Image Segmentation)**: 梨表面缺陷分割与不确定性先验项目，基于拓扑感知不确定性验证（NeurIPS 2023），能够为每张图导出像素级软概率图（likelihood.npy）和拓扑遮罩（topology_mask.npy）
- **项目 B (yolo26-pear)**: 基于 YOLO26 的梨表面缺陷目标检测项目（baseline 版本）

## 工程原则

**核心约束：低侵入、可回退、可消融、可解释**

1. **不做联训**: 不联合训练 Image Segmentation 和 YOLO 模型
2. **不替换检测头**: 不修改 YOLO 检测头架构
3. **首发主线方案**（两件事）:
   - 用 A 的 `topology_mask.npy` 生成 ROI proposal 供 B 做高分辨率局部精检
   - 用 A 的 `likelihood.npy` 做检测框分数重加权
4. **明确命名**: 所有目录、路径、配置、脚本命名明确，不允许隐式依赖

## 源项目信息

### 项目 A: Image Segmentation

**位置**: `/Users/renxd/code/Image Segmentation`

**核心能力**:
- U-Net 分割主干 (Fθ) + GCN 拓扑不确定性模块 (Mϕ)
- 从 YOLO 检测框通过 GrabCut 生成伪遮罩进行训练
- 输出像素级 likelihood map 和拓扑结构遮罩

**关键输出文件**（用于 YOLO 集成）:
- `likelihood.npy`: [H, W] float32, 范围 [0.0, 1.0], 每个像素属于缺陷前景的概率
- `topology_mask.npy`: [H, W] uint8, 取值 0 或 1, 经拓扑结构筛选的最终缺陷区域
- `metadata.json`: 尺寸对齐信息（original_shape_hw, processed_shape_hw, resize_scale_hw）

**批量导出命令**:
```bash
pear-topo infer-dataset \
  --config configs/pear_topology_4090.yaml \
  --data-yaml /home/robot/yolo/datasets/PearSurfaceDefects/li_data.yaml \
  --seg-ckpt outputs/checkpoints/unet_best.pt \
  --uq-ckpt outputs/checkpoints/uq_best.pt \
  --splits train val test \
  --out outputs/yolo_priors
```

**输出目录结构**:
```
outputs/yolo_priors/
├── manifests/
│   ├── train.json
│   ├── val.json
│   └── test.json
├── summary.json
├── train/
│   └── <image_key>/
│       ├── likelihood.npy
│       ├── topology_mask.npy
│       └── metadata.json
├── val/
└── test/
```

**环境**:
- CLI 入口: `pear-topo`
- 配置文件: `configs/pear_topology_4090.yaml`
- 数据集格式: YOLO-style (与项目 B 共享)

### 项目 B: yolo26-pear

**位置**: `/Users/renxd/code/yolo26-pear`

**核心能力**:
- 基于 Ultralytics YOLO26 的目标检测
- 当前为 baseline 版本（已回退）
- 检测类别: bruise (瘀伤), twig (划伤), rot (腐烂)

**训练脚本**: `train.py`
```python
from ultralytics import YOLO
model = YOLO('yolo26s.pt')
results = model.train(
    data='/home/robot/yolo/datasets/PearSurfaceDefects/li_data.yaml',
    epochs=120,
    imgsz=640,
    device='0',
    batch=64,
    workers=8,
    project='PearSurfaceDefects',
    name='yolo26s',
    amp=True,
    cache=True
)
```

**数据集格式**:
```
PearSurfaceDefects/
├── li_data.yaml
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

**环境**:
- Python 包: `ultralytics`
- 模型: YOLO26s
- 设备: RTX 4090 (CUDA)

## 数据集共享

两个项目共享同一个 YOLO 格式数据集:
- 路径: `/home/robot/yolo/datasets/PearSurfaceDefects/`
- 配置文件: `li_data.yaml`
- 类别:
  - 0: pear (背景)
  - 1: bruise (瘀伤) - 前景
  - 2: twig (划伤) - 前景
  - 3: rot (腐烂) - 前景

## 融合方案设计

### 阶段 1: 先验数据准备（当前阶段）

**目标**: 为整个数据集生成 likelihood 和 topology_mask

**步骤**:
1. 使用项目 A 的训练好的模型批量推理
2. 为每张图生成 `likelihood.npy`, `topology_mask.npy`, `metadata.json`
3. 建立 `image_path -> priors_dir` 的映射索引

**输出位置**: `outputs/yolo_priors/` (在项目 A 中生成)

### 阶段 2: ROI 提取与局部精检

**目标**: 用 topology_mask 生成 ROI proposal，在 ROI 上跑高分辨率 YOLO

**实现方式**:
1. 从 `topology_mask.npy` 提取连通域
2. 对每个连通域外扩 10%-30% 生成 ROI patch
3. 在 ROI patch 上运行 YOLO 推理（更高分辨率或更小 stride）
4. 保留全图 YOLO 作为兜底
5. 合并全图和 ROI 检测结果（NMS 去重）

**关键文件**（待创建）:
- `src/roi_proposal.py`: 从 topology_mask 提取 ROI
- `src/roi_detector.py`: ROI 区域 YOLO 推理
- `src/merge_detections.py`: 合并全图和 ROI 检测结果

### 阶段 3: 检测框分数重加权

**目标**: 用 likelihood 对 YOLO 检测框分数进行重加权

**实现方式**:
1. 对每个检测框，计算框内 likelihood 的平均值或最大值
2. 重加权公式: `new_score = original_score * (alpha + beta * likelihood_score)`
3. 参数 alpha, beta 通过验证集调优

**关键文件**（待创建）:
- `src/score_reweight.py`: 检测框分数重加权逻辑

### 阶段 4: 消融实验与评估

**目标**: 验证各组件的增益来源

**实验组**:
1. Baseline: 纯 YOLO26s
2. +ROI: Baseline + ROI 局部精检
3. +Reweight: Baseline + 分数重加权
4. +Both: ROI + Reweight

**评估指标**:
- mAP50 (主要指标)
- mAP50-95
- Precision / Recall
- 按类别分析 (bruise, twig, rot)

## 目录结构规划

```
pear-defect-detection-fusion/
├── CLAUDE.md                    # 本文档
├── README.md                    # 项目说明
├── configs/
│   ├── fusion_config.yaml       # 融合方案配置
│   └── ablation_configs/        # 消融实验配置
├── src/
│   ├── __init__.py
│   ├── priors_loader.py         # 加载 likelihood 和 topology_mask
│   ├── roi_proposal.py          # ROI 提取
│   ├── roi_detector.py          # ROI 区域检测
│   ├── score_reweight.py        # 分数重加权
│   ├── merge_detections.py      # 检测结果合并
│   └── evaluator.py             # 评估工具
├── scripts/
│   ├── prepare_priors.sh        # 调用项目 A 生成先验
│   ├── train_fusion.py          # 融合训练脚本（如需要）
│   ├── infer_fusion.py          # 融合推理脚本
│   └── run_ablation.py          # 消融实验脚本
├── outputs/
│   ├── priors/                  # 软链接到项目 A 的输出
│   ├── detections/              # 检测结果
│   └── ablation_results/        # 消融实验结果
└── tests/
    ├── test_priors_loader.py
    ├── test_roi_proposal.py
    └── test_score_reweight.py
```

## 关键设计决策

### 1. 尺寸对齐策略

**问题**: `likelihood.npy` 可能与原图尺寸不一致（受 `inference.max_side` 限制）

**解决方案**:
- 读取 `metadata.json` 获取 `resize_scale_hw`
- 将 `likelihood` 和 `topology_mask` 插值回原图尺寸
- 或者调整项目 A 的 `inference.max_side` 使其不触发缩放

### 2. ROI 外扩策略

**参数**: 外扩比例 10%-30%（可调）

**原因**:
- topology_mask 可能略小于真实缺陷边界
- 外扩确保完整覆盖缺陷区域
- 过大外扩会引入过多背景，降低精度

### 3. 分数重加权公式

**初始方案**: `new_score = original_score * (0.5 + 0.5 * likelihood_avg)`

**调优空间**:
- likelihood 聚合方式: mean, max, weighted_mean
- 权重参数 alpha, beta
- 是否使用 uncertainty_map 作为折扣项

### 4. 检测结果合并策略

**NMS 参数**:
- IoU 阈值: 0.5-0.7（可调）
- 分数优先级: ROI 检测 > 全图检测（当 IoU 重叠时）

## 开发路线图

### Phase 1: 基础设施搭建（当前）
- [x] 理解两个源项目
- [ ] 创建融合项目目录结构
- [ ] 实现 `priors_loader.py`
- [ ] 编写 `prepare_priors.sh` 脚本

### Phase 2: ROI 分支实现
- [ ] 实现 `roi_proposal.py`
- [ ] 实现 `roi_detector.py`
- [ ] 实现 `merge_detections.py`
- [ ] 单元测试

### Phase 3: 分数重加权实现
- [ ] 实现 `score_reweight.py`
- [ ] 参数调优脚本
- [ ] 单元测试

### Phase 4: 集成与评估
- [ ] 实现 `infer_fusion.py`
- [ ] 实现 `evaluator.py`
- [ ] 在验证集上调优参数

### Phase 5: 消融实验
- [ ] 实现 `run_ablation.py`
- [ ] 运行完整消融实验
- [ ] 分析结果，撰写报告

## 重要注意事项

1. **不修改源项目**: 项目 A 和 B 保持独立，不在其代码库中做修改
2. **软链接管理**: 使用软链接引用源项目的输出，避免数据冗余
3. **配置文件驱动**: 所有路径、参数通过配置文件管理，便于切换实验
4. **版本控制**: 记录每次实验使用的模型 checkpoint 版本和配置
5. **可视化调试**: 保存 ROI 可视化、分数分布图等，便于调试

## 依赖环境

**Python 版本**: 3.8+

**核心依赖**:
- `ultralytics`: YOLO26 推理
- `torch`: 深度学习框架
- `numpy`: 数组操作
- `opencv-python`: 图像处理、连通域提取
- `pyyaml`: 配置文件解析
- `scikit-image`: 形态学操作

**安装**:
```bash
pip install ultralytics torch numpy opencv-python pyyaml scikit-image
```

## 参考文档

- 项目 A 文档: `/Users/renxd/code/Image Segmentation/CLAUDE.md`
- 项目 A 二次开发指南: `/Users/renxd/code/Image Segmentation/docs/SECONDARY_DEVELOPMENT.md`
- 项目 B 文档: `/Users/renxd/code/yolo26-pear/README.md`
- YOLO26 官方文档: https://docs.ultralytics.com/models/yolo26/

## 联系与支持

本项目为研究性质，如有问题请参考源项目文档或联系项目维护者。
