# 项目状态报告

**项目名称**: 梨表面缺陷检测融合项目 (pear-defect-detection-fusion)  
**创建日期**: 2026-04-17  
**当前阶段**: Phase 1 - 基础设施搭建

---

## 项目概述

本项目融合**图像分割先验**（Image Segmentation）与 **YOLO26 目标检测**（yolo26-pear），旨在提升梨表面缺陷（瘀伤、划伤）的检测精度（mAP50）。

### 核心原则
- **低侵入**: 不修改源项目代码
- **可回退**: 保持 baseline 可运行
- **可消融**: 每个组件可独立评估
- **可解释**: 明确每个改进的来源

### 融合方案
1. **ROI 局部精检**: 用 topology_mask 生成 ROI proposal，在 ROI 上运行高分辨率 YOLO
2. **分数重加权**: 用 likelihood 对检测框分数进行重加权

---

## 已完成工作

### 1. 项目结构创建 ✅

```
pear-defect-detection-fusion/
├── CLAUDE.md                    # Claude Code 工作指导文档
├── README.md                    # 项目说明文档
├── PROJECT_STATUS.md            # 本文档
├── configs/
│   ├── fusion_config.yaml       # 融合方案配置文件
│   └── ablation_configs/        # 消融实验配置目录
├── src/
│   ├── __init__.py              # 包初始化
│   ├── priors_loader.py         # 先验数据加载器 ✅
│   └── roi_proposal.py          # ROI 提取模块 ✅
├── scripts/
│   └── prepare_priors.sh        # 先验数据生成脚本 ✅
├── outputs/
│   ├── priors/                  # 先验数据输出目录
│   ├── detections/              # 检测结果目录
│   └── ablation_results/        # 消融实验结果目录
└── tests/
    └── test_priors_loader.py    # 单元测试 ✅
```

### 2. 核心模块实现 ✅

#### `src/priors_loader.py`
- 加载 likelihood.npy 和 topology_mask.npy
- 处理尺寸对齐（通过 metadata.json）
- 支持缓存机制
- 批量加载接口

#### `src/roi_proposal.py`
- 从 topology_mask 提取连通域
- ROI 过滤（面积约束）
- ROI 外扩（可配置比例）
- ROI 可视化
- 重叠 ROI 过滤（NMS）

#### `scripts/prepare_priors.sh`
- 调用 Project A 批量生成先验数据
- 支持指定 splits 和 limit
- 自动验证输出结构

### 3. 配置文件 ✅

#### `configs/fusion_config.yaml`
完整的融合方案配置，包括：
- 项目路径配置
- 数据集配置
- 先验数据配置
- YOLO 配置
- ROI 提取配置
- 分数重加权配置
- 评估配置
- 消融实验配置
- 可视化配置

### 4. 文档 ✅

- **CLAUDE.md**: 详细的 Claude Code 工作指导，包含项目概述、源项目信息、融合方案设计、开发路线图等
- **README.md**: 用户友好的项目说明文档
- **PROJECT_STATUS.md**: 本状态报告

---

## 待完成工作

### Phase 2: ROI 分支实现

#### 2.1 ROI 检测器 (`src/roi_detector.py`)
- [ ] 实现 ROI 区域裁剪
- [ ] 在 ROI 上运行 YOLO 推理
- [ ] 坐标映射回原图
- [ ] 批量 ROI 检测

#### 2.2 检测结果合并 (`src/merge_detections.py`)
- [ ] 合并全图和 ROI 检测结果
- [ ] NMS 去重
- [ ] 优先级策略（ROI vs 全图）

#### 2.3 单元测试
- [ ] `tests/test_roi_proposal.py`
- [ ] `tests/test_roi_detector.py`
- [ ] `tests/test_merge_detections.py`

### Phase 3: 分数重加权实现

#### 3.1 分数重加权模块 (`src/score_reweight.py`)
- [ ] 实现检测框内 likelihood 聚合
- [ ] 分数重加权公式
- [ ] 参数调优接口
- [ ] 可选：uncertainty 折扣

#### 3.2 单元测试
- [ ] `tests/test_score_reweight.py`

### Phase 4: 集成与评估

#### 4.1 融合推理脚本 (`scripts/infer_fusion.py`)
- [ ] 加载配置
- [ ] 加载 YOLO 模型和先验数据
- [ ] 全图检测 + ROI 检测
- [ ] 分数重加权
- [ ] 结果保存

#### 4.2 评估工具 (`src/evaluator.py`)
- [ ] mAP 计算
- [ ] Precision/Recall 计算
- [ ] 按类别分析
- [ ] 结果可视化

#### 4.3 参数调优
- [ ] ROI 外扩比例调优
- [ ] 分数重加权参数调优
- [ ] NMS 阈值调优

### Phase 5: 消融实验

#### 5.1 消融实验脚本 (`scripts/run_ablation.py`)
- [ ] 实现 4 组实验（baseline, +ROI, +Reweight, +Both）
- [ ] 自动运行并记录结果
- [ ] 生成对比报告

#### 5.2 结果分析
- [ ] 各组件增益分析
- [ ] 按类别分析（bruise, twig, rot）
- [ ] 可视化对比
- [ ] 撰写技术报告

---

## 关键设计决策

### 1. 尺寸对齐策略
- 通过 `metadata.json` 获取 resize 信息
- 使用 OpenCV 插值回原图尺寸
- likelihood 用线性插值，topology_mask 用最近邻插值

### 2. ROI 外扩策略
- 默认外扩 20%（可配置）
- 确保完整覆盖缺陷边界
- 避免过大外扩引入过多背景

### 3. 分数重加权公式
- 初始方案: `new_score = original_score * (0.5 + 0.5 * likelihood_avg)`
- 可调参数: alpha, beta
- 可选 uncertainty 折扣

### 4. 检测结果合并策略
- NMS IoU 阈值: 0.5（可调）
- 优先级: ROI 检测 > 全图检测

---

## 依赖项目

### Project A: Image Segmentation
- **路径**: `/Users/renxd/code/Image Segmentation`
- **状态**: 已训练完成
- **输出**: likelihood.npy, topology_mask.npy, metadata.json
- **CLI**: `pear-topo`

### Project B: yolo26-pear
- **路径**: `/Users/renxd/code/yolo26-pear`
- **状态**: Baseline 版本（已回退）
- **模型**: `best.pt` (YOLO26s)
- **数据集**: `/home/robot/yolo/datasets/PearSurfaceDefects/`

---

## 下一步行动

### 立即行动（优先级 P0）
1. **生成先验数据**: 运行 `scripts/prepare_priors.sh` 为整个数据集生成先验
2. **实现 ROI 检测器**: 完成 `src/roi_detector.py`
3. **实现检测结果合并**: 完成 `src/merge_detections.py`

### 短期目标（1-2 周）
1. 完成 Phase 2（ROI 分支）
2. 完成 Phase 3（分数重加权）
3. 实现基础推理脚本

### 中期目标（3-4 周）
1. 完成 Phase 4（集成与评估）
2. 在验证集上调优参数
3. 完成消融实验

---

## 风险与挑战

### 技术风险
1. **尺寸对齐问题**: likelihood 与原图尺寸不一致可能导致精度损失
   - **缓解**: 调整 Project A 的 `inference.max_side` 避免缩放
   
2. **ROI 外扩比例**: 过小可能遗漏缺陷边界，过大引入背景噪声
   - **缓解**: 通过验证集调优，支持可配置

3. **检测结果合并**: NMS 可能误删有效检测
   - **缓解**: 调优 IoU 阈值，支持优先级策略

### 工程风险
1. **依赖项目变更**: Project A 或 B 的更新可能影响融合
   - **缓解**: 记录依赖版本，使用软链接管理

2. **计算资源**: ROI 检测增加推理时间
   - **缓解**: 批量处理，GPU 加速

---

## 评估指标

### 主要指标
- **mAP50**: 主要优化目标
- **mAP50-95**: 综合评估

### 辅助指标
- Precision / Recall
- F1 Score
- 按类别分析（bruise, twig, rot）
- 推理时间

### 消融实验对比
| 实验组 | mAP50 | mAP50-95 | Precision | Recall | 推理时间 |
|--------|-------|----------|-----------|--------|----------|
| Baseline | TBD | TBD | TBD | TBD | TBD |
| +ROI | TBD | TBD | TBD | TBD | TBD |
| +Reweight | TBD | TBD | TBD | TBD | TBD |
| +Both | TBD | TBD | TBD | TBD | TBD |

---

## 联系与支持

如有问题，请参考：
- [CLAUDE.md](CLAUDE.md): 详细技术文档
- [README.md](README.md): 快速开始指南
- Project A 文档: `/Users/renxd/code/Image Segmentation/CLAUDE.md`
- Project B 文档: `/Users/renxd/code/yolo26-pear/README.md`

---

**最后更新**: 2026-04-17  
**更新人**: Claude Code  
**版本**: v0.1.0
