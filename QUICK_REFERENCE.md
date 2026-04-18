# 快速参考指南

## 项目路径

```bash
# 融合项目
cd /Users/renxd/code/pear-defect-detection-fusion

# 项目 A (Image Segmentation)
cd /Users/renxd/code/Image\ Segmentation

# 项目 B (yolo26-pear)
cd /Users/renxd/code/yolo26-pear

# 数据集
cd /home/robot/yolo/datasets/PearSurfaceDefects
```

## 常用命令

### 1. 生成先验数据

```bash
# 完整数据集
bash scripts/prepare_priors.sh

# 指定 splits
bash scripts/prepare_priors.sh --splits val,test

# 限制数量（烟测）
bash scripts/prepare_priors.sh --splits val --limit 20

# 覆盖已有数据
bash scripts/prepare_priors.sh --overwrite
```

### 2. 运行测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定测试
pytest tests/test_priors_loader.py -v

# 运行测试并显示覆盖率
pytest tests/ --cov=src --cov-report=html
```

### 3. Python 使用示例

#### 加载先验数据

```python
from src import PriorsLoader

loader = PriorsLoader(priors_root="outputs/priors")
priors = loader.load_priors(
    image_path="/path/to/image.jpg",
    split="val",
    resize_to_original=True
)

likelihood = priors['likelihood']  # [H, W] float32
topology_mask = priors['topology_mask']  # [H, W] uint8
metadata = priors['metadata']  # dict
```

#### 提取 ROI

```python
from src import ROIProposer

proposer = ROIProposer(
    min_area=100,
    max_area=50000,
    expansion_ratio=0.2
)

rois = proposer.extract_rois(topology_mask, expand=True)

for roi in rois:
    x1, y1, x2, y2 = roi.to_xyxy()
    print(f"ROI: ({x1}, {y1}, {x2}, {y2}), area={roi.area}")
```

## 配置文件

### 主配置文件

```bash
configs/fusion_config.yaml
```

### 关键配置项

```yaml
# ROI 配置
roi:
  enabled: true
  expansion:
    value: 0.2  # 20% 外扩

# 分数重加权配置
score_reweight:
  enabled: true
  alpha: 0.5
  beta: 0.5
  aggregation: mean  # mean, max, weighted_mean

# YOLO 配置
yolo:
  model_path: /Users/renxd/code/yolo26-pear/best.pt
  imgsz: 640
  conf: 0.25
  device: '0'
```

## 目录结构

```
pear-defect-detection-fusion/
├── CLAUDE.md              # Claude Code 工作指导
├── README.md              # 项目说明
├── PROJECT_STATUS.md      # 项目状态
├── QUICK_REFERENCE.md     # 本文档
├── configs/
│   └── fusion_config.yaml
├── src/
│   ├── priors_loader.py   # 先验加载器
│   ├── roi_proposal.py    # ROI 提取
│   ├── roi_detector.py    # [待实现] ROI 检测
│   ├── score_reweight.py  # [待实现] 分数重加权
│   └── merge_detections.py # [待实现] 结果合并
├── scripts/
│   ├── prepare_priors.sh  # 生成先验
│   ├── infer_fusion.py    # [待实现] 融合推理
│   └── run_ablation.py    # [待实现] 消融实验
├── outputs/
│   ├── priors/            # 先验数据
│   ├── detections/        # 检测结果
│   └── ablation_results/  # 消融实验结果
└── tests/
    └── test_*.py          # 单元测试
```

## 数据格式

### likelihood.npy

```python
# 形状: [H, W]
# 类型: float32
# 范围: [0.0, 1.0]
# 含义: 每个像素属于缺陷前景的概率
```

### topology_mask.npy

```python
# 形状: [H, W]
# 类型: uint8
# 取值: 0 或 1
# 含义: 经拓扑筛选的最终缺陷区域
```

### metadata.json

```json
{
  "image": "/abs/path/to/image.jpg",
  "split": "val",
  "original_shape_hw": [1024, 1280],
  "processed_shape_hw": [819, 1024],
  "resized": true,
  "resize_scale_hw": [0.7998, 0.8]
}
```

## 开发工作流

### 1. 创建新功能

```bash
# 1. 在 src/ 下创建新模块
touch src/new_module.py

# 2. 实现功能
# 3. 在 src/__init__.py 中导出
# 4. 编写单元测试
touch tests/test_new_module.py

# 5. 运行测试
pytest tests/test_new_module.py -v
```

### 2. 调试流程

```bash
# 1. 生成少量先验数据
bash scripts/prepare_priors.sh --splits val --limit 10

# 2. 在 Python 中交互式调试
python -i
>>> from src import PriorsLoader, ROIProposer
>>> loader = PriorsLoader("outputs/priors")
>>> # 调试代码...

# 3. 可视化结果
# 使用 matplotlib 或 OpenCV 可视化
```

### 3. 实验流程

```bash
# 1. 修改配置文件
vim configs/fusion_config.yaml

# 2. 运行推理（待实现）
python scripts/infer_fusion.py --config configs/fusion_config.yaml

# 3. 评估结果（待实现）
python scripts/evaluate.py --predictions outputs/detections

# 4. 记录结果
# 在 PROJECT_STATUS.md 中更新实验结果
```

## 常见问题

### Q1: 先验数据尺寸与原图不一致？

**A**: 设置 `resize_to_original=True` 或调整 Project A 的 `inference.max_side`

```python
priors = loader.load_priors(image_path, resize_to_original=True)
```

### Q2: ROI 提取不到连通域？

**A**: 检查 `min_area` 和 `max_area` 设置，或可视化 topology_mask

```python
import cv2
cv2.imshow("mask", topology_mask * 255)
cv2.waitKey(0)
```

### Q3: 如何调整 ROI 外扩比例？

**A**: 修改配置文件中的 `roi.expansion.value`

```yaml
roi:
  expansion:
    value: 0.3  # 改为 30% 外扩
```

### Q4: 如何查看先验数据生成进度？

**A**: 查看 Project A 的输出日志或 manifests

```bash
cat outputs/priors/manifests/val.json | jq '. | length'
```

## 性能优化建议

### 1. 使用缓存

```python
# 启用缓存（默认）
loader = PriorsLoader(priors_root, use_cache=True)

# 批量加载
priors_list = loader.load_batch(image_paths, split="val")
```

### 2. 并行处理

```python
from multiprocessing import Pool

def process_image(image_path):
    # 处理单张图片
    pass

with Pool(8) as p:
    results = p.map(process_image, image_paths)
```

### 3. GPU 加速

```python
# 确保 YOLO 使用 GPU
yolo_config = {
    'device': '0',  # 使用 GPU 0
    'batch': 16,    # 批量推理
}
```

## 相关文档

- [CLAUDE.md](CLAUDE.md): 详细技术文档
- [README.md](README.md): 项目说明
- [PROJECT_STATUS.md](PROJECT_STATUS.md): 项目状态
- [Image Segmentation 二次开发指南](../Image%20Segmentation/docs/SECONDARY_DEVELOPMENT.md)
- [YOLO26 官方文档](https://docs.ultralytics.com/models/yolo26/)

## 联系方式

如有问题，请查阅上述文档或联系项目维护者。
