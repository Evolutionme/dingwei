# 3D Gaussian Splatting 局部精细化 — 详细使用文档 (v2)

## 目录

1. [快速开始](#1-快速开始)
2. [完整工作流程](#2-完整工作流程)
3. [命令行参数参考](#3-命令行参数参考)
4. [消融实验指南](#4-消融实验指南)
5. [输出文件说明](#5-输出文件说明)
6. [常见问题](#6-常见问题)
7. [与原始 3DGS 对比](#7-与原始-3dgs-对比)

---

## 1. 快速开始

### 1.1 环境依赖

在原始 3DGS 环境基础上，额外安装：

```bash
pip install scikit-learn
# opencv-python 已在原始环境中
```

### 1.2 三步完成局部精修

```bash
# 步骤 1: 原始 3DGS 全局训练（完全不变）
python train.py -s <数据集路径> -m <模型输出路径> --eval

# 步骤 2: 运行局部精修流水线
python refine.py -m <模型输出路径> -s <数据集路径>

# 步骤 3: 对比评估
python render.py -m <模型输出路径>                   # 渲染 baseline
python metrics.py -m <模型输出路径>                   # baseline 指标
python metrics_local.py -m <模型输出路径>              # 局部区域指标
```

### 1.3 验证原始逻辑未受影响

```bash
# 以下命令的行为与改进前完全一致
python train.py -s <数据集路径> -m <新模型路径> --eval
python render.py -m <新模型路径>
python metrics.py -m <新模型路径>
```

---

## 2. 完整工作流程

### 2.1 阶段 A：全局基线训练

使用原始 `train.py`，参数不变：

```bash
python train.py \
    -s /path/to/dataset \
    -m /path/to/output \
    --eval \
    --iterations 30000 \
    --test_iterations 7000 30000 \
    --save_iterations 7000 30000
```

训练完成后，`/path/to/output/point_cloud/iteration_30000/` 下会有 baseline 模型。

### 2.2 阶段 B + C：分析 + 局部精修

使用 `refine.py`：

```bash
python refine.py \
    -m /path/to/output \
    -s /path/to/dataset \
    --iteration 30000 \
    --refine_tag exp1
```

这会自动执行：
1. 加载 iteration 30000 的 baseline 模型
2. 对所有训练视角做误差分析（支持自适应阈值）
3. **多策略** 2D→3D 定位（射线三角化 + 深度反投影 + 贡献统计 + 梯度归因）
4. 运行区域感知局部优化（支持局部密度控制和 opacity 校准）
5. 保存精修模型和分析结果

### 2.3 选择定位策略

v2 支持四种可独立开关的定位策略：

```bash
# 仅使用射线三角化 (GS-LPM 方法)
python refine.py -m <model> -s <data> \
    --enable_ray_triangulation \
    --no_enable_depth_backproject --no_enable_contribution_stat --no_enable_gradient_attr

# 仅使用深度反投影 (CL-Splats 方法)
python refine.py -m <model> -s <data> \
    --enable_depth_backproject \
    --no_enable_ray_triangulation --no_enable_contribution_stat --no_enable_gradient_attr

# 全部启用 (默认，推荐)
python refine.py -m <model> -s <data>
```

### 2.4 仅运行分析（不优化）

```bash
python refine.py \
    -m /path/to/output \
    -s /path/to/dataset \
    --iteration 30000 \
    --skip_refine \
    --refine_tag analysis_only
```

用于先检查缺陷检测和区域划分是否合理。

### 2.5 跳过分析（使用已有区域）

```bash
python refine.py \
    -m /path/to/output \
    -s /path/to/dataset \
    --iteration 30000 \
    --skip_analysis \
    --refine_tag exp1
```

用于调整优化超参时避免重复分析。

### 2.6 启用局部密度控制和 opacity 校准

源自 GS-LPM 的局部密度增强策略：

```bash
python refine.py -m <model> -s <data> \
    --local_densify \
    --grad_ratio 0.5 \
    --calibrate_opacity \
    --refine_tag densify_exp
```

### 2.7 渲染精修结果

精修模型保存在 `point_cloud/iteration_refine_<tag>/`，可直接用原始 `render.py` 渲染。

但需注意：原始 `render.py` 的 `--iteration` 参数期望数字。要渲染精修模型，可手动加载或将精修模型拷贝/软链到标准目录名。最便捷的方式是使用 `metrics_local.py` 直接评估。

### 2.8 区域感知评估

```bash
python metrics_local.py \
    -m /path/to/output \
    --refine_tag exp1
```

输出三组指标：
- **全图指标**：与原始 `metrics.py` 一致
- **缺陷区域指标**：仅在检测到的缺陷区域内计算
- **非缺陷区域指标**：验证非目标区域未退化

---

## 3. 命令行参数参考

### 3.1 `refine.py` 基础参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `-m, --model_path` | str | 必需 | 已训练模型路径 |
| `-s, --source_path` | str | 必需 | 数据集路径 |
| `--iteration` | int | -1 | 加载的 baseline 迭代（-1=最新） |
| `--quiet` | flag | False | 静默模式 |
| `--refine_tag` | str | "refine" | 精修运行标签（用于区分不同实验） |
| `--skip_analysis` | flag | False | 跳过分析阶段，加载已有区域 |
| `--skip_refine` | flag | False | 仅分析，不优化 |
| `--skip_eval` | flag | False | 跳过精修后评估 |

### 3.2 误差分析参数 (`ErrorAnalysisParams`)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--w_rgb` | float | 1.0 | L1 光度误差权重 |
| `--w_ssim` | float | 0.5 | 1-SSIM 结构误差权重 |
| `--w_edge` | float | 0.3 | 边缘梯度差异权重 |
| `--w_depth` | float | 0.0 | 深度误差权重（0=禁用） |
| `--w_lpips` | float | 0.0 | 感知误差权重（0=禁用，计算较慢） |
| `--error_percentile` | float | 90.0 | 缺陷像素百分位阈值 |
| `--error_abs_threshold` | float | 0.0 | 绝对误差阈值（0=仅用百分位） |
| `--min_defect_area` | int | 64 | 最小连通域面积（像素） |
| `--mask_dilate_radius` | int | 5 | 形态学膨胀半径 |
| `--min_view_hits` | int | 3 | 缺陷需出现的最少视角数 |
| `--use_adaptive_threshold` | flag | False | **v2新增** 使用 GS-LPM 式分块自适应阈值 |
| `--adaptive_patch_size` | int | 16 | **v2新增** 自适应阈值的 patch 大小 |
| `--adaptive_fill_ratio` | float | 0.5 | **v2新增** patch 中缺陷像素占比阈值 |

### 3.3 定位参数 (`LocalizationParams`)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--proj_overlap_thresh` | float | 0.3 | 投影重叠比例阈值 |
| `--depth_tolerance` | float | 0.1 | 深度一致性相对容差 |
| `--attr_w_xyz` | float | 1.0 | 位置梯度归因权重 |
| `--attr_w_feat` | float | 0.5 | 特征梯度归因权重 |
| `--attr_w_opacity` | float | 0.3 | 不透明度梯度归因权重 |
| `--attr_w_scale` | float | 0.3 | 缩放梯度归因权重 |
| `--attr_w_rotation` | float | 0.2 | 旋转梯度归因权重 |
| `--vote_min_views` | int | 2 | 多视角投票最少视角数 |
| `--vote_score_percentile` | float | 80.0 | 归因分数百分位阈值 |
| `--cluster_eps` | float | 0.05 | DBSCAN 聚类半径（场景比例） |
| `--cluster_min_samples` | int | 5 | DBSCAN 最小样本数 |
| `--context_expand_ratio` | float | 0.1 | 上下文环带扩张比例 |
| `--remove_isolated` | flag | True | 移除单视角孤立点 |
| `--ray_pair_angle_min` | float | 15.0 | **v2新增** 配对视角最小夹角 (度) |
| `--ray_pair_angle_max` | float | 60.0 | **v2新增** 配对视角最大夹角 (度) |
| `--depth_knn_k` | int | 3 | **v2新增** 深度反投影 kNN 近邻数 |
| `--depth_local_radius` | float | 3.0 | **v2新增** kNN 尺度感知半径倍率 |
| `--contribution_min_overlap` | float | 0.1 | **v2新增** 投影贡献最小重叠面积比 |

### 3.4 精修参数 (`RefinementParams`)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--refine_iterations` | int | 5000 | 局部精修迭代次数 |
| `--target_lr_multiplier` | float | 1.0 | 目标区域学习率倍率 |
| `--context_lr_multiplier` | float | 0.1 | 上下文区域学习率倍率 |
| `--protect_lr_multiplier` | float | 0.0 | 保护区域学习率倍率（0=冻结） |
| `--update_xyz` | flag | True | 目标区更新位置 |
| `--update_features` | flag | True | 目标区更新 SH 特征 |
| `--update_opacity` | flag | True | 目标区更新不透明度 |
| `--update_scaling` | flag | True | 目标区更新缩放 |
| `--update_rotation` | flag | True | 目标区更新旋转 |
| `--ctx_update_xyz` | flag | False | 上下文区更新位置 |
| `--ctx_update_features` | flag | True | 上下文区更新特征 |
| `--ctx_update_opacity` | flag | True | 上下文区更新不透明度 |
| `--ctx_update_scaling` | flag | False | 上下文区更新缩放 |
| `--ctx_update_rotation` | flag | False | 上下文区更新旋转 |
| `--local_densify` | flag | False | 启用目标区局部密度控制 |
| `--local_prune` | flag | False | 启用目标区局部裁剪 |
| `--local_densify_from_iter` | int | 100 | 局部 densify 开始迭代 |
| `--local_densify_until_iter` | int | 3000 | 局部 densify 结束迭代 |
| `--local_densify_interval` | int | 100 | 局部 densify 频率 |
| `--local_densify_grad_threshold` | float | 0.0002 | 局部 densify 梯度阈值 |
| `--grad_ratio` | float | 0.5 | **v2新增** 目标区梯度阈值缩放 (源自 GS-LPM) |
| `--calibrate_opacity` | flag | False | **v2新增** 启用目标区 opacity 校准 (源自 GS-LPM) |
| `--calibrate_top_ratio` | float | 0.5 | **v2新增** opacity 校准重置比例 |
| `--lambda_local_rgb` | float | 1.0 | 局部 L1 损失权重 |
| `--lambda_local_ssim` | float | 0.2 | 局部 SSIM 损失权重 |
| `--lambda_anchor` | float | 0.1 | 锚定正则化权重 |
| `--lambda_context` | float | 0.05 | 上下文一致性权重 |
| `--lambda_boundary` | float | 0.0 | **v2新增** 边界约束权重 (源自 CL-Splats) |
| `--protect_mode` | str | "hard" | 保护模式: "hard"=冻结, "soft"=锚定 |
| `--anchor_param_weight` | float | 10.0 | soft 模式下锚定强度 |
| `--view_sample_strategy` | str | "defect" | 视角采样: "defect"=优先缺陷视角 |
| `--defect_view_weight` | float | 3.0 | 缺陷视角采样权重提升 |
| `--prune_hysteresis` | int | 0 | **v2新增** 滞后剪枝计数 (源自 CL-Splats) |

### 3.5 消融控制参数 (`AblationParams`)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--enable_ray_triangulation` | flag | True | **v2新增** 启用射线三角化定位 (GS-LPM) |
| `--enable_depth_backproject` | flag | True | **v2新增** 启用深度反投影定位 (CL-Splats) |
| `--enable_contribution_stat` | flag | True | **v2新增** 启用投影贡献统计 (FlashSplat) |
| `--enable_gradient_attr` | flag | True | **v2新增** 启用梯度归因定位 (v1) |

### 3.6 `metrics_local.py` 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `-m, --model_paths` | str[] | 必需 | 模型路径列表 |
| `--refine_tag` | str | "refine" | 匹配的精修运行标签 |

---

## 4. 消融实验指南

### 4.1 缺陷检测消融

```bash
# EA-1: 仅 RGB
python refine.py -m <model> -s <data> --skip_refine \
    --w_rgb 1.0 --w_ssim 0.0 --w_edge 0.0 \
    --refine_tag ea1_rgb_only

# EA-2: RGB + SSIM
python refine.py -m <model> -s <data> --skip_refine \
    --w_rgb 1.0 --w_ssim 0.5 --w_edge 0.0 \
    --refine_tag ea2_rgb_ssim

# EA-3: RGB + SSIM + Edge（默认配置）
python refine.py -m <model> -s <data> --skip_refine \
    --refine_tag ea3_full

# EA-6: 自适应阈值 vs 固定百分位
python refine.py -m <model> -s <data> --skip_refine \
    --use_adaptive_threshold --adaptive_patch_size 16 \
    --refine_tag ea6_adaptive

# EA-7: 不同 patch 大小
python refine.py -m <model> -s <data> --skip_refine \
    --use_adaptive_threshold --adaptive_patch_size 8 \
    --refine_tag ea7_patch8
python refine.py -m <model> -s <data> --skip_refine \
    --use_adaptive_threshold --adaptive_patch_size 32 \
    --refine_tag ea7_patch32
```

### 4.2 定位策略消融

```bash
# LOC-1: 仅射线三角化 (GS-LPM 方法)
python refine.py -m <model> -s <data> \
    --enable_ray_triangulation \
    --no_enable_depth_backproject --no_enable_contribution_stat --no_enable_gradient_attr \
    --refine_tag loc1_ray_only

# LOC-2: 仅深度反投影 (CL-Splats 方法)
python refine.py -m <model> -s <data> \
    --enable_depth_backproject \
    --no_enable_ray_triangulation --no_enable_contribution_stat --no_enable_gradient_attr \
    --refine_tag loc2_depth_only

# LOC-3: 仅贡献统计 (FlashSplat 思想)
python refine.py -m <model> -s <data> \
    --enable_contribution_stat \
    --no_enable_ray_triangulation --no_enable_depth_backproject --no_enable_gradient_attr \
    --refine_tag loc3_contrib_only

# LOC-4: 仅梯度归因 (v1 方法)
python refine.py -m <model> -s <data> \
    --enable_gradient_attr \
    --no_enable_ray_triangulation --no_enable_depth_backproject --no_enable_contribution_stat \
    --refine_tag loc4_grad_only

# LOC-5: 射线 + 深度 (两种几何方法)
python refine.py -m <model> -s <data> \
    --enable_ray_triangulation --enable_depth_backproject \
    --no_enable_contribution_stat --no_enable_gradient_attr \
    --refine_tag loc5_geometry

# LOC-6: 全部策略 (默认)
python refine.py -m <model> -s <data> --refine_tag loc6_all

# LOC-9: 配对角度敏感性
python refine.py -m <model> -s <data> \
    --ray_pair_angle_min 30 --ray_pair_angle_max 90 \
    --refine_tag loc9_wide_angle
```

### 4.3 局部优化消融

```bash
# REF-3: 默认 hard freeze
python refine.py -m <model> -s <data> --refine_tag ref3_hard_freeze

# REF-4: soft anchor
python refine.py -m <model> -s <data> \
    --protect_mode soft --protect_lr_multiplier 0.01 \
    --lambda_anchor 1.0 --refine_tag ref4_soft_anchor

# REF-5: + GS-LPM 式局部 densify
python refine.py -m <model> -s <data> \
    --local_densify --grad_ratio 0.5 \
    --refine_tag ref5_densify

# REF-6: + opacity 校准
python refine.py -m <model> -s <data> \
    --local_densify --grad_ratio 0.5 --calibrate_opacity \
    --refine_tag ref6_calibrate

# REF-7: + CL-Splats 式边界约束
python refine.py -m <model> -s <data> \
    --lambda_boundary 0.1 \
    --refine_tag ref7_boundary

# REF-11: 梯度阈值缩放
python refine.py -m <model> -s <data> \
    --local_densify --grad_ratio 0.3 --refine_tag ref11_gr03
python refine.py -m <model> -s <data> \
    --local_densify --grad_ratio 0.7 --refine_tag ref11_gr07
```

### 4.4 结果收集

```bash
for tag in loc1_ray_only loc2_depth_only loc3_contrib_only loc4_grad_only loc6_all \
           ref3_hard_freeze ref5_densify ref6_calibrate ref7_boundary; do
    python metrics_local.py -m /path/to/output --refine_tag $tag
done
```

结果保存在 `results_local_<tag>.json` 中，格式：

```json
{
  "method_name": {
    "full": {"PSNR": 25.5, "SSIM": 0.85, "LPIPS": 0.12},
    "defect_region": {"PSNR": 22.3, "SSIM": 0.78, "LPIPS": 0.18},
    "clean_region": {"PSNR": 27.1, "SSIM": 0.89, "LPIPS": 0.09},
    "region_stats": {"target": 12345, "context": 5678, "protect": 380000}
  }
}
```

---

## 5. 输出文件说明

### 5.1 分析输出 (`analysis/<tag>/`)

| 文件 | 说明 |
|------|------|
| `error_maps/00000.png` ... | 每视角归一化误差热力图（灰度，亮=误差高） |
| `defect_masks/00000.png` ... | 每视角二值缺陷掩码（白=缺陷区域） |
| `defect_regions/00000.json` ... | 每视角缺陷连通域 bbox 列表 |
| `defect_views.json` | 有显著缺陷的视角索引列表 |

### 5.2 区域输出 (`regions/<tag>/`)

| 文件 | 说明 |
|------|------|
| `target_mask.pt` | `[N]` bool tensor，目标区域 Gaussian |
| `context_mask.pt` | `[N]` bool tensor，上下文环带 Gaussian |
| `protect_mask.pt` | `[N]` bool tensor，保护区域 Gaussian |
| `scores.pt` | `[N]` float tensor，每个 Gaussian 的融合归因分数 |
| `zone_bboxes.json` | 3D zone bounding boxes (射线三角化结果) |
| `region_meta.json` | 区域统计信息 |

### 5.3 精修模型

| 文件 | 说明 |
|------|------|
| `point_cloud/iteration_refine_<tag>/point_cloud.ply` | 精修后 Gaussian 模型（标准 PLY 格式） |

### 5.4 配置与日志

| 文件 | 说明 |
|------|------|
| `refine_config_<tag>.json` | 完整精修配置记录 |
| `refine_logs_<tag>/` | Tensorboard 训练日志 |
| `results_local_<tag>.json` | 区域感知评估结果 |

### 5.5 查看 Tensorboard 日志

```bash
tensorboard --logdir /path/to/output/refine_logs_<tag>
```

监控指标：
- `refine/total_loss`：总损失
- `refine/local_loss`：局部区域损失
- `refine/anchor_loss`：锚定正则化损失
- `refine/context_loss`：上下文一致性损失
- `refine/boundary_loss`：边界约束损失 (v2)

---

## 6. 常见问题

### Q1: "No significant defects found" 怎么办？

**原因**：baseline 模型质量已经很好，或阈值设得太高。

**解决**：
```bash
# 降低百分位阈值
python refine.py ... --error_percentile 80.0

# 使用自适应阈值 (GS-LPM 式)
python refine.py ... --use_adaptive_threshold --adaptive_patch_size 16

# 降低最小区域面积
python refine.py ... --min_defect_area 16

# 增加误差检测灵敏度
python refine.py ... --w_ssim 1.0 --w_edge 0.5
```

### Q2: 精修后非缺陷区域指标下降了？

**原因**：保护不够严格，或上下文区域太大。

**解决**：
```bash
# 确保 hard freeze
python refine.py ... --protect_mode hard --protect_lr_multiplier 0.0

# 缩小上下文环带
python refine.py ... --context_expand_ratio 0.05 --context_lr_multiplier 0.05

# 增强锚定
python refine.py ... --lambda_anchor 0.5

# 增加边界约束 (CL-Splats 式)
python refine.py ... --lambda_boundary 0.1
```

### Q3: 精修后缺陷区域改善不明显？

**原因**：迭代次数不够，或目标区域覆盖不完整。

**解决**：
```bash
# 增加精修迭代
python refine.py ... --refine_iterations 10000

# 降低定位阈值
python refine.py ... --vote_score_percentile 70.0 --vote_min_views 1

# 启用局部 densification (GS-LPM 式)
python refine.py ... --local_densify --grad_ratio 0.5

# 启用 opacity 校准 (GS-LPM 式)
python refine.py ... --calibrate_opacity

# 增加目标区学习率
python refine.py ... --target_lr_multiplier 2.0
```

### Q4: 精修时显存不足？

**解决**：
- 禁用梯度归因（最消耗显存）：`--no_enable_gradient_attr`
- 仅使用射线三角化 + 深度反投影（轻量级）
- 使用 `--skip_analysis` 复用已有区域划分
- 减少 `--refine_iterations`

### Q5: 定位阶段运行很慢？

v2 已大幅优化：
- 投影计算完全向量化（无逐点循环）
- DBSCAN 仅对候选子集进行
- 深度反投影使用批量 kNN

如果仍然慢，可禁用部分策略：
```bash
# 仅用最快的深度反投影
python refine.py ... --enable_depth_backproject \
    --no_enable_ray_triangulation --no_enable_contribution_stat --no_enable_gradient_attr
```

### Q6: 如何只对特定场景部分做精修？

可以手动编辑区域 mask：

```python
import torch

# 加载现有 mask
target = torch.load("regions/refine/target_mask.pt")
# 自定义修改...
torch.save(target, "regions/refine/target_mask.pt")

# 然后跳过分析直接精修
# python refine.py ... --skip_analysis
```

### Q7: 如何与原始 baseline 做公平对比？

```bash
# 1. 训练 baseline
python train.py -s <data> -m <model> --eval

# 2. 渲染 baseline
python render.py -m <model> --iteration 30000

# 3. baseline 指标
python metrics.py -m <model>

# 4. 精修
python refine.py -m <model> -s <data> --refine_tag v1

# 5. 区域感知对比
python metrics_local.py -m <model> --refine_tag v1
```

`metrics_local.py` 会同时评估 `test/` 下所有 method（包括 baseline 的 `ours_30000` 和精修的结果），使用同一套缺陷掩码，确保公平对比。

---

## 7. 与原始 3DGS 对比

### 7.1 代码层面

| 维度 | 原始 3DGS | 本方案 v2 |
|------|-----------|-----------|
| 训练入口 | `train.py` | `train.py`（不变）+ `refine.py`（新增） |
| 渲染入口 | `render.py` | `render.py`（不变） |
| 评估入口 | `metrics.py` | `metrics.py`（不变）+ `metrics_local.py`（新增） |
| Gaussian 模型 | `GaussianModel` | `GaussianModel`（不变）+ `LocalGaussianModel`（继承） |
| 渲染函数 | `render()` | `render()`（不变）+ `render_with_local_loss()`（包装） |
| 优化范围 | 全局 | 全局（阶段A）→ 局部（阶段C） |
| 密度控制 | 全局 densify/prune | 全局（阶段A）→ 局部 densify + opacity 校准（阶段C） |
| 2D→3D 定位 | 无 | 射线三角化 + 深度反投影 + 贡献统计 + 梯度归因 |
| 区域意识 | 无 | 三区域（目标/上下文/保护）+ 梯度掩码 |

### 7.2 效果预期

| 指标 | 预期变化 |
|------|---------|
| 全图 PSNR | 轻微提升或持平 |
| 全图 SSIM | 轻微提升或持平 |
| 全图 LPIPS | 轻微下降（改善） |
| 缺陷区域 PSNR | **显著提升** |
| 缺陷区域 SSIM | **显著提升** |
| 非缺陷区域指标 | **保持不变**（验证保护有效性） |

### 7.3 计算开销

| 阶段 | 额外耗时 | 说明 |
|------|---------|------|
| 分析 (B1) | ~1-2 分钟 | 渲染所有训练视角 + 误差计算 |
| 定位 (B2-B4) | ~2-5 分钟 | 多策略定位 + 聚类 |
| 精修 (C) | ~5-15 分钟 | 取决于 `refine_iterations` |
| **总计额外** | **~10-20 分钟** | 在 30k iter baseline 训练（~30 min）基础上 |

---

## 附录 A：模块架构图 (v2)

```
refine.py (主入口)
    │
    ├── arguments/refine_args.py
    │   ├── ErrorAnalysisParams
    │   ├── LocalizationParams
    │   ├── RefinementParams
    │   └── AblationParams (v2)
    │
    ├── 阶段 A: load_baseline()
    │   └── scene/gaussian_model_local.py → LocalGaussianModel
    │
    ├── 阶段 B: run_analysis_and_localization()
    │   ├── utils/error_analysis.py
    │   │   ├── compute_composite_error_map()
    │   │   ├── compute_adaptive_error_map() ← GS-LPM
    │   │   ├── extract_defect_mask()
    │   │   ├── extract_defect_regions() ← GS-LPM
    │   │   └── analyze_all_views()
    │   │
    │   ├── utils/localization.py (核心重写)
    │   │   ├── compute_camera_rays()            ← GS-LPM
    │   │   ├── find_paired_views()              ← GS-LPM
    │   │   ├── triangulate_3d_zones()           ← GS-LPM
    │   │   ├── find_gaussians_in_zones()        ← GS-LPM
    │   │   ├── depth_backproject_to_gaussians() ← CL-Splats
    │   │   ├── compute_contribution_scores()    ← FlashSplat
    │   │   ├── compute_gradient_attribution()   (v1 修复)
    │   │   ├── multiview_fusion()               (新增)
    │   │   ├── cluster_and_expand()             (v1 优化)
    │   │   └── run_full_localization()          (v2 重写)
    │   │
    │   └── utils/region_utils.py
    │       └── RegionManager (修复)
    │
    ├── 阶段 C: run_local_refinement()
    │   ├── gaussian_renderer/render_analysis.py
    │   │   └── render_with_local_loss() (修复 API)
    │   │
    │   └── scene/gaussian_model_local.py
    │       ├── apply_gradient_mask()          (CL-Splats 广播模式)
    │       ├── compute_anchor_loss()
    │       ├── compute_boundary_loss()        ← CL-Splats
    │       ├── calibrate_target_opacity()     ← GS-LPM
    │       └── local_densify_and_prune()      (GS-LPM grad_ratio)
    │
    └── 评估: refine_eval()

metrics_local.py (区域评估)
    ├── compute_masked_metrics()
    └── evaluate_local()
```

## 附录 B：推荐超参配置

### 室外场景（MipNeRF360 outdoor）

```bash
python refine.py -m <model> -s <data> \
    --w_rgb 1.0 --w_ssim 0.5 --w_edge 0.3 \
    --use_adaptive_threshold \
    --error_percentile 90.0 --min_defect_area 64 \
    --cluster_eps 0.05 --context_expand_ratio 0.1 \
    --refine_iterations 5000 \
    --protect_mode hard \
    --local_densify --grad_ratio 0.5 \
    --refine_tag outdoor_default
```

### 室内场景（MipNeRF360 indoor）

```bash
python refine.py -m <model> -s <data> \
    --w_rgb 1.0 --w_ssim 0.8 --w_edge 0.5 \
    --use_adaptive_threshold --adaptive_patch_size 12 \
    --error_percentile 85.0 --min_defect_area 32 \
    --cluster_eps 0.03 --context_expand_ratio 0.15 \
    --refine_iterations 8000 \
    --protect_mode hard \
    --local_densify --grad_ratio 0.5 --calibrate_opacity \
    --refine_tag indoor_default
```

### 合成场景（NeRF Synthetic）

```bash
python refine.py -m <model> -s <data> \
    --w_rgb 1.0 --w_ssim 0.3 --w_edge 0.2 \
    --error_percentile 92.0 --min_defect_area 48 \
    --cluster_eps 0.04 --context_expand_ratio 0.08 \
    --refine_iterations 3000 \
    --protect_mode hard \
    --refine_tag synthetic_default
```

## 附录 C：技术来源对照表

| 功能 | 来源 | 原始实现 | 本方案实现 |
|------|------|---------|-----------|
| 分块自适应阈值 | GS-LPM | `lpm/utils.py::get_errormap()` | `error_analysis.py::compute_adaptive_error_map()` |
| 射线三角化 | GS-LPM | `lpm/zones_projection.py` | `localization.py::triangulate_3d_zones()` |
| 配对视角筛选 | GS-LPM | `lpm/utils.py::get_paired_views()` | `localization.py::find_paired_views()` |
| 局部密度控制 | GS-LPM | `lpm/lpm.py::lpm_densify_and_clone()` | `gaussian_model_local.py::local_densify_and_prune()` |
| Opacity 校准 | GS-LPM | `lpm/lpm.py::points_calibration()` | `gaussian_model_local.py::calibrate_target_opacity()` |
| 深度反投影 | CL-Splats | `lifter/depth_anything_lifter.py::lift()` | `localization.py::depth_backproject_to_gaussians()` |
| 梯度掩码保护 | CL-Splats | `trainer.py::apply_mask()` | `gaussian_model_local.py::apply_gradient_mask()` |
| 边界约束 | CL-Splats | `constraints/primitives.py` | `gaussian_model_local.py::compute_boundary_loss()` |
| 投影贡献统计 | FlashSplat | `flashsplat_render() + used_count` | `localization.py::compute_contribution_scores()` |
| 多视角标签优化 | FlashSplat | `multi_instance_opt()` | `localization.py::multiview_fusion()` |
