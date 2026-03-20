# 3D Gaussian Splatting 局部精细化改进方案 — 实施方案文档 (v2)

## 一、方案概述

本方案在原始 3D Gaussian Splatting (3DGS) 代码基础上，新增一条**"先全局训练、后局部精修"**的两阶段流水线。核心思想是：

1. **阶段 A**：沿用原始 3DGS 完成全局训练，获得稳定的全局场景表示（baseline）。
2. **阶段 B**：对 baseline 渲染结果进行多视角误差分析，结合**射线三角化 3D 区域投影**（源自 GS-LPM）、**深度反投影 2D→3D 提升**（源自 CL-Splats）和**alpha 混合贡献统计**（源自 FlashSplat），精准定位低质量区域对应的三维 Gaussian 子集。
3. **阶段 C**：仅对定位出的低质量局部进行二次优化，同时采用**梯度掩码保护**（源自 CL-Splats）+ 锚定正则化严格保护非目标区域，并在目标区域内实施**局部密度控制与几何校准**（源自 GS-LPM），实现"局部增强 + 全局稳定"。

### 设计原则

| 原则 | 说明 |
|------|------|
| **零侵入** | 原始 `train.py` / `render.py` / `metrics.py` 完全不修改，原始参数调用原始逻辑不受影响 |
| **模块化** | 新增功能以独立 Python 文件形式组织，不以插件/hook 形式附加 |
| **可对比** | 输出格式与原始 3DGS 完全兼容，baseline 与 refined 结果可直接用同一套 `render.py` + `metrics.py` 对比 |
| **可消融** | 所有关键步骤均有独立超参开关，可任意组合做消融实验 |

---

## 二、v1 版本存在问题分析

### 2.1 核心缺陷：2D→3D 定位机制不完整

v1 版本的 `localization.py` 存在以下关键问题：

| 问题 | 描述 | 影响 |
|------|------|------|
| **投影逐点循环** | `filter_by_projection()` 中逐 Gaussian 遍历(L115)，对 ~400 万 Gaussian 不可行 | 运行时间数小时级别 |
| **无几何三角化** | 仅使用单视角投影重叠，无法精确确定缺陷对应的 3D 位置 | 定位精度差，误判率高 |
| **无深度反投影** | 不利用深度图将 2D 缺陷像素反向映射到 3D 空间 | 缺少直接的空间对应关系 |
| **无 alpha 贡献统计** | 未利用光栅化过程中每个 Gaussian 对像素的实际渲染贡献 | 遗漏遮挡关系 |
| **CPU DBSCAN** | 在 CPU 上对全量点做 DBSCAN，内存和时间开销大 | 大场景不可用 |

### 2.2 误差分析问题

| 问题 | 描述 |
|------|------|
| **GroupParams 用作 dict** | `analyze_all_views()` 中 `error_params.get()` 但 `GroupParams` 无 `get` 方法 |
| **无自适应阈值** | 固定百分位阈值，未采用 GS-LPM 的 patch 统计自适应方式 |
| **无归一化** | 误差图未做亮度归一化，不同曝光条件下不鲁棒 |
| **无跨视角关联** | 每张图独立检测，缺少多视角一致性验证 |

### 2.3 区域管理问题

| 问题 | 描述 |
|------|------|
| **构造函数不匹配** | `RegionManager(N)` 仅传整数，但 `__init__` 需要三个 mask 参数 |
| **load() 是类方法** | `refine.py` 中先实例化再 `.load()`，但 `load` 是 `@classmethod` 返回新实例 |
| **可视化逐点循环** | `visualize_regions_on_render()` 逐 Gaussian 画点，极慢 |

### 2.4 局部优化问题

| 问题 | 描述 |
|------|------|
| **LR 统一设置** | `refine_training_setup()` 对所有参数用 `target_mult`，应按区域分配 |
| **函数签名错误** | `refine.py` 调用 `set_region_masks()` 但方法名是 `set_regions()` |
| **render_analysis API 不匹配** | `render_with_local_loss()` 参数签名与 `refine.py` 调用不一致 |
| **无曝光调度器** | `refine_training_setup` 缺少 `exposure_scheduler_args` |
| **densify 参数传递错误** | `local_densify_and_prune()` 参数签名与调用不匹配 |

### 2.5 refine.py 主流程问题

| 问题 | 描述 |
|------|------|
| **`run_full_localization` 签名错误** | 传参与定义不匹配 |
| **`dataset.opt` 不存在** | GroupParams 无 `opt` 属性 |
| **`separate_sh=dataset.sh_degree` 逻辑错误** | `separate_sh` 应为 bool，不是 SH 度数 |

---

## 三、参考仓库技术分析与迁移方案

### 3.1 GS-LPM (CVPR 2025): Localized Point Management

**核心技术**：基于特征匹配的多视角误差区域 3D 投影。

| 模块 | 核心思路 | 迁移方式 |
|------|---------|---------|
| `lpm/utils.py::get_errormap()` | 亮度归一化 + 分块统计自适应阈值 | **迁移** → `error_analysis.py` 中增加 `compute_adaptive_error_map()` |
| `lpm/utils.py::set_rays()` | 为每个相机生成像素级射线方向 | **迁移** → `localization.py` 中增加 `compute_camera_rays()` |
| `lpm/utils.py::get_paired_views()` | 基于相机朝向角度筛选邻近视角 | **简化迁移** → 基于相机位置和朝向筛选配对视角，不依赖 LightGlue |
| `lpm/region_matching.py` | 连通域 bounding box + 特征点匹配确定对应区域 | **简化迁移** → 连通域区域提取保留，跨视角用 3D 投影替代特征匹配 |
| `lpm/zones_projection.py::zones3d_projection()` | 双视角射线三角化得到 3D bounding box | **核心迁移** → `localization.py` 中增加 `triangulate_3d_zones()` |
| `lpm/zones_projection.py::get_points_in_cones()` | 锥体内 Gaussian 检测 | **迁移** → `localization.py` 中增加 `find_gaussians_in_zones()` |
| `lpm/lpm.py::points_calibration()` | 缺陷区域内 opacity 重置 | **迁移** → `gaussian_model_local.py` 中增加 `calibrate_target_opacity()` |
| `lpm/lpm.py::lpm_densify_and_clone/split()` | 降低梯度阈值的局部密度控制 | **迁移** → 改写 `local_densify_and_prune()` 增加梯度阈值缩放 |

**关键改动**：不引入 LightGlue/SuperPoint 外部依赖，改用 baseline 已有的相机参数和投影矩阵实现射线计算和区域三角化。

### 3.2 CL-Splats (ICCV 2025): Continual Learning with Local Optimization

**核心技术**：变化检测 + 深度反投影提升 + 梯度掩码局部优化。

| 模块 | 核心思路 | 迁移方式 |
|------|---------|---------|
| `change_detection/dinov2_detector.py` | DINOv2 特征相似度检测变化区域 | **不迁移**（需额外模型），用误差图替代 |
| `lifter/depth_anything_lifter.py::lift()` | 深度反投影 + kNN + 多视角证据累积 | **核心迁移** → `localization.py` 中增加 `depth_backproject_to_gaussians()` |
| `trainer.py::_train_step()::apply_mask()` | 梯度掩码保护非活跃 Gaussian | **已有**，改进 `apply_gradient_mask()` |
| `trainer.py::_train_step()` | photometric loss + boundary constraint loss | **迁移思路** → 改进 `render_analysis.py` |
| `constraints/primitives.py` | 几何基元约束活跃 Gaussian 边界 | **简化迁移** → 用 bounding box 约束替代 |
| 滞后剪枝 (hysteresis pruning) | 连续多次越界才剪枝 | **迁移** → `local_densify_and_prune()` 增加滞后计数 |

**关键改动**：不引入 DINOv2 和 Depth-Anything 外部模型。深度信息直接利用光栅化渲染输出的 `depth_image`；变化检测用已有的复合误差图替代。kNN 查找和多视角证据累积逻辑直接迁移。

### 3.3 FlashSplat (ECCV 2024): 2D to 3D Segmentation

**核心技术**：基于 alpha 混合的 Gaussian 贡献统计 + 线性规划标签分配。

| 模块 | 核心思路 | 迁移方式 |
|------|---------|---------|
| `flashsplat_render()` | 修改光栅化器统计每个 Gaussian 对 mask 像素的 alpha 贡献 | **近似迁移** → 不修改 CUDA 光栅化器，通过 visibility + 投影近似 |
| `multi_instance_opt()` | 多视角 alpha 累计 + 线性规划求解最优标签 | **迁移** → `localization.py` 中增加 `multiview_contribution_voting()` |
| `used_count` 机制 | 每个 Gaussian 累积被 mask 像素使用的次数 | **近似迁移** → 通过投影 + radii + mask 重叠面积近似 |

**关键改动**：FlashSplat 的核心需要自定义 CUDA 光栅化器，不直接引入。采用"投影 + 渲染半径 + 缺陷 mask 面积统计"的方式近似 alpha 贡献，用向量化操作替代逐点循环。

---

## 四、新增文件清单 (v2)

```
gaussian-splatting3/
├── train.py                              # [不修改] 原始训练入口
├── render.py                             # [不修改] 原始渲染入口
├── metrics.py                            # [不修改] 原始评估入口
│
├── refine.py                             # [重写] 局部精修主入口
├── metrics_local.py                      # [重写] 局部区域评估
│
├── arguments/
│   ├── __init__.py                       # [不修改] 原始参数定义
│   └── refine_args.py                    # [重写] 精修超参定义（增加定位策略选择等）
│
├── scene/
│   ├── __init__.py                       # [不修改]
│   ├── gaussian_model.py                 # [不修改]
│   └── gaussian_model_local.py           # [重写] 区域感知 Gaussian 模型
│
├── gaussian_renderer/
│   ├── __init__.py                       # [不修改] 原始渲染函数
│   └── render_analysis.py               # [重写] 带局部损失的渲染包装
│
├── utils/
│   ├── error_analysis.py                 # [重写] 误差图构建与缺陷检测
│   ├── localization.py                   # [重写] 2D-3D 精准定位（核心重写）
│   └── region_utils.py                   # [重写] 三区域管理器
│
└── docs/
    ├── IMPLEMENTATION_PLAN.md            # [重写] 本文档
    └── USAGE_GUIDE.md                    # [重写] 使用指南
```

**总计新增/重写 9 个文件，修改 0 个原始文件。**

---

## 五、技术架构 (v2)

### 5.1 整体流水线

```
┌─────────────────────────────────────────────────────────────────┐
│                      阶段 A: 全局基线训练                          │
│  python train.py -s <data> -m <model> --eval                    │
│  → 输出: point_cloud/iteration_30000/point_cloud.ply            │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      阶段 B: 分析与定位                            │
│                                                                  │
│  B1. 多视角误差图构建 [error_analysis.py]                          │
│      ├── 亮度归一化 (源自 GS-LPM)                                 │
│      ├── L1 光度误差 + 1-SSIM 结构误差 + Sobel 边缘              │
│      ├── 分块统计自适应阈值 (源自 GS-LPM)                         │
│      └── 连通域提取 + 形态学处理                                   │
│           ↓                                                       │
│  B2. 多策略 2D→3D 定位 [localization.py]                          │
│      ├─ 策略 1: 射线三角化 (源自 GS-LPM)                          │
│      │  ├── 相机射线计算                                          │
│      │  ├── 配对视角选择 (相机朝向角度筛选)                        │
│      │  ├── 跨视角区域匹配 (连通域 + 3D 投影)                     │
│      │  └── 双视角射线交叉 → 3D zone bounding box                 │
│      │                                                            │
│      ├─ 策略 2: 深度反投影 (源自 CL-Splats)                       │
│      │  ├── 渲染深度图 → 缺陷像素反投影到 3D                      │
│      │  ├── kNN 查找最近 Gaussian                                 │
│      │  ├── 尺度感知距离 + 深度一致性验证                          │
│      │  └── 多视角正/负证据累积                                    │
│      │                                                            │
│      ├─ 策略 3: 投影贡献统计 (源自 FlashSplat 思想)                │
│      │  ├── Gaussian 中心投影到 2D (向量化)                        │
│      │  ├── 投影 + radii 计算 mask 重叠面积                       │
│      │  └── 多视角贡献累积 + 阈值过滤                              │
│      │                                                            │
│      └─ 策略 4: 梯度归因 (保留 v1)                                 │
│         ├── masked loss 反传                                      │
│         └── 逐 Gaussian 梯度幅值评分                               │
│           ↓                                                       │
│  B3. 多视角投票融合                                                │
│      ├── 加权融合多策略得分                                        │
│      ├── 多视角一致性过滤 (min_views 投票)                         │
│      └── 最终 Gaussian 选择                                        │
│           ↓                                                       │
│  B4. 3D 空间聚类与区域划分                                         │
│      ├── DBSCAN 空间聚类 (GPU 加速距离计算)                        │
│      ├── 上下文环带扩张                                            │
│      └── 最终三区域划分: 目标 / 上下文 / 保护                      │
│                                                                  │
│  → 输出: regions/<tag>/, analysis/<tag>/                          │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      阶段 C: 局部精修训练                          │
│                                                                  │
│  ┌───────────────┐  ┌────────────────┐  ┌────────────────┐      │
│  │  目标区域      │  │  上下文区域     │  │  保护区域       │      │
│  │  正常 LR      │  │  小 LR         │  │  冻结/锚定     │      │
│  │  全参数更新    │  │  部分参数      │  │  不更新        │      │
│  │  可 densify   │  │  不 densify    │  │  不 densify    │      │
│  │  可 calibrate │  │  不 calibrate  │  │  不 calibrate  │      │
│  └───────────────┘  └────────────────┘  └────────────────┘      │
│                                                                  │
│  保护机制 (源自 CL-Splats):                                       │
│  ├── hard: 梯度掩码直接置零 (apply_mask 模式)                     │
│  └── soft: 锚定正则化 + 几何约束                                  │
│                                                                  │
│  局部密度控制 (源自 GS-LPM):                                      │
│  ├── 目标区域降低梯度阈值 (grad_threshold * grad_ratio)            │
│  ├── 目标区域 opacity 校准 (reset in zones)                       │
│  └── 额外点补偿性剪枝 (等量低 opacity 移除)                       │
│                                                                  │
│  损失函数:                                                        │
│  L = L_local + λ_anchor·L_anchor + λ_ctx·L_context               │
│    + λ_bound·L_boundary                                          │
│                                                                  │
│  → 输出: point_cloud/iteration_refine_<tag>/point_cloud.ply      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      评估与对比                                    │
│                                                                  │
│  python render.py -m <model> --iteration refine_<tag>            │
│  python metrics.py -m <model>         # 全图指标                 │
│  python metrics_local.py -m <model>   # 局部区域指标              │
│                                                                  │
│  对比维度:                                                        │
│  ├── 全图 PSNR / SSIM / LPIPS (baseline vs refined)             │
│  ├── 缺陷区域 PSNR / SSIM / LPIPS                               │
│  └── 非缺陷区域 PSNR / SSIM / LPIPS (稳定性验证)                │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 核心模块职责 (v2 更新)

#### `arguments/refine_args.py`

四组超参数类，均继承自原始 `ParamGroup`：

| 参数组 | 职责 | v2 新增超参 |
|--------|------|------------|
| `ErrorAnalysisParams` | 误差图构建 | `use_adaptive_threshold`, `adaptive_patch_size`, `adaptive_fill_ratio` |
| `LocalizationParams` | 2D-3D 定位 | `loc_strategy`, `ray_pair_angle`, `depth_knn_k`, `depth_local_radius`, `contribution_min_overlap` |
| `RefinementParams` | 局部优化 | `grad_ratio`, `calibrate_opacity`, `calibrate_top_ratio`, `lambda_boundary`, `prune_hysteresis` |
| `AblationParams` | 消融控制 | `enable_ray_triangulation`, `enable_depth_backproject`, `enable_contribution_stat`, `enable_gradient_attr` |

#### `utils/error_analysis.py` (v2 重写)

| 函数 | 功能 | v2 变化 |
|------|------|--------|
| `compute_rgb_error()` | 逐像素 L1 光度误差 | 增加亮度归一化选项 |
| `compute_ssim_error_map()` | 1-SSIM | 不变 |
| `compute_edge_error()` | Sobel 边缘 | 不变 |
| `compute_composite_error_map()` | 加权复合 | 增加归一化 |
| `compute_adaptive_error_map()` | **新增** 分块自适应阈值 | 源自 GS-LPM |
| `extract_defect_mask()` | 阈值 + 形态学 | 增加自适应阈值模式 |
| `extract_defect_regions()` | **新增** 连通域 → bbox 列表 | 源自 GS-LPM region_matching |
| `analyze_all_views()` | 批量分析 | 修复 GroupParams 调用方式 |

#### `utils/localization.py` (v2 核心重写)

| 函数 | 功能 | 来源 |
|------|------|------|
| `compute_camera_rays()` | 计算相机像素级射线方向 | GS-LPM `set_rays()` |
| `find_paired_views()` | 基于朝向角度筛选配对视角 | GS-LPM `get_paired_views()` |
| `match_regions_cross_view()` | 跨视角区域匹配 (投影法) | GS-LPM `get_paired_regions()` |
| `triangulate_3d_zones()` | 射线三角化 → 3D zone bbox | GS-LPM `zones3d_projection()` |
| `find_gaussians_in_zones()` | 锥体/bbox 内 Gaussian 检测 | GS-LPM `get_points_in_cones()` |
| `depth_backproject_to_gaussians()` | 深度反投影 + kNN + 证据累积 | CL-Splats `lift()` |
| `compute_contribution_scores()` | 投影贡献统计 (向量化) | FlashSplat 思想 |
| `compute_gradient_attribution()` | masked loss 梯度归因 | v1 保留，修复 |
| `multiview_fusion()` | 多策略加权融合 + 投票 | 新增 |
| `cluster_and_expand()` | DBSCAN + 扩张 | v1 保留，GPU 加速 |
| `run_full_localization()` | 完整流水线 | v2 重写 |

#### `utils/region_utils.py` (v2 修复)

| 类/函数 | 功能 | v2 变化 |
|---------|------|--------|
| `RegionManager` | 三区域管理 | 修复构造函数，增加 `from_masks()` 类方法 |
| `save_error_analysis_results()` | 保存分析结果 | 修复 |
| `visualize_regions_on_render()` | 区域可视化 | 向量化重写，移除逐点循环 |

#### `scene/gaussian_model_local.py` (v2 重写)

| 方法 | 功能 | v2 变化 |
|------|------|--------|
| `refine_training_setup()` | 优化器初始化 | 修复 LR 设置、增加 exposure_scheduler |
| `apply_gradient_mask()` | 梯度掩码 | 改进为 CL-Splats 的 `apply_mask` 广播模式 |
| `compute_anchor_loss()` | 锚定正则 | 增加 per-param 权重 |
| `compute_boundary_loss()` | **新增** 边界约束 | 源自 CL-Splats primitives |
| `calibrate_target_opacity()` | **新增** 目标区域 opacity 重置 | 源自 GS-LPM `reset_localized_points()` |
| `local_densify_and_prune()` | 局部密度控制 | 增加 `grad_ratio` 缩放 + 补偿性剪枝 |

#### `gaussian_renderer/render_analysis.py` (v2 重写)

| 函数 | 功能 | v2 变化 |
|------|------|--------|
| `render_with_local_loss()` | 渲染+损失 | 修复 API 签名，增加 boundary loss |
| `compute_view_defect_weight()` | 采样权重 | 不变 |

---

## 六、关键技术决策 (v2 更新)

### 6.1 多策略定位融合

v2 采用四种互补的 2D→3D 定位策略，通过加权融合提高精度：

| 策略 | 原理 | 优势 | 局限 | 超参控制 |
|------|------|------|------|---------|
| **射线三角化** | 双视角射线交叉定位 3D 区域 | 几何精确 | 需配对视角 | `enable_ray_triangulation` |
| **深度反投影** | 深度图反投影 + kNN 关联 | 直接空间对应 | 依赖深度精度 | `enable_depth_backproject` |
| **贡献统计** | 投影面积加权统计 | 考虑渲染覆盖 | 近似 alpha | `enable_contribution_stat` |
| **梯度归因** | masked loss 梯度评分 | 考虑遮挡关系 | 计算开销 | `enable_gradient_attr` |

每种策略可独立开关，消融实验验证各策略贡献。默认全部启用。

### 6.2 射线计算方案 (源自 GS-LPM)

利用 baseline 的相机内外参数构造射线，无需额外依赖：

```python
# 从相机参数构造射线 (参考 GS-LPM set_rays_od)
proj_inv = cam.projection_matrix.T.inverse()
cam2world = cam.world_view_transform.T.inverse()
# 像素网格 → NDC → 相机坐标 → 世界坐标方向
```

### 6.3 深度反投影方案 (源自 CL-Splats)

利用光栅化器输出的 `depth_image` 进行反投影：

```python
# 缺陷像素的深度值 → 相机坐标系 3D 点
x_cam = (px - cx) / fx * depth
y_cam = (py - cy) / fy * depth
# → 世界坐标 → kNN 查找最近 Gaussian
```

与 CL-Splats 的区别：不使用 Depth-Anything 估计深度，直接使用光栅化器已渲染的深度图。

### 6.4 向量化贡献统计 (源自 FlashSplat 思想)

FlashSplat 修改了 CUDA 光栅化器来统计 alpha 贡献。我们不修改光栅化器，采用近似方案：

```python
# 所有 Gaussian 中心投影到 2D (向量化，无循环)
xy_pixel, depth = project_gaussians_to_2d(gaussians, cam)  # [N, 2], [N]
# 用 radii 估算覆盖范围
# 统计每个 Gaussian 投影覆盖区域与 defect_mask 的重叠面积
```

### 6.5 保护机制 (源自 CL-Splats)

采用 CL-Splats 的 `apply_mask` 广播模式实现梯度掩码：

```python
def apply_mask(param, mask, extra_dims):
    if param.grad is None:
        return
    view_shape = (mask.shape[0],) + (1,) * extra_dims
    param.grad *= mask.view(*view_shape)
```

比 v1 的逐区域条件判断更简洁高效。

### 6.6 局部密度控制 (源自 GS-LPM)

GS-LPM 的核心思想：在误差集中的 3D zone 内降低 densification 的梯度阈值，使更多 Gaussian 被 clone/split：

```python
# 目标区域内降低梯度阈值
local_grad_threshold = grad_threshold * grad_ratio  # grad_ratio < 1
# 额外添加的点通过等量移除低 opacity 点来维持总量平衡
```

### 6.7 上下文环带

- 直接在目标区和保护区之间硬切分会导致**边界接缝**
- 上下文环带允许小幅度更新，实现**平滑过渡**
- 类似 CL-Splats 的局部更新 + 边界缓冲思路

### 6.8 保护机制: hard freeze vs soft anchor

| 模式 | 实现 | 优点 | 缺点 |
|------|------|------|------|
| `hard` | 梯度直接置零 | 绝对保证不变 | 可能有微小接缝 |
| `soft` | 参数偏差惩罚 + boundary loss | 更平滑 | 理论上仍可能有小漂移 |

默认使用 `hard`，消融实验中对比 `soft`。

---

## 七、消融实验设计 (v2 增强)

### 7.1 缺陷检测消融

| 实验 ID | 配置 | 验证目标 |
|---------|------|---------|
| EA-1 | 仅 `w_rgb=1` | RGB 误差基线 |
| EA-2 | `w_rgb=1, w_ssim=0.5` | + 结构信息 |
| EA-3 | `w_rgb=1, w_ssim=0.5, w_edge=0.3` | + 边缘信息 |
| EA-4 | EA-3 + `w_depth=0.5` | + 深度信息 |
| EA-5 | 不同 `error_percentile`: 80/85/90/95 | 阈值敏感性 |
| EA-6 | 固定阈值 vs `use_adaptive_threshold=True` | **新增**: 自适应阈值效果 |
| EA-7 | 不同 `adaptive_patch_size`: 8/16/32 | **新增**: patch 粒度 |

### 7.2 定位策略消融

| 实验 ID | 配置 | 验证目标 |
|---------|------|---------|
| LOC-1 | 仅射线三角化 | GS-LPM 方法单独效果 |
| LOC-2 | 仅深度反投影 | CL-Splats 方法单独效果 |
| LOC-3 | 仅贡献统计 | FlashSplat 思想单独效果 |
| LOC-4 | 仅梯度归因 | v1 方法单独效果 |
| LOC-5 | 射线 + 深度 | 两种几何方法组合 |
| LOC-6 | 全部四策略融合 | **默认**: 完整配置 |
| LOC-7 | 不同 `vote_min_views`: 1/2/3/5 | 多视角一致性 |
| LOC-8 | 不同 `cluster_eps`: 0.02/0.05/0.1 | 聚类粒度 |
| LOC-9 | 不同 `ray_pair_angle`: 30/45/60/90 | 配对角度 |

### 7.3 局部优化消融

| 实验 ID | 配置 | 验证目标 |
|---------|------|---------|
| REF-1 | 全局重训（对照组） | baseline |
| REF-2 | 局部优化，不冻结 (`protect_lr_multiplier=1.0`) | 不保护效果 |
| REF-3 | 局部优化 + hard freeze | 本方案默认 |
| REF-4 | 局部优化 + soft anchor | soft 模式对比 |
| REF-5 | REF-3 + local densify (GS-LPM 式) | + 局部密度控制 |
| REF-6 | REF-5 + calibrate_opacity | **新增**: + 目标 opacity 校准 |
| REF-7 | REF-3 + boundary_loss (CL-Splats 式) | **新增**: + 边界约束 |
| REF-8 | REF-3，仅更新 features | 参数子集消融 |
| REF-9 | REF-3，更新 features + opacity | 参数子集消融 |
| REF-10 | REF-3，全参数更新 | 参数子集消融 |
| REF-11 | 不同 `grad_ratio`: 0.3/0.5/0.7/1.0 | **新增**: 梯度阈值缩放 |
| REF-12 | 不同 `refine_iterations`: 1k/3k/5k/10k | 迭代次数 |

### 7.4 评估指标对比

每组实验输出三组指标：

| 指标类型 | 说明 |
|---------|------|
| 全图 PSNR / SSIM / LPIPS | 整体质量，与 baseline 对比 |
| 缺陷区域 PSNR / SSIM / LPIPS | 局部改善效果 |
| 非缺陷区域 PSNR / SSIM / LPIPS | 稳定性验证（不应下降） |
| Gaussian 数量变化 | 密度控制效果 |
| 区域统计 (target/context/protect 比例) | 定位覆盖度 |

---

## 八、数据流与存储格式

```
<model_path>/
├── cfg_args                                    # [原始] 训练配置
├── cameras.json                                # [原始] 相机参数
├── input.ply                                   # [原始] 初始点云
├── exposure.json                               # [原始] 曝光参数
│
├── point_cloud/
│   ├── iteration_7000/point_cloud.ply          # [原始] 7k checkpoint
│   ├── iteration_30000/point_cloud.ply         # [原始] baseline 模型
│   └── iteration_refine_<tag>/point_cloud.ply  # [新增] 精修后模型
│
├── analysis/<tag>/                              # [新增] 分析结果
│   ├── error_maps/00000.png ...                # 每视角误差图
│   ├── defect_masks/00000.png ...              # 每视角缺陷掩码
│   ├── defect_regions/00000.json ...           # 每视角缺陷区域 bbox
│   └── defect_views.json                       # 缺陷视角索引
│
├── regions/<tag>/                               # [新增] 区域划分
│   ├── target_mask.pt                          # 目标区域 mask
│   ├── context_mask.pt                         # 上下文 mask
│   ├── protect_mask.pt                         # 保护区域 mask
│   ├── scores.pt                               # 融合归因分数
│   ├── zone_bboxes.json                        # 3D zone bounding boxes
│   └── region_meta.json                        # 区域统计
│
├── refine_config_<tag>.json                     # [新增] 精修配置
├── refine_logs_<tag>/                           # [新增] Tensorboard 日志
│
├── results.json                                 # [原始] 全图评估结果
└── results_local_<tag>.json                     # [新增] 局部区域评估结果
```

---

## 九、兼容性保证

### 9.1 原始流程完全可用

```bash
# 以下命令行为完全不变
python train.py -s <data> -m <model> --eval
python render.py -m <model>
python metrics.py -m <model>
```

### 9.2 精修结果可被原始工具评估

精修模型以 `point_cloud.ply` 标准格式保存，可用原始 `render.py` 加载渲染。

### 9.3 新增依赖

| 依赖 | 用途 | 是否必须 |
|------|------|---------|
| `scikit-learn` | DBSCAN 聚类 | 是（定位阶段） |
| `opencv-python` | 形态学操作、连通域 | 是（已在原始代码中使用） |
| `kornia` | 像素网格生成 (射线计算) | 否（可用 torch.meshgrid 替代） |

---

## 十、实施路线图

### 第一阶段：基础框架 (已完成)

- [x] 超参定义模块 v1
- [x] 误差分析模块 v1
- [x] 定位模块 v1
- [x] 区域管理模块 v1
- [x] 局部 Gaussian 模型 v1
- [x] 渲染分析模块 v1
- [x] 精修主入口 v1
- [x] 局部评估脚本 v1

### 第二阶段：v2 重写 (当前)

- [ ] 重写 `arguments/refine_args.py` — 增加定位策略、密度控制超参
- [ ] 重写 `utils/error_analysis.py` — 增加自适应阈值、归一化、区域提取
- [ ] 重写 `utils/localization.py` — 射线三角化 + 深度反投影 + 贡献统计 + 梯度归因融合
- [ ] 重写 `utils/region_utils.py` — 修复构造函数、向量化可视化
- [ ] 重写 `scene/gaussian_model_local.py` — 修复优化器、增加 calibrate/boundary
- [ ] 重写 `gaussian_renderer/render_analysis.py` — 修复 API 签名
- [ ] 重写 `refine.py` — 修复全流程调用
- [ ] 重写 `metrics_local.py` — 增加区域统计

### 第三阶段：功能验证

- [ ] 在 bicycle 场景端到端测试
- [ ] 验证 baseline 逻辑未受影响
- [ ] 验证区域划分的合理性
- [ ] 验证保护机制有效性

### 第四阶段：消融实验

- [ ] 按消融表逐组运行
- [ ] 收集全图/局部/非局部三组指标
- [ ] 确定最优超参组合

---

## 十一、参考文献与技术来源

| 来源 | 借鉴点 | 涉及模块 |
|------|--------|---------|
| **GS-LPM** (CVPR 2025 Highlight) | 射线三角化 3D zone、分块自适应阈值、局部 densify/calibrate | error_analysis, localization, gaussian_model_local |
| **CL-Splats** (ICCV 2025) | 深度反投影提升、梯度掩码保护、边界约束、滞后剪枝 | localization, gaussian_model_local, render_analysis |
| **FlashSplat** (ECCV 2024) | Alpha 贡献统计思想、多视角标签优化 | localization |
| **原始 3DGS** (SIGGRAPH 2023) | 全部基础架构、相机参数、光栅化器 | 所有模块 |
