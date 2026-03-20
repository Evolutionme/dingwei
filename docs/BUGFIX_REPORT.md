# Local Refinement Pipeline — Bug 修复报告

**日期**: 2026-03-19  
**范围**: 修复 `refine.py` 及其依赖模块在大规模高斯模型（3.9M Gaussians）上运行时的多个致命错误  
**验证**: 3 视图端到端测试通过（error analysis → localization → region management），所有 4 种定位策略均正常工作

---

## 一、问题概述

在 bicycle 数据集（3,907,825 个高斯体）上运行 `refine.py --skip_refine` 时，pipeline 在 Stage B（Error Analysis & Localization）阶段连续崩溃，表现为以下错误链：

| 错误类型 | 错误信息 | 根因 |
|---------|---------|------|
| **Bug #1** | `RuntimeError: numel: integer multiplication overflow` | rasterizer 内部 int32 溢出 |
| **Bug #2** | `torch.cuda.OutOfMemoryError: CUDA out of memory` (22GB/24GB) | 显存严重浪费 |
| **Bug #3** | `torch.cuda.OutOfMemoryError` (14.56 GiB allocation) | depth_backproject 距离矩阵 OOM |
| **Bug #4** | `AttributeError: 'NoneType' object has no attribute 'zero_grad'` | optimizer 未初始化 |
| **Bug #5** | `torch.cuda.OutOfMemoryError` (11.69 GiB allocation) | cluster_and_expand 距离矩阵 OOM |

---

## 二、根因分析与修复

### Bug #1: Rasterizer Int32 溢出

**根因**: `separate_sh=False` 时，SH 特征以完整 `[N, 16, 3]` 张量传入 rasterizer 的 `shs` 参数（`dc=empty`），导致 C++ 内核计算 `sampleBuffer` 大小时 int32 溢出。原版 `render.py` 使用 `separate_sh=SPARSE_ADAM_AVAILABLE`（即 `True`），将 DC 和 rest 分开传入，避免了此问题。

**修复文件**: `refine.py`

```python
# 新增检测
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

# 所有 render 调用传入 separate_sh=SPARSE_ADAM_AVAILABLE
run_analysis_and_localization(..., separate_sh=SPARSE_ADAM_AVAILABLE)
run_local_refinement(..., separate_sh=SPARSE_ADAM_AVAILABLE)
```

**影响**: 所有 3 个 stage 函数签名新增 `separate_sh` 参数，一路透传至 `render()` 调用。

---

### Bug #2: 显存浪费导致 OOM (22GB/24GB)

此 bug 有 **三个独立来源**，逐一修复后总显存降至 ~1GB：

#### 2a. Camera 图像占满 GPU (~4GB)

**根因**: `cfg_args` 中 `data_device='cuda'`，导致 194 张训练图像全部加载到 GPU（每张 ~20MB × 194 = ~3.8GB）。原版 `render.py` 只渲染 25 张测试图不受影响。

**修复文件**: `refine.py` → `load_baseline()`

```python
original_data_device = dataset.data_device
dataset.data_device = "cpu"  # 图像留在 CPU，render 时按需 .cuda()
gaussians = LocalGaussianModel(dataset.sh_degree)
scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
dataset.data_device = original_data_device
```

#### 2b. Optimizer 状态提前分配 (~1.8GB)

**根因**: `load_baseline()` 中过早调用 `training_setup(opt)`，为 3.9M 高斯体创建 Adam 优化器状态（每参数 2 个等大小状态张量），在分析阶段完全不需要。

**修复文件**: `refine.py` → `load_baseline()`

```python
# 移除: gaussians.training_setup(opt)
# training_setup 延迟到 Stage C 的 refine_training_setup() 中调用
```

#### 2c. 分析结果累积占满 GPU (~12GB+)

**根因**: `analyze_all_views()` 在 `results[idx]` 中存储了完整的 `render_pkg`（含 `viewspace_points [N,3]` 和 `radii [N]`），169 个视角 × 3.9M 高斯体 = ~12GB。还存储了不再需要的 `rendered`/`gt` 图像。

**修复文件**: `utils/error_analysis.py` → `analyze_all_views()`

```python
# 修改前: 存储完整 render_pkg + rendered + gt
results[idx] = {"render_pkg": {k: v.detach() ...}, "render": rendered, "gt": gt, ...}

# 修改后: 仅保留 localization 需要的 depth 和 radii，移至 CPU
slim_pkg = {}
if "depth" in render_pkg: slim_pkg["depth"] = render_pkg["depth"].detach().cpu()
if "radii" in render_pkg: slim_pkg["radii"] = render_pkg["radii"].detach().cpu()
results[idx] = {"error_map": error_map.cpu(), "defect_mask": defect_mask,
                "defect_regions": defect_regions, "render_pkg": slim_pkg}
```

---

### Bug #3: `depth_backproject_to_gaussians` 距离矩阵 OOM

**根因**: `torch.cdist(pts_chunk, xyz)` 使用固定 `chunk_size=1000`，产生 `[1000, 3.9M]` = **14.9GB** 的距离矩阵，远超显存。

**修复文件**: `utils/localization.py` → `depth_backproject_to_gaussians()`

```python
# 修改前: 固定 chunk_size
chunk_size = min(1000, M)

# 修改后: 根据可用显存动态计算安全 chunk_size
bytes_per_row = N * 4
free_mem = torch.cuda.mem_get_info()[0]
safe_chunk = max(16, int(free_mem * 0.3 / bytes_per_row))
chunk_size = min(safe_chunk, M)
```

同时改进: 预计算 `scale_denom`、使用 `dists.div_()` 原地运算、`del dists` 及时释放。

---

### Bug #4: Gradient Attribution 缺少 Optimizer

**根因**: `compute_gradient_attribution()` 在函数开头和结尾均调用 `gaussians.optimizer.zero_grad()`，但分析阶段未调用 `training_setup()`，`optimizer` 为 `None`。

**修复文件**: `utils/localization.py` → `compute_gradient_attribution()`

```python
# 两处 zero_grad 均改为:
if hasattr(gaussians, 'optimizer') and gaussians.optimizer is not None:
    gaussians.optimizer.zero_grad(set_to_none=True)
else:
    for p in [gaussians._xyz, gaussians._features_dc, ...]:
        if p.grad is not None:
            p.grad = None  # 或 p.grad.zero_()
```

---

### Bug #5: `cluster_and_expand` 距离矩阵 OOM

**根因**: 上下文环扩展使用 `torch.cdist(chunk, target_xyz)` 时固定 `chunk_size=8192`。当 target Gaussians 数量 T 很大时（例如 T=383K），距离矩阵 `[8192, 383K]` = **11.7GB**，远超显存。

**修复文件**: `utils/localization.py` → `cluster_and_expand()`

```python
# 修改前: 固定 chunk_size
chunk_size = 8192

# 修改后: 根据 target 数量 T 和可用显存动态计算
bytes_per_row = T * 4
free_mem = torch.cuda.mem_get_info()[0]
chunk_size = max(64, min(8192, int(free_mem * 0.3 / max(bytes_per_row, 1))))
```

---

## 三、辅助改进

### 3.1 Localization 进度日志

**修复文件**: `utils/localization.py` → `run_full_localization()`

```
Localization view 1/169 (idx=0) -> ['ray', 'depth', 'contrib', 'grad']
Localization view 2/169 (idx=1) -> ['ray', 'depth', 'contrib']
```

### 3.2 Gradient Attribution OOM 容错

```python
try:
    grad_scores = compute_gradient_attribution(...)
except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
    if "out of memory" in str(e).lower():
        enable_grad = False  # 自动禁用后续视图的梯度归因
        torch.cuda.empty_cache()
```

### 3.3 CPU↔GPU 张量迁移

Localization 函数中增加了对 `depth`、`radii`、`defect_mask` 从 CPU 到 GPU 的按需迁移：

```python
if depth_img is not None and depth_img.device.type == 'cpu':
    depth_img = depth_img.cuda()
```

---

## 四、修改文件清单

| 文件 | 修改类型 | 说明 |
|------|---------|------|
| `refine.py` | 核心修复 | +SPARSE_ADAM_AVAILABLE 检测, separate_sh 透传, data_device='cpu', 移除 training_setup, +--max_loc_views |
| `utils/error_analysis.py` | 内存优化 | slim render_pkg (仅保留 depth/radii), 结果移至 CPU |
| `utils/localization.py` | 多项修复 | depth_backproject + cluster_and_expand chunk 动态化, optimizer None 处理, OOM 容错, 进度日志, CPU→GPU 迁移 |

---

## 五、验证结果

端到端测试 (`test_e2e.py`)：analysis → localization → fusion → clustering → RegionManager save/load

```
=== ALL E2E TESTS PASSED ===
  169 cams, 3907825 Gs, GPU=0.93GB          ✅ (原 22GB → 0.93GB)
  Error analysis (3 views): 0.7s            ✅ (原 int32 溢出)
  Depth backproject: OK                     ✅ (原 14.56GB OOM)
  Gradient attribution: OK                  ✅ (原 AttributeError)
  Localization 2 views × 4 strategies       ✅ (ray, depth, contrib, grad)
  cluster_and_expand: OK                    ✅ (原 11.69GB OOM)
  Target=383,129 | Context=814,212 | Protect=2,710,484
  RegionManager save + reload: OK           ✅
  GPU final: 0.99GB                         ✅
```

**注意**: 完整 169 视图的 localization 预计需要 ~5-8 小时（3.9M Gaussians × 4 策略 × 169 视图）。可通过以下方式加速：
- `--max_loc_views 20` 限制 localization 视图数量（推荐用于快速测试/调参）
- `-r 2` 或 `-r 4` 降低分辨率（减少 defect 像素数量）
- 禁用耗时策略: 在 `AblationParams` 中设置 `--no_gradient_attr` 或 `--no_ray_triangulation`
- 减少高斯数量: 使用更激进的 prune 参数训练 baseline

---

## 六、与原版 Baseline 的兼容性

| 原版功能 | 兼容状态 |
|---------|---------|
| `train.py` | ✅ 未修改 |
| `render.py` | ✅ 未修改 |
| `metrics.py` | ✅ 未修改 |
| `GaussianModel` | ✅ 未修改（LocalGaussianModel 仅继承扩展） |
| CLI 参数 | ✅ 新增参数均有默认值，不影响原有命令行 |
