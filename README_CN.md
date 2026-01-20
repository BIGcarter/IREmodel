# FERIA-Python

**FERIA (Formation of Envelope and Rotation In Astronomy) - Python 实现版**

[English Documentation](README.md)

FERIA-Python 是 FERIA 模型的纯 Python 实现，专用于模拟和拟合原恒星周围的内落旋转包层 (IRE) 和开普勒盘的运动学特征。它提供了生成 3D 数据立方体 (Data Cube)、位置-速度图 (PV Diagram) 和矩图 (Moment Maps) 的工具，并包含一个强大的网格搜索拟合模块，用于从观测数据中反演物理参数。

## 功能特性

- **正向建模 (Forward Modeling)**: 生成原恒星包层的合成谱线观测数据。
  - 支持内落旋转包层 (IRE) 和开普勒盘 (Keplerian Disk) 运动学模型。
  - 生成 FITS 数据立方体 (RA, Dec, Velocity)。
  - 生成 PV 图和 Moment 0/1 图。
- **可视化**: 自动生成 PNG 图片以便快速检查结果。
- **拟合引擎**: 内置网格搜索框架，将模型与观测数据（Moment-1 速度场）进行拟合。
  - 高效的内存计算（拟合过程中无需磁盘 I/O，速度极快）。
  - 自动计算损失函数（MSE/Chi-square）并可视化（1D 曲线，2D 热图）。
- **纯 Python**: 基于 `numpy` 和 `astropy`，易于安装和二次开发。

## 依赖环境

- python >= 3.8
- numpy
- astropy
- matplotlib

通过 pip 安装依赖:
```bash
pip install numpy astropy matplotlib
```

## 快速开始

### 1. 生成单个模型
运行 `example.py` 生成一个标准的 IRE 模型。该脚本展示了如何定义物理参数并生成数据产品。

```bash
python example.py
```
**输出**: `xfw.fits` (Cube), `example_PV.fits`, `example_mom0.fits`, `example_mom1.fits` 以及对应的 PNG 图片。

### 2. 参数拟合演示
运行 `fit_demo.py` 查看拟合引擎的工作流程。它会先生成一个“伪观测数据”，然后通过网格搜索恢复出设定的输入参数。

```bash
python fit_demo.py
```
**输出**: `fit_results_demo_rcb/` 文件夹，包含损失函数热图（例如：质量 vs 离心半径）。

## 文件结构与功能说明

| 文件名 | 功能描述 |
|------|-------------|
| **`models.py`** | 核心逻辑 `IREModel` 类。处理物理计算、3D 网格生成和投影。 |
| **`fitter.py`** | 包含 `ModelFitter` 类和 `fit_grid` 函数，用于参数估计和网格搜索。 |
| **`params.py`** | 定义 `SourceParams` 数据类，用于管理物理和观测参数。 |
| **`sky.py`** | 实现 `SkyPlane` 类，处理从 3D 网格到 2D 天空平面的投影和卷积。 |
| **`mesh.py`** | 定义 3D 坐标网格 (`Mesh`) 并处理坐标变换。 |
| **`pv_diagram.py`** | 用于从数据立方体中提取位置-速度 (PV) 切片的逻辑。 |
| **`io_utils.py`** | 处理 FITS 文件的读写 (Cube, PV, Moments)，包含完整的 WCS 头信息处理。 |
| **`plot_utils.py`** | 使用 `matplotlib` 可视化 FITS 数据的工具函数。 |
| **`coords.py`** | 坐标转换辅助函数 (RA/Dec <-> Arcsec)。 |
| **`config.py`** | 全局常量配置 (物理常数, 单位换算)。 |
| **`example.py`** | **入口脚本**。用于生成单个模型和模型网格的示例。 |
| **`fit_demo.py`** | **入口脚本**。用于演示如何将模型拟合到数据的示例。 |

## 使用指南

### 定义参数
参数通过字典定义（参考 `example.py`）并传递给 `SourceParams`。关键参数包括：
- `mass_msun`: 原恒星质量 (太阳质量)。
- `rcb_au`: 离心势垒半径 (AU)。
- `inc_deg`: 倾角 (度)。
- `rout_au`: 包层外半径 (AU)。

### 网格搜索拟合
拟合您自己的数据：
1. 准备观测的 Moment-1 FITS 文件。
2. 创建一个类似 `fit_demo.py` 的脚本。
3. 定义一个 `base_params` 字典，匹配观测数据的几何参数（分辨率、距离等）。
4. 定义一个 `grid` 字典，包含需要变化的参数范围。
5. 调用 `fit_grid(obs_file, base_params, grid, ...)`。

```python
from fitter import fit_grid

# ... 定义 base_params ...

grid = {
    'mass_msun': np.arange(100, 200, 10),
    'inc_deg': np.arange(30, 60, 5)
}

fit_grid("my_observation_mom1.fits", base_params, grid, output_dir="my_results")
```
