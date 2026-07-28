# 实验九：CUDA 物理锚定频谱残差学习先导实验

## 目的

用 512 个独立物理状态验证 AI 数据接口、分组划分、物理残差标签和 CUDA 训练闭环。网络不从零预测参数，而是修正固定三参数物理反演得到的 `Delta log(Ls)` 与 `Delta log(Cps)`。

## 数据

- 独立物理状态：512。
- 每状态测量扰动：4。
- 总记录：2048。
- 频率令牌：64 个，80 kHz--10 MHz。
- 划分：320 train / 64 val / 64 interpolation test / 64 shifted test，按物理状态分组，无重复窗口泄漏。

## CUDA

- GPU：NVIDIA GeForce RTX 5070。
- PyTorch：2.11.0+cu128，CUDA runtime：12.8。
- 混合精度：float16 AMP。

## Cps MAE

| 测试集 | 纯物理 (%) | MLP 修正 (%) | Spectral Perceiver 修正 (%) |
|---|---:|---:|---:|
| validation | 3.610 | 1.075 | 1.167 |
| interpolation test | 3.584 | 1.018 | 1.253 |
| shifted test | 6.686 | 2.484 | 2.827 |

## 边界

这是便宜频域复杂前向模型上的 AI 先导实验，不是独立 SPICE、FEM 或样机精度。shifted test 只能检验预设分布偏移，不能替代跨求解器和硬件验证。后续是否扩充数据，应先看学习曲线和跨域误差，而不是盲目生成一万组同源样本。
