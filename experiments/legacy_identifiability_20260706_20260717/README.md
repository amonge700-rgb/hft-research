# 早期可辨识性与在线辨识实验归档

本目录将 2026-07-06 至 2026-07-17 期间原先散落在
`D:\高频变压器\可辨识性分析` 中的实验代码、配置、原始/处理结果、图表、日志、笔记与报告纳入 Git 管理。
迁移采用原样复制，未重新计算或修改数值结果；原始目录仍保留不动。

## 实验地图

| 阶段 | 主要问题 | 关键脚本/结果 |
|---|---|---|
| 基础可辨识性 | PWM 谱权重下哪些参数可辨识 | `run_pwm_sweep_identifiability.m`、`pwm_sweep_*` |
| 多端口观测 | `Y11/Y12/Y22` 是否改善 Fisher 秩与条件数 | `run_multiport_identifiability_demo.m` |
| Monte Carlo | Fisher 预测能否对应带噪反演稳定性 | `run_parameter_mc_identifiability.m`、`mc_parameter_*` |
| OED 选频 | 用最少频点最大化最弱参数信息 | `run_multiport_frequency_oed_demo.m`、`oed_frequency_*` |
| 参数降维 | 固定弱可辨识参数后能否改善反演 | `run_multiport_frequency_oed_reduced_demo.m` |
| Robust OED | 名义模型失配时选频是否仍可靠 | `run_robust_oed_mismatch_demo.m`、`robust_oed_*` |
| Vector Fitting | 宽频响应的有理函数拟合与噪声加权 | `run_vector_fitting_hft_demo.m` |
| 实验一至四 | 文献电容提取复现及 PWM 在线频响桥接 | 总报告第 1--4 章 |
| 实验五 | `f_zero -> Cps ->` 跨绕组位移电流闭环 | `run_cps_common_mode_engineering_demo.py` |
| 实验六 | DAB 多窗口两端口 `Ls+Cps` 联合辨识 | `run_loaded_dab_multiport_cps_identification.py` |
| 实验七 | 独立复杂前向模型制造可解释残差 | `run_loaded_dab_forward_model_mismatch.py` |
| 实验八 | 显式开关状态、死区和多工况先导验证 | `run_switch_state_dab_cps_pilot.py` |
| 实验九 | CUDA 物理锚定 AI 残差学习 | `run_ai_residual_pilot_cuda.py` |

## 目录说明

- `模型与代码/`：MATLAB 与 Python 原始脚本及运行日志。
- `数据/`：MAT/CSV/JSON 结果、统计表和论文图表。
- `笔记/`：每轮实验目的、设置、结果和局限性记录。
- `报告/latex/`：实验一至九总报告 LaTeX 源文件。
- `报告/pdf/`：总报告及实验五、七、八、九专题 PDF。
- `MANIFEST_SHA256.csv`：纳入 Git 的文件哈希，用于迁移完整性审计。

## 已排除内容

- Python/Matplotlib 缓存与 `__pycache__`；
- 从论文 PDF 临时渲染的页面图片 `wideband_paper_pages/`，避免在公开仓库重复分发第三方版权内容；
- LaTeX 临时构建目录。

## 结论边界

这些结果主要属于解析模型、频域模型和开关状态仿真证据。它们证明方法链路和数值可辨识性，
不等同于真实高频变压器样机精度，也不能单独建立 `Cps` 与绝缘老化的唯一对应关系。

