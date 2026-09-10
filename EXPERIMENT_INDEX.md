# 高频变压器实验总索引

## 早期可辨识性阶段（2026-07-06 至 2026-07-17）

完整归档位于 `experiments/legacy_identifiability_20260706_20260717/`，包含：

1. PWM/Fisher 信息矩阵可辨识性扫描；
2. 多端口 `Y11/Y12/Y22` 可观测性比较；
3. Monte Carlo 带噪参数反演；
4. E-optimal OED 频点选择；
5. 弱参数固定后的降维 OED；
6. 模型失配条件下的 Robust OED；
7. Vector Fitting 及噪声加权拟合；
8. 文献 IC/TC 电容提取复现；
9. 时域 PWM/PRBS 在线频响重构；
10. 宽频机理模型与接地条件复现；
11. `Cps` 到跨绕组位移电流的工程闭环；
12. DAB 多窗口两端口 `Ls+Cps` 联合辨识；
13. 独立复杂前向模型失配验证；
14. 开关状态 DAB 先导实验；
15. CUDA 物理锚定 AI 残差学习。

## 独立实验目录

- `experiment8_switch_state_dab_20260715/`：实验八完整结果。
- `experiment9_ai_residual_pilot_20260716/`：实验九数据、模型权重、图表与记录。
- `experiment10_segmented_ladder_identification_20260721/`：分段梯形网络与局部参数可辨识性。
- `experiment11_segmented_hft_simscape_validation_20260728/`：Simscape 与矩阵模型独立实现比较。
- `experiment12_segmentation_strategy_20260817/`：均匀、物理层和信息引导分段比较。
- `experiment13_parametric_four_terminal_hft_20260821/`：参数化四端子完整电感/跨绕组电容 Simscape 模型。
- `experiment14_dab_switching_20260821/`：分布参数高频变压器接入 DAB 开关运行的回归实验。
- `experiment15_hft_graph_physics_20260901/`：匝级图数据契约、可微矩阵/Kron 求解与物理合规基线。
- `experiment16_falcon_hft_gnn_20260909/`：FALCON 风格同质 GNN、HFT 分型关系 GNN 与结构化物理反演闭环。
- `experiment17_fisher_oed_subspace_20260909/`：八参数逐端口/逐频点 Fisher 信息、log-det OED、原参数筛选与低秩可辨识子空间反演。
- `experiment18_local_hetero_gnn_20260910/`：局部耦合扰动、DAB 自然谐波条件下的逐匝同质/分型/edge-centric 异构 GNN 对照；含绝对目标与局部残差目标消融。

## 阅读顺序

建议先阅读早期归档中的实验一至九总报告，再依次阅读实验十至十八报告。
Fisher/OED 工作不是被后续实验替代；EXP-017 已进一步证明，形式满秩不等于数值稳定，三端口互补、频点 OED 和参数子空间降维应先于继续增加网络复杂度。
EXP-018 回到逐匝 GNN 主线，结论是异构关系在局部残差的端口和内部电压任务上仅有小幅优势，跨绕组边电流和局部定位仍未解决。
