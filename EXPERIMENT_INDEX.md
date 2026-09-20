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
- `experiment19_learnable_coupling_selection_20260910/`：在相同跨绕组电容边预算下比较距离、幅值、Fisher 代理、真值 oracle 与可学习门控；由精确矩阵求解器验证端口、内部电压和位移电流率失真。
- `experiment20_comsol_turn_matrix_teacher_20260915/`：COMSOL 逐匝高保真教师模型；先提取 4+4 匝 Maxwell 电容矩阵，再扩展至频变阻抗/电感矩阵，并回灌 EXP-015/019。
- `experiment21_comsol_gnn_minimal_20260916/`：12 个参数化 COMSOL 几何、多窗口端口观测、五维物理合法参数化、CUDA edge-centric GNN 与物理反演最小闭环。
- `experiment22_model_mismatch_robust_inverse_20260916/`：在频变电阻、对地寄生和测量链路失配下比较标称、GNN、纯物理及混合反演；结果显示当前 GNN 尚未超过物理基线。
- `experiment23_edge_selection_measurement_update_20260917/`：GNN同时学习静态重要耦合边和测量条件化的稀疏$C_{ij}/M_{ij}$更新；物理矩阵/Kron求解器负责最终响应验证。
- `experiment24_fisher_identifiable_graph_modes_20260917/`：对56维逐匝$C/M$边更新构造Fisher/SVD模态；结果显示现有两端口观测仅支持约3个强耦合组合方向。
- `experiment26a_task_preserving_graph_sparsification_20260920/`：在4+4至32+32匝矩阵物理模型上，对跨绕组电容边与互感关系进行等预算压缩；新增位移电流任务指标，证明电容图可稀疏排序，而互感矩阵必须采用保持正定性的低秩/模态压缩，不能独立二值删边。
- `experiment27_online_modal_identification_20260920/`：将逐匝图在线更新收缩为跨绕组电容整体尺度和两个正定磁模态；使用 DAB 多窗口、宽频小信号、Spectral Mode Perceiver 与物理校正。结果证明磁模态可恢复，但当前端口观测中的电容 Fisher 信息仍不足，必须进入 Cps 定向 OED。

## 最新成果入口

- EXP-016 PDF：`experiments/experiment16_falcon_hft_gnn_20260909/output/pdf/EXP016_FALCON_HFT_GNN_实验报告.pdf`
- EXP-017 PDF：`experiments/experiment17_fisher_oed_subspace_20260909/output/pdf/`
- EXP-018 PDF：`experiments/experiment18_local_hetero_gnn_20260910/output/pdf/EXP018_report.pdf`
- EXP-019 报告：`experiments/experiment19_learnable_coupling_selection_20260910/实验十九阶段报告.md`
- EXP-019 关键结果：`experiments/experiment19_learnable_coupling_selection_20260910/results/`
- EXP-021 报告：`experiments/experiment21_comsol_gnn_minimal_20260916/实验二十一阶段报告.md`
- EXP-021 关键结果：`experiments/experiment21_comsol_gnn_minimal_20260916/results/`
- EXP-022 报告：`experiments/experiment22_model_mismatch_robust_inverse_20260916/实验二十二阶段报告.md`
- EXP-023 报告：`experiments/experiment23_edge_selection_measurement_update_20260917/实验二十三阶段报告.md`
- EXP-023 关键结果：`experiments/experiment23_edge_selection_measurement_update_20260917/results/`
- EXP-024 报告：`experiments/experiment24_fisher_identifiable_graph_modes_20260917/实验二十四阶段报告.md`
- EXP-024 关键结果：`experiments/experiment24_fisher_identifiable_graph_modes_20260917/results/`
- EXP-025 报告：`experiments/experiment25_dab_pwm_time_domain_observability_20260917/实验二十五阶段报告.md`
- EXP-025 关键结果：`experiments/experiment25_dab_pwm_time_domain_observability_20260917/results/`
- EXP-026A 报告：`experiments/experiment26a_task_preserving_graph_sparsification_20260920/实验二十六A阶段报告.md`
- EXP-026A 关键结果：`experiments/experiment26a_task_preserving_graph_sparsification_20260920/results/`
- EXP-027 报告：`experiments/experiment27_online_modal_identification_20260920/实验二十七阶段报告.md`
- EXP-027 诊断激励结果：`experiments/experiment27_online_modal_identification_20260920/results_pilot_20V/`
- EXP-020--022 综合 PDF：`reports/exp21_22_detailed_20260916/output/pdf/EXP020_022_COMSOL_GNN_detailed_report.pdf`
- PDF 总合集：`reports/experiment_report_pdf_collection_20260908/`
- 任富强博士论文：`literature/source_papers/任富强4117004009博士学位论文.pdf`

## 阅读顺序

建议先阅读早期归档中的实验一至九总报告，再依次阅读实验十至十九报告。
Fisher/OED 工作不是被后续实验替代；EXP-017 已进一步证明，形式满秩不等于数值稳定，三端口互补、频点 OED 和参数子空间降维应先于继续增加网络复杂度。
EXP-018 回到逐匝 GNN 主线，结论是异构关系在局部残差的端口和内部电压任务上仅有小幅优势，跨绕组边电流和局部定位仍未解决。
EXP-019 首次把 AI 输出改成耦合边选择。可学习门控接近位移电流 oracle，但也证明当前合成 Cps 矩阵在 40% 边预算下仍有约 28% 位移电流误差；只看端口误差会错误地认为裁剪已经足够。
