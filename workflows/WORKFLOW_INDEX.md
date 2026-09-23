# HFT Research Workflow Index

本目录将现有实验按“科学问题”而不是时间顺序重新组织。原有 `experiments/` 目录保持不动，避免破坏历史引用和代码依赖。

## 三个工作流

### WF1 — Port Online Identification
**问题：** 在 DAB/PWM 正常运行下，如何在线辨识低阶端口等效参数与宽频端口模型？

主要对象：
- `Cps, Ls, Rs, Lm, Cp, Cs`
- 未来增强型宽频端口模型
- 多窗口 DAB、Fisher/OED、在线参数更新

主要历史实验：
- `legacy_identifiability_20260706_20260717/`
- `experiment8_switch_state_dab_20260715/`
- `experiment9_ai_residual_pilot_20260716/`
- `experiment14_dab_switching_20260821/`（DAB平台/工况支撑）

下一步重点：
1. 新低阶宽频端口结构；
2. 离线扫频/脉冲标定；
3. 在线自然 PWM 谐波辨识；
4. 真实 DAB 硬件验证。

详见：`WF1_port_online_identification/ROADMAP.md`

---

### WF2 — Turn-Level Forward Modeling
**问题：** 如何用低复杂度模型准确保持逐匝内部电磁状态？

主要对象：
- 逐匝 `C/L/R/G` 矩阵
- 图结构/Kron 消元
- 内部电压、位移电流、谐振
- C 稀疏化 + L/互感低秩模态化

主要历史实验：
- `experiment11_segmented_hft_simscape_validation_20260728/`
- `experiment12_segmentation_strategy_20260817/`
- `experiment13_parametric_four_terminal_hft_20260821/`
- `experiment14_dab_switching_20260821/`
- `experiment15_hft_graph_physics_20260901/`
- `experiment19_learnable_coupling_selection_20260910/`
- `experiment20_comsol_turn_matrix_teacher_20260915/`
- `experiment26a_task_preserving_graph_sparsification_20260920/`

下一步重点：
1. C 稀疏 + L PSD 保持模态压缩；
2. sparse/low-rank solver；
3. 4+4→32+32/64+64 多尺度验证；
4. 内部电压/位移电流/谐振保持；
5. COMSOL 与样机内部测量验证。

详见：`WF2_turn_level_forward_modeling/ROADMAP.md`

---

### WF3 — Internal Inverse Monitoring
**问题：** 如何从有限外部观测恢复内部逐匝耦合状态变化？

主要对象：
- 物理可解释内部模态
- Fisher/SVD 可辨识性
- 定向 OED
- 在线内部状态重构

主要历史实验：
- `experiment10_segmented_ladder_identification_20260721/`
- `experiment16_falcon_hft_gnn_20260909/`
- `experiment17_fisher_oed_subspace_20260909/`
- `experiment18_local_hetero_gnn_20260910/`
- `experiment21_comsol_gnn_minimal_20260916/`
- `experiment22_model_mismatch_robust_inverse_20260916/`
- `experiment23_edge_selection_measurement_update_20260917/`
- `experiment24_fisher_identifiable_graph_modes_20260917/`
- `experiment25_dab_pwm_time_domain_observability_20260917/`
- `experiment27_online_modal_identification_20260920/`

下一步重点：
1. 定义物理候选模态库；
2. 计算 sensitivity / correlation / Fisher / SVD；
3. 设计目标模态定向 OED；
4. 验证自然 PWM、多工况累积、小扰动三种信息来源；
5. 以内部真值而非端口拟合误差作为成功标准。

详见：`WF3_internal_inverse_monitoring/ROADMAP.md`

---

## 交叉实验

部分实验天然跨工作流：
- EXP10：逐匝/分段正向模型 → 内部反演的早期桥梁；
- EXP14：DAB 开关工况，可同时服务 WF1/WF2/WF3；
- EXP17：Fisher/OED 方法可复用于 WF1 与 WF3；
- EXP20：COMSOL 教师同时为 WF2 正向真值和 WF3 反演真值提供基础。

## 管理约定

后续新增实验继续保留 `EXP-xxx` 总编号，但在每个 README 顶部加入：

```
Workflow: WF1 / WF2 / WF3
Role: Core / Bridge / Validation / Historical
Paper target: ...
```

这样实验编号保持连续，同时研究主线保持分离。
