# WF2 — Turn-Level Forward Modeling

## Scientific Question

如何构建一个既保持逐匝内部电磁状态、又具有可控复杂度和可计算性的高频变压器正向模型？

## Existing Evidence

- EXP11–14：分段/四端子/DAB 开关交叉验证；
- EXP15：统一图数据规范、矩阵物理、Kron 消元、内部状态恢复；
- EXP19：边重要性筛选，证明端口保持不等于内部位移电流保持；
- EXP20：COMSOL 逐匝 Maxwell 电容矩阵与 L/M 高保真教师；
- EXP26A：任务保持图压缩，得到：
  - C 图具有可稀疏性；
  - M 不能逐边独立删除；
  - 磁侧必须采用 PSD 保持的低秩/模态表示；
  - 现有 dense solver 尚未获得真实速度收益。

## Core Model

完整节点导纳：

[
Y_n(omega)=A(R+jomega L)^{-1}A^T+G+jomega C
]

Kron 消元：

[
Y_{port}=Y_{pp}-Y_{pq}Y_{qq}^{-1}Y_{qp}
]

内部恢复：

[
v_q=-Y_{qq}^{-1}Y_{qp}v_p
]

## Current Gap

高质量论文不能只证明“可以删边”，必须形成一套统一的、物理合法的模型降阶方法，并给出真实计算收益。

## Next Experiments

### F1 — Unified C/L Reduction
电容：
[
Cightarrow C_{sparse}
]

磁侧：
[
Lapprox D+U_rLambda_rU_r^T,quad Lambda_rge0
]

目标：统一结构下同时保持：
- 端口导纳；
- 逐匝电压；
- 局部位移电流；
- 谐振频率/峰值。

### F2 — Sparse/Low-Rank Solver
实现：
- sparse C assembly；
- low-rank L update；
- Woodbury/Schur 等低秩求解；
- runtime/memory scaling。

规模：
- 4+4；
- 8+8；
- 16+16；
- 32+32；
- 64+64（若资源允许）。

### F3 — Multi-Geometry Generalization
覆盖：
- 单层；
- 多层；
- 交错；
- 非对称；
- 不同绝缘间距。

### F4 — High-Fidelity / Hardware Validation
- COMSOL 全模型作为教师；
- 代表性样机内部匝电压测量；
- 必要时局部位移电流通过模型/场量间接验证。

## Success Criteria

必须同时报告：
- `Y_port` 误差；
- `V_turn` 误差；
- `I_displacement` 误差；
- resonance error；
- runtime；
- memory；
- reduction ratio。

## Paper Boundary

本工作是**正向模型与降阶**，不研究如何从端口反推出内部参数。
