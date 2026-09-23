# WF3 — Internal Inverse Monitoring

## Scientific Question

在有限端口观测条件下，哪些逐匝内部耦合状态真正可辨？如何通过物理先验与观测设计稳定恢复这些内部变化？

## Existing Evidence

- EXP10：4 个局部 `Cps_i` 虽 Jacobian 满秩，但第 2–4 段高度相关（0.992–0.996），提示空间分辨率受限；
- EXP16/18：复杂 GNN 架构未从根本上解决局部定位；
- EXP17：形式满秩不等于数值稳定，低秩可辨识子空间更可靠；
- EXP21/22：GNN 未超过物理反演基线，模型失配下仍存在参数混淆；
- EXP23：静态边重要性可学，但动态 M 边定位失败；
- EXP24：56 个逐边 C/M 更新仅约 3 个强可辨识组合方向；
- EXP25：DAB 多窗口能恢复两端口矩阵，但不能自动增加内部边信息维度；
- EXP27：3 维物理坐标下，磁模态可辨，跨绕组电容尺度方向 Fisher 信息仍明显不足。

## Core Principle

本工作不再追求：
[
	ext{all } C_{ij},M_{ij}
]

而采用：
[
	ext{physical candidate modes}
ightarrow
	ext{Fisher/SVD screening}
ightarrow
	ext{targeted OED}
ightarrow
	ext{online reconstruction}
]

## Candidate Physical Modes

建议首版候选：
- `z_WW_upper`
- `z_WW_middle`
- `z_WW_lower`
- `z_IT`
- `z_TC`
- `z_M1`
- `z_M2`

这些方向由几何、耦合类型、场能量和关键谐振定义，而不是直接用 SVD 数学模态替代。

## Next Experiments

### I1 — Physical Mode Library
为每个候选模态定义：
- 作用区域；
- 参数模板；
- 对 `C/L` 的合法更新方式；
- 物理含义；
- 对内部电压/位移电流的预期影响。

### I2 — Identifiability Audit
计算：
[
J_z=rac{partial y}{partial z},qquad
F_z=J_z^H WJ_z
]

并报告：
- 单模态 sensitivity；
- 模态相关系数；
- singular spectrum；
- `sigma_min`；
- condition number；
- conditional Fisher information。

### I3 — Targeted OED
目标不只是最大能量，而是：
- 增强目标模态 sensitivity；
- 降低与 nuisance 模态的相关性。

例如：
[
F_{C|M}=F_{CC}-F_{CM}F_{MM}^{-1}F_{MC}
]

优化：
- 频率；
- DAB 相移；
- 电压比；
- 运行窗口；
- 必要时小扰动。

### I4 — Three-Level Online Strategy
1. 自然 PWM 谐波；
2. 自然多工况窗口累积；
3. 信息不足时最小诊断扰动。

### I5 — Internal Truth Validation
必须使用内部真值评估：
- modal parameter error；
- 区域定位；
- turn voltage；
- displacement current；
- resonance shift。

禁止只用端口拟合误差作为成功证据。

## Success Criteria

- 目标模态是否可辨；
- 模态之间是否可区分；
- OED 是否提升 `sigma_min` / conditional Fisher；
- 内部状态恢复误差；
- 未见几何/工况泛化；
- 在线计算成本。

## Paper Boundary

本工作研究**内部结构变化反演**；端口模型只作为观测接口，不再把端口等效参数辨识作为主要贡献。
