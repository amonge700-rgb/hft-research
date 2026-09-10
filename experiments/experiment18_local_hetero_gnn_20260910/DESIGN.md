# EXP-018 设计：局部耦合扰动下的逐匝异构 GNN

## 1. 核心问题

在局部匝间/跨绕组电容和局部/低秩磁耦合发生变化时，显式区分物理边类型的逐匝 GNN，能否比把所有关系压成同一邻接的 FALCON 风格模型更准确地预测：

1. 端口宽频导纳；
2. DAB 自然 PWM 谐波下的内部节点电压；
3. 跨绕组电容边的位移电流强度。

本实验不重新研究分段。每个绕组段/等效匝均是图节点。

## 2. 数据来源和边界

- 物理教师：EXP-015 的精确矩阵/Kron 求解器；
- 拓扑规模：4+4、6+6、8+8；
- 工况：DAB 两桥方波的奇次自然谐波，条件包括开关频率、相移、电压比和有限边沿时间；
- 合成扰动：局部电阻/漏感、局部纵向与对地电容、带状非局部 `Cps_ij`、主磁通低秩变化、局部漏磁耦合变化；
- 限制：不是 Simulink 开关器件全模型，不含真实负载动态、死区、Coss、测量链路、COMSOL 或实测数据。

## 3. 图数据契约

### 节点

每匝/导体段一个节点，共 `2N` 个：

```text
node_x = [winding_id, normalized_position, is_port,
          log_Rii, log_Lii,
          fs, phase_shift, voltage_ratio, edge_time]
```

### 四类边

```text
series:       同绕组相邻导体段，属性为局部 R/L 和距离
local_cap:    同绕组纵向/对地等效电容关系
cross_cap:    Cps_ij，允许面对区域及邻近非局部耦合
magnetic:     全局主磁通与局部漏磁的非局部 Mij
```

每条边保存 `source, destination, relation_type, signed_value, distance`。网络内部对无向物理边使用双向消息，但物理矩阵只盖章一次。

### 标签

```text
graph_y:  K 个自然 PWM 谐波上的 Y11/Y22/Y12 幅相
node_y:   每个节点在 K 个谐波下的复电压（实部/虚部，按母线归一化）
edge_y:   每条 cross_cap 边的多谐波 RMS 位移电流（log 标度）
```

## 4. 三个对照模型

1. `falcon_homogeneous`：所有边合并后做同质加权消息传递；
2. `hft_typed`：四类关系分别聚合，但每类只使用标量邻接权；
3. `edge_centric_hetero`：每类独立 edge MLP，显式使用发送/接收节点状态、边值、距离和 DAB 条件，并增加 cross-cap 边读出头。

三者均输出 graph/node/edge 三类任务，避免只给改进模型更多监督。

## 5. 损失与评价

训练损失：

\[
L=\lambda_g L_Y+\lambda_n L_V+\lambda_e L_{I_{ps}}.
\]

评价指标：

- 端口归一化响应 MAE；
- 内部节点复电压 MAE及峰值位置命中率；
- 跨绕组边位移电流 log-MAE；
- 各拓扑和未见 DAB 工况的误差；
- 教师的互易性、无源性、内部 KCL 残差；
- 参数量、训练时间和 CUDA 设备。

## 6. 验收判断

只有当 `edge_centric_hetero` 在未见工况测试集上同时改善至少两项内部任务，并且端口任务不明显退化，才能说异构逐匝边有价值。若仅训练误差更低或仅参数量更多，不构成优势。

Fisher 在本实验中只用于后验评估局部图更新坐标是否能由端口看到，不负责选择任意频点，也不替代 GNN 主线。
