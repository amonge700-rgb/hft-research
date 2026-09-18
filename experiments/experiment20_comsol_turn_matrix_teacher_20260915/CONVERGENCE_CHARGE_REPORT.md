# EXP-020 网格收敛与独立导体电荷法验证

执行日期：2026-09-15

## 结论

基于 `model/exp20_teacher_4p4_axisym_C_LM.mph` 完成粗、中、细三档二维轴对称网格计算。电容矩阵 C 与电感/互感矩阵 L/M 的对角元及重要互项在相邻网格间的最大相对变化均低于 1%。细网格下，COMSOL Voltage Terminal 反应电荷法重建的 C 矩阵与能量法 C 矩阵一致，验证通过。

未修改 `data/raw`；新结果全部写入 `data/convergence`、`model/revisions`、`figures/convergence` 和 `logs`。未加入 DAB、温度、非线性磁芯或 GNN。

## 网格与求解指标

| 网格 | hauto | 单元数 | 顶点数 | 静电 DOF | 磁场 DOF | 网格时间 / s | C 36 工况 / s | L/M 36 工况 / s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 粗 | 5 | 1,164 | 609 | 4,762 | 4,762 | 0.0444 | 43.824 | 40.100 |
| 中 | 3 | 2,175 | 1,136 | 8,892 | 8,892 | 0.1187 | 42.449 | 43.048 |
| 细 | 1 | 18,898 | 9,622 | 76,282 | 76,282 | 0.2038 | 63.928 | 60.220 |

三档计算状态均为收敛。

## 矩阵收敛

| 矩阵 | 网格比较 | 最大对角元相对变化 | 最大重要互项相对变化 | `<1%` |
|---|---|---:|---:|---|
| C | 粗→中 | 0.4483% | 0.6551% | 通过 |
| L/M | 粗→中 | 0.0857% | 0.1190% | 通过 |
| C | 中→细 | 0.4015% | 0.5081% | 通过 |
| L/M | 中→细 | 0.0996% | 0.1270% | 通过 |

## 独立导体电荷法

细网格模型中，将 8 个导体设置为独立 Voltage Terminal。依次对每个端子施加 1 V，其余端子为 0 V，读取 COMSOL 原生端子反应电荷 `es.Q0_i`，以 8 次解的电荷列重建 `C_charge`。

| 指标 | 数值 | 结果 |
|---|---:|---|
| Frobenius 相对误差 | 1.495030457099899e-14 | 通过 |
| 最差矩阵元相对误差 | 6.454191376849653e-12 | 通过 |

## 核心产物

- 最终验证模型：`model/revisions/exp20_teacher_4p4_axisym_C_LM_converged_terminal_verified.mph`
- 三档模型：`model/revisions/*_coarse.mph`、`*_medium.mph`、`*_fine.mph`
- 三档矩阵：`data/convergence/capacitance_maxwell_*.csv`、`inductance_matrix_*.csv`
- 电荷法矩阵及差值：`data/convergence/capacitance_charge_fine.csv`、`capacitance_charge_minus_energy.csv`
- 指标汇总：`data/convergence/mesh_solver_metrics.csv`、`mesh_convergence_summary.csv`、`charge_method_validation.csv`、`summary.json`
- 场图：`figures/convergence/geometry.png`、`final_mesh.png`、`P1_electric_field.png`、`P1_magnetic_flux_density.png`、`P1_S1_coupled_field.png`
- 日志：`logs/convergence_charge_solver_log.txt`、`logs/terminal_charge_validation_log.txt`

## 可追溯说明

`converged_charge_verified.mph` 与其 `v2` 是边界通量积分法的中间排查模型；尖角边界通量积分误差较大，因此不作为最终电荷验证结果。最终验收以原生 Terminal 反应电荷模型 `converged_terminal_verified.mph` 为准。
