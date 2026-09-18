# EXP-020：COMSOL 逐匝矩阵教师模型

## 1. 研究目的

用独立于现有合成矩阵的 COMSOL 高保真场模型，生成逐匝电容、电感/阻抗标签，替换 EXP-015～019 中的合成教师参数。第一阶段只做静电场，避免把几何、磁场、频变损耗和 DAB 开关同时引入而无法定位误差来源。

## 2. 分阶段范围

### EXP-020A：逐匝 Maxwell 电容矩阵

- 几何：二维轴对称、原边 4 匝、副边 4 匝、磁芯/屏蔽参考导体和空气/绝缘区域；
- 每匝定义为独立导体终端；
- 依次对第 `j` 匝施加 1 V，其余导体及参考导体为 0 V；
- 积分所有导体电荷，按 `Q = C V` 得到 Maxwell 电容矩阵第 `j` 列；
- 输出完整 `8 x 8` 矩阵及匝对参考导体的电容信息。

场论输出采用 Maxwell 约定：

\[
Q_i=\sum_j C_{ij}^{\mathrm{Maxwell}}V_j,
\qquad C_{ii}>0,\quad C_{ij}\leq 0\;(i\ne j).
\]

转换为电路互电容时：

\[
c_{ij}^{\mathrm{mutual}}=-C_{ij}^{\mathrm{Maxwell}},\qquad i\ne j.
\]

### EXP-020B：逐匝阻抗/电感矩阵

- 使用 Magnetic Fields 或频域线圈模型；
- 逐匝 1 A 激励，提取多端口 `Z_ij(f)`；
- 在准静态有效频带内计算 `L_ij(f)=Im{Z_ij(f)}/omega`；
- 保留 `Re{Z_ij(f)}`，避免把高频损耗错误压缩成固定电阻。

该阶段在 EXP-020A 验证通过后实施。

## 3. 第一版参数

第一版是方法验证样机，不宣称对应最终实物。所有长度和材料必须集中定义为 COMSOL Parameters，不允许散落硬编码。

| 参数 | 初值 | 含义 |
|---|---:|---|
| `Np`, `Ns` | 4, 4 | 原/副边独立导体数 |
| `r_core` | 10 mm | 中心参考导体/磁芯等效半径 |
| `t_ins_core` | 1 mm | 绕组到参考导体绝缘间隙 |
| `w_turn` | 1 mm | 轴向导体宽度 |
| `t_turn` | 0.5 mm | 径向导体厚度 |
| `g_turn` | 0.25 mm | 同绕组相邻匝间隙 |
| `g_ps` | 1.5 mm | 原副边径向绝缘间距 |
| `epsr_ins` | 3.5 | 第一版均匀绝缘相对介电常数 |
| `r_air` | 35 mm | 外空气域半径 |
| `h_air` | 25 mm | 外空气域高度 |

这些数值只用于打通提取流程；进入论文数据集前必须替换为样机尺寸或可追溯文献/设计值。

## 4. 数据契约

COMSOL 导出到 `data/raw/`：

- `turn_metadata.csv`：匝编号、原/副边、层号、几何中心、尺寸和材料；
- `capacitance_maxwell.csv`：带符号的 Maxwell 电容矩阵，单位 F；
- `capacitance_mutual.csv`：正值互电容边表，单位 F；
- `capacitance_to_reference.csv`：各匝对参考导体的等效电容；
- `model_parameters.csv`：本次求解的全部参数；
- `solver_log.txt`：网格、自由度、收敛信息和 COMSOL 版本；
- `field_energy_check.csv`：电荷法与电场能量法一致性检查。

导入 EXP-015 的图结构时，节点顺序固定为：

`P1, P2, P3, P4, S1, S2, S3, S4`。

## 5. 验收标准

1. 对称性：`||C-C^T||_F/||C||_F < 1e-6`；
2. 物理符号：对角元为正，非对角元非正；
3. 正半定性：最小特征值只允许存在网格/舍入量级负偏差；
4. 电荷守恒：包含参考导体的完整 Maxwell 矩阵行和接近零；
5. 网格收敛：关键互电容在连续两级网格间变化小于 1%；
6. 能量一致性：`0.5*V^T*C*V` 与 COMSOL 电场能量相对误差小于 1%；
7. 矩阵组网后能够由 EXP-015 求解器恢复无源、互易的端口导纳。

## 6. 现成模型的使用边界

本机 COMSOL 6.4 AC/DC Module 提供：

- `ecore_transformer.mph`：借鉴磁芯、线圈和外场域设置；
- `inductance_matrix_pcb_coils.mph`：借鉴多线圈电感矩阵提取；
- `capacitor_dc.mph` / `capacitor_fringing_fields.mph`：借鉴静电终端、能量与网格设置。

这些模型只作为设置模板，不直接充当训练数据，也不直接复制其材料和尺寸。

## 7. 与 GNN 主线的接口

EXP-020 输出是高保真教师图：

\[
\mathcal G_{\mathrm{geometry}}
\rightarrow
\{\mathbf C,\mathbf Z(\omega)\}.
\]

它首先用于重新运行 EXP-019 的边预算扫描，回答在真实场提取矩阵上保留多少边才满足端口、内部电压和位移电流误差要求；随后再训练 GNN 学习跨几何结构的耦合先验。COMSOL 不负责在线运行，在线阶段仍由端口测量、低维参数更新和快速矩阵求解器完成。

