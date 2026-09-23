# WF1 — Port Online Identification

## Scientific Question

在不脱离 DAB/PWM 正常运行的前提下，如何建立一个低阶、宽频、可解释且可在线辨识的端口模型，并稳定获得关键等效参数？

## Existing Evidence

- 早期工作已经覆盖：
  - Liu/TC 低阶参数模型；
  - `Cps` 特征频率提取；
  - PWM/Fisher 信息扫描；
  - Monte Carlo 带噪反演；
  - E-optimal OED；
  - DAB 多窗口两端口 `Ls+Cps` 联合辨识；
  - switch-state DAB 与模型失配测试。
- EXP8：引入显式开关状态、死区、有限边沿、弱振铃及负载/温度/接地变化。
- EXP9：探索物理锚定残差学习，但不应作为主线。
- EXP14：提供参数化四端子 HFT 接入 DAB 开关运行的仿真支撑。

## Current Gap

当前主要还是沿用既有低阶结构。高质量论文需要一个真正适合“宽频 PWM + 在线辨识”的新端口模型，而不是只换辨识算法。

建议模型结构方向：
1. 差模/共模分离作为主框架；
2. 一阶频变漏感/损耗支路；
3. 少量附加谐振模态；
4. 总参数数控制在约 8–10 个。

## Next Experiments

### P1 — Enhanced Port Model Selection
目标：比较 Liu 基线与若干候选增强结构。

输出：
- 宽频 FRF 拟合误差；
- 参数可辨识性；
- Fisher 特征值/条件数；
- 模型复杂度；
- 是否存在冗余参数。

### P2 — Offline Calibration
使用扫频或单次宽频脉冲建立：
- 标称参数；
- 敏感频带；
- 目标特征；
- 频率—参数灵敏度地图。

### P3 — Online Natural PWM Identification
只使用正常 DAB 谐波：
- 多窗口端口矩阵重构；
- 信息阈值触发更新；
- 参数跟踪误差；
- 工况变化鲁棒性。

### P4 — Hardware Validation
真实 DAB 平台：
- `v1,i1,v2,i2` 同步采样；
- 多相移、多负载、多电压比；
- 死区、温漂、传感器误差；
- 离线 FRA/脉冲结果作参考真值。

## Success Criteria

- 端口参数误差；
- FRF 误差；
- 在线更新时间；
- 多工况稳定性；
- 相比传统模型是否显著改善；
- 参数变化是否能对应明确工程量（如位移电流/ZVS边界）。

## Paper Boundary

本工作只研究**端口等效参数在线辨识**，不宣称恢复逐匝内部 `Cij/Mij`。
