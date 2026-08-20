# Monte Carlo 参数可辨识性实验

生成时间：2026-07-07

## 实验目的

这一步不再只比较频响拟合误差，而是直接比较参数反演误差。

目标问题是：

> 在不同 PWM 陡脉冲条件下，`Lsigma`, `Cp`, `Rac` 的估计方差是否真的不同？

这比前面的 Fisher 信息扫描更接近在线宽频辨识项目的核心目标。

## 迁移进来的方法

本实验借鉴了三个外部方向的方法：

1. 机器人主动标定中的 observability-aware active calibration：用 Fisher 信息矩阵的最小特征值 `lambda_min(FIM)` 衡量参数可观测性。
2. Bayesian optimal experiment design：不只在一个标称参数点计算 Fisher 信息，而是在参数先验扰动下做平均。
3. Robust regression：用 Huber loss 降低少量异常频点对参数反演的影响。

## 实验设置

- 参数：`theta = [Lsigma, Cp, Rac]`
- 真值：`[2e-6 H, 200e-12 F, 0.2 Ohm]`
- 频率范围：1 kHz 到 10 MHz
- Monte Carlo 重复次数：80
- 基础相对噪声：0.006
- PWM 权重噪声底：0.03
- 最大相对噪声：0.12
- 反演方法：多起点 `fminsearch` + Huber robust loss

## 主要结果

| case | meanRmsePct | std Lsigma % | std Cp % | std Rac % | lambdaMinPriorAvg | condPriorAvg |
|---|---:|---:|---:|---:|---:|---:|
| good_tr100ns_fs500k_D03 | 0.04022 | 0.03836 | 0.03763 | 0.04501 | 3.5795 | 1.8447e7 |
| best_tr50ns_fs500k_D03 | 0.04023 | 0.03674 | 0.04088 | 0.04221 | 4.0244 | 1.7170e7 |
| mid_tr50ns_fs100k_D05 | 0.04827 | 0.04890 | 0.05147 | 0.04498 | 2.9550 | 2.1882e7 |
| weak_tr500ns_fs50k_D07 | 0.06192 | 0.07218 | 0.07455 | 0.03966 | 2.9779 | 2.1068e7 |

## 结论

1. `100 ns / 500 kHz / D=0.3` 和 `50 ns / 500 kHz / D=0.3` 几乎并列最好。
2. `500 ns / 50 kHz / D=0.7` 最差，尤其对 `Lsigma` 和 `Cp` 的估计方差明显变大。
3. 这说明“快上升沿 + 高开关频率”的运行片段确实更适合作为在线宽频辨识的数据来源。
4. Fisher 信息矩阵能预测大趋势，但不能完全代替 Monte Carlo 参数反演，因为非线性、噪声模型、Huber loss 和参数耦合都会影响最终估计误差。

## 下一步

下一步应该把简化模型升级到多端口高频变压器模型：

- 从单端口 `Zin(jw)` 扩展到多端口 `Y(jw)` 或 `Z(jw)` 矩阵；
- 参数从 `[Lsigma, Cp, Rac]` 扩展到 `[Lsigma, Cps, Cg, Cs, Rac, G]`；
- 用实际测得的陡脉冲波形 FFT 替代理想 PWM 包络；
- 比较“运行片段选择”对在线参数跟踪稳定性的影响。
