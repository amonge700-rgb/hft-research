# OED 频点选择与参数降维实验结果

生成时间：2026-07-07

## 为什么要改实验

前一轮多端口五参数实验虽然证明了 `Y12` 对完整可辨识性很关键，但 Monte Carlo 参数反演结果仍然一般。

主要问题是：

`Gg` 的估计误差非常大，五参数实验中 `Gg` 的 RMSE 达到 200% 以上，导致整体平均 RMSE 被严重拉高。

这说明在当前频段、噪声模型和测量设置下，`Gg` 属于 practical unidentifiable 参数。继续强行估计它，反而会污染其他参数的判断。

## 改法

引入两个方法：

1. E-optimal frequency selection：从候选频率中选择能最大化 `lambda_min(FIM)` 的频点；
2. practical identifiability parameter reduction：固定弱可辨识参数 `Gg`，先估计核心四参数：

`theta = [Lsigma, Cps, Cg, Rac]`

## 五参数 OED 结果

五参数情况下，40 个频点的 Monte Carlo 排名为：

| strategy | K | meanRmsePct |
|---|---:|---:|
| eopt_nominal | 40 | 47.50 |
| eopt_prior_avg | 40 | 52.24 |
| full_260_freq | 260 | 52.67 |
| log_uniform | 40 | 60.55 |

解释：

- OED 频点选择比等间隔频点更好；
- 但整体误差仍然很大，根本原因是 `Gg` 很难被辨识；
- 所以单纯换频点不能解决所有问题，必须做参数降维或重参数化。

## 固定 Gg 后的四参数结果

固定 `Gg = 1e-6` 后，重新估计 `[Lsigma, Cps, Cg, Rac]`：

| strategy | K | meanRmsePct | Lsigma RMSE % | Cps RMSE % | Cg RMSE % | Rac RMSE % |
|---|---:|---:|---:|---:|---:|---:|
| full_260_4param | 260 | 0.2847 | 0.0223 | 0.6513 | 0.4359 | 0.0296 |
| eopt_nominal_4param | 40 | 0.4034 | 0.0596 | 0.6506 | 0.4389 | 0.4646 |
| eopt_prior_avg_4param | 40 | 0.5019 | 0.0474 | 1.0918 | 0.7689 | 0.0995 |
| log_uniform_4param | 40 | 0.6239 | 0.0542 | 1.4107 | 0.9593 | 0.0715 |

## 关键结论

1. 固定 `Gg` 后，平均参数 RMSE 从几十个百分点降到 1% 以下。
2. 这说明实验“结果一般”的主要原因不是优化算法差，而是参数集合里包含了当前观测下不适合估计的弱辨识参数。
3. 在相同 40 个频点预算下，E-optimal 选频优于等间隔选频：
   - `eopt_nominal_4param`: 0.4034%
   - `log_uniform_4param`: 0.6239%
4. 使用全部 260 个频点仍然最好，但 40 个 E-optimal 频点已经接近全频结果，说明频点选择是有价值的。

## 对项目路线的影响

后续项目不应该直接追求“所有寄生参数都在线辨识”。更合理的路线是：

1. 先做 practical identifiability 分析，筛掉或固定弱辨识参数；
2. 对核心参数做端口-频点-激励联合设计；
3. 用 OED 选择少量高价值频点或运行片段；
4. 最后再逐步放开 `Gg` 等慢变量或弱变量，用强先验/慢时间尺度跟踪。

这条路线比“硬上五参数在线估计”更稳，也更容易写成有说服力的论文方法。
