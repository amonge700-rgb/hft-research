# 论文 TC 方法在线化脉冲/PWM 实验记录

对象：Liu 2017 prototype 1，使用论文表 II/III/IV 的 Lm、Ls、f1、fu、fzero 和 TC 电容值。

## 实验设置

- fs = 1e+08 Hz, N = 524288, averages = 16
- 目标频率：f1=8500 Hz, fu=1.04e+06 Hz, fzero=6.1e+06 Hz
- 论文 TC 电容：Cp=5.31 pF, Cs=215.59 pF, Cps=28.36 pF

## 参数反演结果

| method | f1 Hz | fu Hz | fzero Hz | Cp pF | Cs pF | Cps pF | err Cp % | err Cs % | err Cps % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| paper_TC_reference | 8500 | 1.04e+06 | 6.1e+06 | 5.31 | 215.59 | 28.36 | 0 | 0 | 0 |
| single_steep_pulse | 8614.5 | 1.0407e+06 | 3.0036e+06 | 533.29 | 126.61 | 116.99 | 9943 | -41.27 | 312.5 |
| random_prbs_edges | 22119 | 1.0407e+06 | 6.4914e+06 | -3174.4 | 218.55 | 25.047 | -5.988e+04 | 1.374 | -11.68 |
| targeted_binary_pwm | 8198.3 | 1.0407e+06 | 6.0926e+06 | 289.53 | 215.17 | 28.433 | 5352 | -0.1962 | 0.2574 |
| swept_binary_pwm | 8563.5 | 1.0407e+06 | 5.3884e+06 | 12.034 | 207.25 | 36.35 | 126.6 | -3.869 | 28.18 |

## 特征频率误差

| method | feature | target Hz | estimated Hz | err % |
|---|---|---:|---:|---:|
| paper_TC_reference | f1 | 8500 | 8500 | 0 |
| paper_TC_reference | fu | 1.04e+06 | 1.04e+06 | 0 |
| paper_TC_reference | fzero | 6.1e+06 | 6.1e+06 | 0 |
| single_steep_pulse | f1 | 8500 | 8614.5 | 1.347 |
| single_steep_pulse | fu | 1.04e+06 | 1.0407e+06 | 0.072 |
| single_steep_pulse | fzero | 6.1e+06 | 3.0036e+06 | -50.76 |
| random_prbs_edges | f1 | 8500 | 22119 | 160.2 |
| random_prbs_edges | fu | 1.04e+06 | 1.0407e+06 | 0.072 |
| random_prbs_edges | fzero | 6.1e+06 | 6.4914e+06 | 6.416 |
| targeted_binary_pwm | f1 | 8500 | 8198.3 | -3.55 |
| targeted_binary_pwm | fu | 1.04e+06 | 1.0407e+06 | 0.072 |
| targeted_binary_pwm | fzero | 6.1e+06 | 6.0926e+06 | -0.1212 |
| swept_binary_pwm | f1 | 8500 | 8563.5 | 0.7467 |
| swept_binary_pwm | fu | 1.04e+06 | 1.0407e+06 | 0.072 |
| swept_binary_pwm | fzero | 6.1e+06 | 5.3884e+06 | -11.67 |

## 初步判断

该实验检验了论文离线 TC 方法能否通过时域激励重构。若 fzero 附近的频谱能量不足，则 Cps 会首先退化；若 f1 分辨率或低频能量不足，则 Cp 会明显漂移。
