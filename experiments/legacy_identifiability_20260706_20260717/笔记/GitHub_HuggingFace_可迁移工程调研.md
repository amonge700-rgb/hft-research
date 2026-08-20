# GitHub / Hugging Face 可迁移工程调研

时间：2026-07-10

目标：为“高频变压器在线宽频辨识”收集可复用工程，重点关注机器人控制、系统辨识、主动实验设计、物理约束学习、宽频阻抗拟合。

## 总体结论

Hugging Face 暂时不是主战场。HF 上能搜到少量 `system identification`、`neural ODE` 相关模型，但下载量和维护度都很低，更像演示模型，不适合作为我们工程主线。

GitHub / PyPI / MATLAB 才是主战场。可迁移工程可以分成五类：

1. 宽频频响有理函数拟合：Vector Fitting / scikit-rf；
2. 稀疏系统辨识：PySINDy；
3. 可微分时域模型：torchdiffeq / torchode；
4. 不确定性与贝叶斯优化：GPyTorch / BoTorch / Ax / Optuna；
5. 物理模型 + 残差学习：PyTorch 自写 residual model。

## 可直接考虑的工程

### 1. Vector Fitting 类

用途：

```text
频域 Y(jw)/Z(jw)
-> 有理函数
-> 状态空间 / 等效宏模型
```

候选：

- `scikit-rf`：Python 射频网络工具，含 VectorFitting 功能，适合处理 S 参数、Z/Y 参数。
- `Hofsmo/vectorFitting`：Python 和 MATLAB 版本的 time-domain vector fitting。
- `kadlecpt/VFTool_0.1`：MATLAB Vector Fitting 工具。
- `EstebanEnriquez/vectorFitting`：受 Gustavsen `vectfit3.m` 启发的实现。

判断：

Vector Fitting 适合做频域宏模型，但不是我们论文的新意。它可以作为工具层，用于把频响曲线转成可仿真的有理函数。

### 2. PySINDy

地址：`https://github.com/dynamicslab/pysindy`

用途：

```text
时域状态响应
-> 从候选函数库里自动发现稀疏动力学方程
```

对我们可能的用法：

- 对 PWM/陡脉冲响应做低阶动态模型发现；
- 在物理模型之外，寻找缺失项；
- 做“可解释 AI”，避免纯黑箱。

风险：

高频变压器的频域响应更自然，PySINDy需要先构造合适状态变量。如果只拿端口电压电流波形直接喂，未必稳定。

### 3. torchdiffeq / torchode

用途：

```text
可微分 ODE 模型
-> 端到端从时域波形反演参数
```

对我们的用法：

```text
theta = [Lsigma, Cps, Cg, Rac, ...]
ODE/状态空间仿真 PWM 响应
loss = measured waveform - simulated waveform
PyTorch 自动微分更新 theta
```

判断：

这是“MATLAB 前端 + PyTorch 后端”最值得探索的方向。MATLAB 用来生成/显示/验证电路响应，PyTorch 用来做可微分优化和 residual learning。

### 4. GPyTorch / BoTorch / Ax

用途：

```text
贝叶斯优化 / 主动实验设计 / 信息增益采样
```

对我们的用法：

```text
候选 PWM 参数：rise time, duty, fs, pulse interval, port combination
候选频点/频段
-> 预测这次观测能降低多少参数不确定性
-> 选择最有信息量且不破坏运行约束的激励
```

判断：

这比传统 FIM-OED 更适合我们，因为可以自然处理：

- 噪声；
- 参数先验；
- 模型失配；
- 安全约束；
- 少量实验预算。

### 5. SIPPY / control-oriented system identification

地址：`https://github.com/CPCLAB-UNIPI/SIPPY`

用途：

传统系统辨识工具，包括 ARX、OE、状态空间等。

对我们的用法：

可作为传统基线，不建议作为主创新。

## 机器人控制可迁移思想

### 主动激励轨迹设计

机器人里会设计关节轨迹来最大化参数可辨识性。迁移到我们这里：

```text
机器人 q(t) 激励轨迹
对应
高频变压器 PWM/陡脉冲运行激励
```

我们的创新可以写成：

```text
运行约束下的自然激励选择，而不是停机扫频。
```

### 模型失配残差学习

机器人 sim-to-real 常见路线：

```text
真实系统 = 物理模型 + learned residual
```

迁移到我们：

```text
Y_meas(jw) = Y_physics(jw, theta) + DeltaY_residual(jw, condition)
```

这正好解决我们已经发现的问题：简化模型有用，但模型失配会导致 OED 失效。

### 可微分物理系统辨识

机器人里用 differentiable physics 从少量真实交互中反演材料/动力学参数。迁移到我们：

```text
等效电路/状态空间可微仿真
-> 端到端拟合 PWM 波形
-> 参数仍然有物理意义
```

这比纯神经网络更适合论文，因为可解释。

## 推荐技术架构

### A. 保守可落地架构

```text
MATLAB:
  生成等效电路频响
  生成 PWM/陡脉冲时域响应
  画图和报告

Python/PyTorch:
  参数优化
  residual 网络
  贝叶斯优化/主动实验设计

数据接口:
  CSV / MAT 文件
```

优点：最容易和现有代码衔接。

缺点：MATLAB 与 Python 之间频繁传文件，工程略繁琐。

### B. 全 MATLAB 架构

```text
MATLAB System Identification Toolbox
MATLAB Optimization Toolbox
Deep Learning Toolbox
自写 FIM / OED / Monte Carlo
```

优点：统一、易画图、用户本地已有 MATLAB。

缺点：深度学习和贝叶斯优化生态不如 PyTorch/BoTorch灵活；复现机器人学习方法会别扭。

### C. 全 Python 架构

```text
numpy/scipy
scikit-rf
PyTorch
BoTorch / GPyTorch
matplotlib
```

优点：更适合深度学习、主动学习、可微分建模。

缺点：要把已有 MATLAB 实验迁移一部分。

## 我建议的路线

不要一上来就做大模型或 Hugging Face 模型。先做一个很小但有学术味的迁移实验：

```text
物理模型：四参数/五参数 Y(jw)
复杂真值：非对称 + Lsigma(f) + Rac(f) + Cg1/Cg2
方法对比：
  1. 纯物理简化模型
  2. 物理模型 + residual GP
  3. 物理模型 + 小 MLP residual
  4. FIM-OED
  5. Bayesian/robust OED
指标：
  响应拟合误差
  参数偏差
  模型失配下频点选择稳定性
  少量观测下的不确定度下降
```

如果这个实验跑通，论文创新点会更清晰：

```text
借鉴机器人主动辨识和模型失配学习，
提出面向高频变压器在线宽频辨识的
物理约束残差模型与鲁棒激励/频点选择方法。
```

## 第一批推荐复现顺序

1. `scikit-rf VectorFitting` 或 MATLAB VF：复现频响到有理函数宏模型。
2. `PyTorch residual model`：在物理 Y(jw) 上学习失配残差。
3. `GPyTorch / BoTorch`：做少量频点下的 Bayesian OED。
4. `PySINDy`：作为可解释模型发现的补充，不放第一位。
5. Hugging Face：暂不作为主线，只用于观察是否有时间序列 foundation model 可做基线。

