# 模型失配验证实验

生成时间：2026-07-08 13:26:47

## 目的

用更复杂的非对称两端口模型生成频响，再用四参数简化模型反演，检验前面方法是否只对同模型合成数据有效。

## 复杂真值模型

- Ls1 = 1.700e-06 H
- Ls2 = 2.400e-06 H
- C12 = 8.500e-11 F
- Cg1 = 9.500e-11 F
- Cg2 = 1.500e-10 F
- Rac1(f) = 0.080 + 0.055 sqrt(f/1MHz) Ohm
- Rac2(f) = 0.110 + 0.075 sqrt(f/1MHz) Ohm

## 简化辨识模型

`theta = [Lsigma, Cps, Cg, Rac]`, fixed Gg = 1e-6 S.

标称值：Lsigma=2.000e-06 H, Cps=8.000e-11 F, Cg=1.200e-10 F, Rac=2.000e-01 Ohm.

无噪声全频最佳等效参数：Lsigma=2.051e-06 H, Cps=8.489e-11 F, Cg=1.499e-10 F, Rac=9.923e-02 Ohm, cost=2.912e+03.

## Monte Carlo 结果

| strategy | K | meanFitRelErrPct | stdFitRelErrPct |
|---|---:|---:|---:|
| uniform_40_mismatch | 40 | 3.764 | 0.02317 |
| full_260_mismatch | 260 | 6.186 | 0.005877 |
| eopt_40_mismatch | 40 | 33.43 | 0.05028 |

## 解释

如果简化模型在复杂真值模型下仍能保持较小响应拟合误差，说明它可作为核心频段的有效等效模型；如果误差明显增大，则说明必须升级辨识模型或缩窄适用频段。
