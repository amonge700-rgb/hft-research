# 第一轮 Robust OED 实验记录

时间：2026-07-11

脚本：

`D:\高频变压器\可辨识性分析\模型与代码\run_robust_oed_mismatch_demo.m`

输出：

- `D:\高频变压器\可辨识性分析\数据\robust_oed_mismatch_summary.csv`
- `D:\高频变压器\可辨识性分析\数据\robust_oed_design_diagnostics.csv`
- `D:\高频变压器\可辨识性分析\数据\robust_oed_mismatch_fit_error.png`
- `D:\高频变压器\可辨识性分析\数据\robust_oed_selected_points.png`
- `D:\高频变压器\可辨识性分析\数据\robust_oed_design_diagnostics.png`

## 实验目的

验证一个关键问题：

```text
传统 FIM / E-optimal 频点选择在模型失配下为什么会失效？
加入 robust FIM 或宽频覆盖约束后，能不能改善？
```

## 模型设置

复杂真值模型：

```text
非对称两端口
+ Rac(f)
+ Lsigma(f)
+ Cg1/Cg2 不对称
+ C12
+ 弱损耗项
```

简化辨识模型：

```text
theta = [Lsigma, Cps, Cg, Rac]
Gg fixed = 1e-6 S
```

比较方法：

1. `uniform_40`
2. `random_40`
3. `nominal_eopt_40`
4. `prior_avg_eopt_40`
5. `robust_worst_eopt_40`
6. `band_robust_eopt_40`
7. `full_260`

其中 `band_robust_eopt_40` 是新增方法：

```text
先把 1 kHz 到 10 MHz 分成多个频段
每个频段内再做 worst-case robust E-opt
```

这样做是为了避免 E-opt 把频点集中到少数对简化模型敏感、但对复杂真值模型不稳定的频段。

## 主要结果

| strategy | K | meanFitRelErrPct |
|---|---:|---:|
| random_40 | 40 | 3.630 |
| band_robust_eopt_40 | 40 | 4.976 |
| uniform_40 | 40 | 5.481 |
| full_260 | 260 | 6.782 |
| prior_avg_eopt_40 | 40 | 7.264 |
| nominal_eopt_40 | 40 | 29.041 |
| robust_worst_eopt_40 | 40 | 29.182 |

## 解释

第一，普通 `nominal_eopt_40` 明显失败，响应误差约 29%。这说明只按照简化模型的 Fisher 信息选点，会选到对简化模型最敏感、但对复杂真实对象不稳的频段。

第二，`robust_worst_eopt_40` 在简化模型先验族上的 FIM 指标很好，但真实复杂模型误差仍然约 29%。这说明：

```text
对简化模型族 robust
不等于
对真实模型失配 robust
```

第三，`band_robust_eopt_40` 把误差降到约 4.98%，比 `uniform_40` 的 5.48% 更好。这说明宽频覆盖约束非常重要，不能让 OED 自由地把点全部集中到某些频段。

第四，本轮 `random_40` 最低，约 3.63%。这不能直接说明随机最好，因为这里只跑了一个随机种子；但它提示我们：在强模型失配下，频点分散性可能比简化模型 FIM 更重要。下一步必须做多随机种子统计。

## 当前结论

这次实验不是最终方法，但它给出了很有价值的论文线索：

```text
传统 FIM-OED 在模型失配下可能产生过度自信的频点选择。
仅对简化模型参数先验做 robust FIM 仍然不够。
加入宽频覆盖约束后，模型失配下的响应拟合稳定性明显改善。
```

这可以发展成：

```text
模型失配感知的鲁棒频点选择
```

而不是简单的：

```text
FIM 最大化
```

## 下一步

建议下一步做两个实验：

1. 多随机种子统计：

```text
运行 50 到 100 个 random_40
比较 random 分布、uniform、band_robust 的误差分布
```

目标是判断 `random_40` 是偶然好，还是“分散覆盖”真的比 FIM 集中选点更稳。

2. Python / BoTorch 版本的 Bayesian OED：

```text
用 GP 或 ensemble surrogate 预测不同频点组合下的响应误差
目标函数不再只看 FIM，而是看 expected response error / expected uncertainty reduction
```

这会更接近我们从机器人主动辨识里迁移过来的方法。

