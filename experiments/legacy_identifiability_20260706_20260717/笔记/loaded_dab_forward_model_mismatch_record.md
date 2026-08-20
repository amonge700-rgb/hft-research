# 实验七：有载多端口独立前向模型失配验证

## 目的

实验六的生成模型与反演模型完全一致，0.047% 主要是数值自洽精度。本实验故意令前向模型复杂于反演模型，判断在频变铜阻、漏感色散、绕组间介质损耗、弱附加跨端口模态以及通道幅相误差下，简化物理估计器是否留下稳定、可重复的系统残差。

## 边界

- 这是带 DAB 式多运行窗口的频域两端口模型级实验，不是完整开关器件级 DAB。
- 附加 RLC 模态和频变参数是用于压力测试的经验干扰模型，不等价于完整有限元或实物寄生网络。
- 当前结果用于决定后续是否需要物理引导残差学习，不能作为样机精度。

## 结果摘要

- 匹配控制组，重构后三参数物理拟合的 Cps MAE：0.086%。
- 组合失配组，纯模型失配三参数拟合的 Cps MAE：3.787%。
- 组合失配组，重构后二参数物理拟合的 Cps MAE：5.091%。
- 组合失配组，重构后三参数物理拟合的 Cps MAE：5.091%。

## 判断

若组合失配下三参数拟合仍明显高于匹配控制组，且残差随 Cps 状态具有稳定趋势，则后续 AI 的任务应定义为预测 `Delta log(Cps)`，而不是从原始波形黑箱预测 Cps。若增加 Rs 后误差已基本消失，则应优先扩充物理模型，而不是引入神经网络。

## 保存文件

- `模型与代码/run_loaded_dab_forward_model_mismatch.py`
- `数据/forward_model_mismatch_trials.csv`
- `数据/forward_model_mismatch_summary.csv`
- `数据/forward_model_mismatch_scenario_metrics.csv`
- `数据/forward_model_mismatch_config.json`
- `数据/forward_model_mismatch_error_decomposition.png`
- `数据/forward_model_mismatch_residual_structure.png`
- `数据/forward_model_mismatch_representative_y12.png`
