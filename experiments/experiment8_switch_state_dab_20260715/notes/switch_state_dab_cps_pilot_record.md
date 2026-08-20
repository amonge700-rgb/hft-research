# 实验八：开关状态 DAB 多工况 Cps 辨识先导实验

## 结果

- 全部多工况 Cps MAE：0.612%
- 匹配开关状态基线 Cps MAE：0.128%
- 弱附加跨端口模态 Cps MAE：1.288%
- 完整测量链路场景 Cps MAE：0.419%
- 仅用 Cps 预测总电容源电流的平均误差：87.951%

## 边界

本实验是刚性直流母线下的开关状态先导模型。负载通过相移运行点进入；死区和 Coss 用零状态与有限边沿表示。它不是逐器件 SPICE，也没有闭环输出电容动态。输出中的 `ips` 是跨绕组位移电流；`iground_source` 是对地电容源电流，两者之和仍不等同于完整系统共模回流电流。

## 数值校准记录

第一版使用零初值时域递推，短预热窗口没有覆盖约 1 ms 的漏感支路时间常数，导致匹配模型也出现 45%--63% 的伪误差。正式结果改为整数个开关周期记录，并对线性变压器支路求周期稳态频谱响应。匹配开关状态 MAE 随后降至 0.128%。失败版本未作为正式 CSV 保留，避免与校准后结果混用。

正式结果共 108 条逐次试验、72 条工况汇总，随机种子为 20260715。

## 保存文件

- `模型与代码/run_switch_state_dab_cps_pilot.py`
- `数据/switch_state_dab_cps_trials.csv`
- `数据/switch_state_dab_cps_summary.csv`
- `数据/switch_state_dab_cps_config.json`
- `数据/switch_state_dab_cps_summary.png`
- `数据/switch_state_dab_representative_waveform.png`
