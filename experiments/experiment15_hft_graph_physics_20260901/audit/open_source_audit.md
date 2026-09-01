# EXP-015 开源实现轻量审计

审计日期：2026-09-01。仓库均为浅克隆，位于主仓库之外，未修改全局 CUDA/Python 环境，未进行大规模训练。

| 项目 | 官方仓库 | 固定提交 | 最小验证 | 结论 |
|---|---|---|---|---|
| FALCON | `AsalMehradfar/FALCON` | `845145c0091fd10c3eaa0bc3d32ca95bfc519550` | 导入 `models.circuit_gnn` | 因隔离环境无 `torch_geometric` 阻塞；完成静态审计 |
| GraphGPS | `rampasek/GraphGPS` | `28015707cbab7f8ad72bed0ee872d068ea59c94b` | 导入 `GPSLayer` | 因隔离环境无 PyG/GraphGym 阻塞；不污染现有环境 |
| EGNN | `vgsatorras/egnn` | `e9ca6c0c3e1d30a7598efbd66034121b4af8dccc` | 6 节点前向+反向 | 成功，输出 `(6,4)`，输入梯度有限非零 |

## 适配判断

FALCON 最有价值的是电路关系分型与物理量作为边特征；GraphGPS 最有价值的是局部 MPNN 与全局注意力并行；EGNN 最有价值的是几何不变量编码。三者都不能直接替代本项目的电磁矩阵求解器和无源性约束。

EXP-016 推荐实现轻量自有骨干：分型局部消息传递 + GPS 式全局注意力 + 固定坐标的相对几何编码。禁止直接使用 EGNN 坐标更新改变真实绕组几何；禁止为跑通第三方仓库降级现有 PyTorch/CUDA。

完整命令及原始错误见 `commands/` 和 `logs/`。
