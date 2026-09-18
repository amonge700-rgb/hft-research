# HFT Research

高频变压器宽频建模、分段梯形网络、在线参数辨识与物理引导 AI 实验仓库。

## 当前研究主线

1. 每匝作为一个导体段，建立包含完整电感矩阵和分布电容矩阵的四端子模型；
2. 用 MNA/Kron 矩阵法计算端口频响和内部状态，并用 Simscape 独立验证；
3. 利用 DAB 正常 PWM 运行窗口研究端口、频点和参数子空间的可辨识性；
4. 以 Fisher/OED 选择在线可辨识坐标，不从少数端口直接自由恢复全部匝间参数；
5. 用 GNN 学习跨结构的匝级耦合先验或正向代理，再由物理矩阵完成闭环；
6. 将在线更新结果连接到跨绕组位移电流、内部电压与 EMI 风险等工程量。

## 实验导航

- [完整实验索引](EXPERIMENT_INDEX.md)
- [下一阶段实施路线](docs/NEXT_EXPERIMENT_PLAN.md)
- [PDF 报告合集与阅读顺序](reports/experiment_report_pdf_collection_20260908/README_报告索引.md)
- [参考文献与来源说明](literature/README.md)

### 当前核心实验（EXP-013 至 EXP-019）

- `experiment13_parametric_four_terminal_hft_20260821`：参数化四端子 Simscape 模型；
- `experiment14_dab_switching_20260821`：分布参数变压器接入 DAB 开关运行；
- `experiment15_hft_graph_physics_20260901`：统一图契约与可微矩阵/Kron 求解器；
- `experiment16_falcon_hft_gnn_20260909`：FALCON 风格 GNN 与 HFT 分型 GNN 基线；
- `experiment17_fisher_oed_subspace_20260909`：Fisher/OED、参数筛选与低秩可辨识子空间。
- `experiment18_local_hetero_gnn_20260910`：匝级正向异构 GNN 与 residual 消融；
- `experiment19_learnable_coupling_selection_20260910`：物理解码的跨绕组电容边选择与率失真分析。
- `experiment20_comsol_turn_matrix_teacher_20260915`：COMSOL 逐匝 Maxwell 电容矩阵与阻抗矩阵教师数据。
- `experiment21_comsol_gnn_minimal_20260916`：12 个参数化 COMSOL 匝级教师、DAB 风格多窗口观测、CUDA GNN 与低维物理反演闭环。
- `experiment22_model_mismatch_robust_inverse_20260916`：复杂前向/简化反演模型失配实验；证明未观测的测量链路与寄生变量会和内部参数混淆，当前 GNN 未超过纯物理反演。

当前最新实验节点为 EXP-022。它在复杂前向、简化反演条件下检验 AI 残差补偿，结果仍由纯物理反演取得最佳参数精度。下一步必须通过温度/测量链路校准量或额外端口观测解除参数混淆，而不是继续盲目扩大网络。

## 仓库目录

| 目录 | 内容 |
|---|---|
| `experiments/` | 各编号实验的代码、配置、关键结果、图表和报告源文件 |
| `reports/` | 适合直接阅读的 PDF 汇总与校验清单 |
| `literature/` | 文献索引、阅读说明及经明确授权纳入仓库的原文 |
| `docs/` | 跨实验路线、下一步计划和项目级说明 |
| `shared_data_manifest/` | 仓库外大型数据的清单和可追溯说明 |

## 重要边界

当前分段模型的参数主要来自文献尺度与合成设定。Simscape 验证证明两种实现的一致性，
不等同于证明模型已经准确对应某台实物高频变压器。物理参数仍需通过文献样机、有限元
或实测进行标定。

EXP-021 已建立首批 12 个参数化 COMSOL 匝级矩阵教师，但规模仍不足以支持跨结构泛化结论；当前仍未在实物 DAB 上验证在线辨识精度。

## 仓库规则

- 每项实验使用独立目录和编号；
- 保存代码、配置、随机种子、关键结果、图表和报告源文件；
- 大型原始数据、软件缓存和本地编译产物不进入 Git；
- 每个实验必须说明研究目的、假设、参数来源、验收指标、结果与局限；
- 稳定实验节点提交并推送到 GitHub，不对每个临时运行结果自动提交。

