# EXP-017: Fisher/OED 与可辨识参数子空间

本实验承接 EXP-015 精确矩阵/Kron 物理教师和 EXP-016 八参数反演失败分析。目标不是继续堆叠 GNN，而是回答三个问题：

1. `Y11/Y22/Y12` 的哪些频率最有辨识信息？
2. 八个结构化尺度是否都能稳定反演？
3. 原参数筛选与 Fisher 主子空间降维，哪一种更能抵抗噪声？

运行：

```powershell
D:\HFT_AI\envs\hft-ident\python.exe run_exp17.py --device cuda --cases 100
D:\HFT_AI\envs\hft-ident\python.exe test_exp17.py
```

数据完全来自合成的 8+8 分段参考模型。没有调用 COMSOL、实物测量或 DAB 开关窗口，因此结论只针对当前局部模型和端口观测。
