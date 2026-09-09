# EXP-016：FALCON 风格匝级图网络与物理闭环

本实验是方法链路试验，不是实物高频变压器结论。数据来自 EXP-015 的透明 4+4、8+8 匝无源参考网络，经 8 个正值对数尺度参数扰动生成。

比较两种纯 PyTorch 图模型：

1. `falcon_homogeneous`：借鉴 FALCON 的同质消息传递、全局平均池化和图级性能预测；
2. `hft_typed`：分别处理串联、电磁耦合和电容耦合三类关系，并采用原边、次边及端口感知池化。

精确物理教师仍采用 EXP-015 的矩阵方程

\[
Y_n=A(R+j\omega L)^{-1}A^T+G+j\omega C
\]

以及 Kron 消元。参数化采用正值尺度，保证电阻为正、电感矩阵正定、电容矩阵半正定。最后使用可微矩阵求解器从端口频响反演 8 个结构化尺度参数，验证“图代理—物理矩阵—逆问题”闭环。

运行：

```powershell
D:\HFT_AI\envs\hft-ident\python.exe run_exp16.py --device cuda
```

快速自检：

```powershell
D:\HFT_AI\envs\hft-ident\python.exe run_exp16.py --quick --device cuda
D:\HFT_AI\envs\hft-ident\python.exe test_exp16.py
```

输出保存在 `results/`：权重、训练历史和指标 JSON。当前版本没有 COMSOL 或实测标签，因此不能宣称跨样机泛化或在线匝间矩阵唯一辨识。
