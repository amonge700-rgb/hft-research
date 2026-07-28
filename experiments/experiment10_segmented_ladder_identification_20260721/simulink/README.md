# 实验十 Simulink/Simscape 独立验证模型

本目录用于把 Python 中的四段高频变压器梯形矩阵模型，转换为由
Simscape DAE 求解器独立求解的物理网络。

第一步先编译自定义八绕组耦合电感：

```matlab
cd('D:\高频变压器\experiment10_segmented_ladder_identification_20260721\simulink')
build_custom_library
```

该元件采用实验十基线的结构化 `8×8` 电感矩阵：

```text
L = diag(Lleak) + Lcommon * a * a'
v = diag(R) i + L di/dt
```

它能够同时表达四个原边分段、四个副边分段及由公共磁通形成的全部互感，
并与当前 Python 基线矩阵数值等价。采用结构化参数是为了避免 MATLAB
R2021b 将 64 个矩阵元素全部作为独立符号参数时发生编译表达式膨胀。

后续脚本将在此元件外围连接纵向电容、对地电容、原副边分段耦合电容、
双端口激励及传感器，并保存可直接打开的 `.slx` 文件和扫频交叉验证结果。

## 当前编译诊断

自定义 `.ssc` 元件可以由 `ssc_build` 成功生成 `hftlib_lib.slx`，模型文件也能
成功保存。当前机器上通过命令行无界面模式执行
`set_param(model,'SimulationCommand','update')` 会阻塞；控制实验表明，标准
Simscape RC 模型和纯 Simulink 的 Sine-Gain-Out 模型也出现同样阻塞，因此
它不是本模型专属的方程或拓扑错误。

请在 MATLAB 桌面中执行：

```matlab
cd('D:\高频变压器\experiment10_segmented_ladder_identification_20260721\simulink')
open_system('hft_segmented_ladder_simscape.slx')
set_param('hft_segmented_ladder_simscape','SimulationCommand','update')
```

如果桌面更新返回错误，完整错误将用于继续定位；如果状态栏正常完成更新，
则可确认问题仅存在于本机的 headless/batch 编译路径。
