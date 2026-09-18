# EXP-024：Fisher可辨识逐匝图模态

比较四种反演：56条逐边自由度、5个工程区域分组、8个Fisher主模态，以及已知真实变化边的oracle 8维上限。目标是判断EXP-023中互感边定位失败是否来自观测不可辨识，而不是网络容量不足。

```powershell
python run_exp24.py --cases 60 --seeds 5
```
