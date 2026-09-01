# 审计命令说明

三个项目分别使用独立 `venv --system-site-packages`，共享已验证 Torch/CUDA，仅执行导入或微型前反向，不安装额外包、不训练、不生成数据。

```powershell
python -c "import models.circuit_gnn"
python -c "from graphgps.layer.gps_layer import GPSLayer"
python -c "import torch; from models.egnn_clean.egnn_clean import EGNN; ..."
```
