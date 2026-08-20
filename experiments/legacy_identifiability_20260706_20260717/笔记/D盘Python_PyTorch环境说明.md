# D盘 Python / PyTorch 环境说明

时间：2026-07-11

## 安装原则

本机 AI / Python 工具统一放在 D 盘，避免占用 C 盘。

```text
D:\HFT_AI\miniforge3     conda 管理器
D:\HFT_AI\envs           conda 环境目录
D:\HFT_AI\pkgs           conda 包缓存
D:\HFT_AI\pip-cache      pip 下载缓存
```

不污染 `base` 环境。每个任务单独创建 conda environment。

## 当前项目环境

环境路径：

```text
D:\HFT_AI\envs\hft-ident
```

Python：

```text
Python 3.11.15
```

核心包：

```text
torch 2.11.0+cu128
torchvision 0.26.0+cu128
torchaudio 2.11.0+cu128
numpy 2.4.4
scipy 1.17.1
pandas 3.0.3
matplotlib 3.11.0
scikit-learn 1.9.0
scikit-rf 2.0.1
gpytorch 1.15.2
botorch 0.18.1
jupyter 1.1.1
h5py 3.16.0
pymatreader 1.2.3
```

GPU 验证：

```text
torch.cuda.is_available() = True
CUDA runtime = 12.8
GPU = NVIDIA GeForce RTX 5070
```

## 推荐使用方式

方式一：直接用环境里的 Python，不需要激活：

```powershell
D:\HFT_AI\envs\hft-ident\python.exe your_script.py
```

方式二：激活环境：

```powershell
D:\HFT_AI\miniforge3\Scripts\activate.bat D:\HFT_AI\envs\hft-ident
```

或者在 PowerShell 中：

```powershell
& D:\HFT_AI\miniforge3\Scripts\conda.exe activate D:\HFT_AI\envs\hft-ident
```

如果当前 shell 没有初始化 conda，推荐使用方式一，最稳。

## pip 安装新包

为了避免 pip 缓存进入 C 盘，使用：

```powershell
$env:PIP_CACHE_DIR='D:\HFT_AI\pip-cache'
D:\HFT_AI\envs\hft-ident\python.exe -m pip install package_name
```

## 以后新建环境模板

例如新建一个 Bayesian OED 专用环境：

```powershell
D:\HFT_AI\miniforge3\Scripts\conda.exe create -p D:\HFT_AI\envs\hft-oed python=3.11 pip -y
$env:PIP_CACHE_DIR='D:\HFT_AI\pip-cache'
D:\HFT_AI\envs\hft-oed\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
```

原则：

```text
不要在 base 里装实验包。
不要把 D:\HFT_AI\envs\hft-ident\Scripts 加入系统 PATH。
每个方向单独环境，需要什么再装什么。
```

