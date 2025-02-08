
# SegmsNet

SegmsNet 可用于医学图像分割（2D 和 3D），该项目基于 PyTorch 实现，并提供了训练、验证和测试的完整流程。

## 项目结构

```
SegmsNet/
├── checkpoint/               # 保存训练好的模型权重
├── data/                     # 数据集存放目录
├── dataloader/               # 数据加载器模块
├── network/                  # 2D 分割网络模型
├── network_3d/               # 3D 分割网络模型
├── src/                      # 源代码目录
├── utils/                    # 工具函数和辅助脚本
├── datamodule_my.py          # 自定义数据模块
├── environment.yaml          # 环境配置文件
├── main_kvasir.py            # Kvasir 数据集的主程序
├── main.py                   # 2D 分割的主程序
├── main3d.py                 # 3D 分割的主程序
├── train.sh                  # 训练脚本
├── trainer.py                # 训练器模块
└── README.md                 # 项目说明文档
```

## 环境配置

### 1. 安装依赖
确保已安装 [Anaconda](https://www.anaconda.com/) 或 [Miniconda](https://docs.conda.io/en/latest/miniconda.html)，然后使用以下命令创建环境：

```bash
conda env create -f environment.yaml
conda activate uxnet3d
```

## 数据集准备


将数据集放置在 `data/` 目录下。支持的数据集包括：
- Kvasir 数据集,Busi数据集，TNSCUI数据集，ISIC18数据集，BTCV数据集


## 模型训练
运行train.sh脚本，其中包括所有数据的训练设置
