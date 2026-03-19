# IPNet: An Interaction Pattern-aware Neural Network for Temporal Link Prediction

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/) [![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/) [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

IPNet 是面向动态社会网络时序链接预测的轻量级深度学习模型，通过挖掘节点交互模式、融合时序上下文特征提升预测精度，支持直推式（Transductive）和归纳式（Inductive）两种任务场景。

# 目录

- [环境配置](#环境配置)
- [数据准备](#数据准备)
- [快速运行](#快速运行)
- [结果输出](#结果输出)
- [模型结构](#模型结构)
- [论文引用](#论文引用)

# 环境配置

## 1. 克隆仓库

```bash
git clone https://github.com/ACPasser/IPNet.git
cd IPNet
```

## 2. 创建虚拟环境（推荐）

```bash
# Conda 创建环境
conda create -n ipnet python=3.10
conda activate ipnet

# 或 venv 创建环境
python -m venv ipnet-env
source ipnet-env/bin/activate  # Linux/Mac
# ipnet-env\Scripts\activate  # Windows
```

## 3. 安装依赖

```bash
# 安装核心依赖
pip install -r requirements.txt
```

## 4. 依赖清单（requirements.txt）

```bash
gensim==4.4.0
joblib==1.5.3
networkx==3.6.1
numpy==2.4.3
pandas==3.0.1
scikit_learn==1.8.0
torch==2.10.0
tqdm==4.67.3
```

## 5. 设备支持

| 设备类型              | 参数配置      | 说明                        |
| --------------------- | ------------- | --------------------------- |
| CPU                   | `--device -2` | 强制使用 CPU                |
| Apple MPS（M1/M2/M3） | `--device -1` | 自动适配 Apple Silicon 加速 |
| NVIDIA GPU            | `--device 0`  | 指定 GPU 卡号（0/1/2...）   |

# 数据准备

1. 数据集需放在 `data/[数据集名称]/` 目录下，文件结构：

```plaintext
data/
└── UCI/  # 数据集名称
    ├── 0.origin/
    │   └── graph.txt  # 核心列："源节点ID"、"目标节点ID"、"时间戳"（空白字符分隔）
    └── preprocess.py/	# 预处理脚本
```

2. 仓库内置 IA、UCI 两个小的公开数据集以及自建的 ZhiHu 数据集（仅提供预处理后的 csv 格式文件）；自定义数据集需遵循上述格式并参考预处理脚本。

# 快速运行

基础运行命令

```bash
# 默认参数（UCI数据集，直推式任务，mean聚合版本，cpu训练）
python main.py

# 归纳式任务（指定序列长度、游走次数、游走长度，NVIDIA GPU加速）
python main.py --dataset UCI --ty I --mask 0.15 --il 20 --wn 20 --wl 20 --device 0
```

# 结果输出

1. **模型保存**（可在 data/config.py 中修改默认配置）：
   1. 最优模型的参数字典保存至：`outputs/[数据集]/best_models/[时间戳]/best-model.pth`；
   2. 初始化时的参数保存至目录：`outputs/[数据集]/model_param/{时间戳}` 下，包括节点特征、交互序列、上下文窗口及其他参数；
2. **训练日志**：暂时仅输出到终端，输出实验细节、每轮训练 / 验证及最终测试的 Acc、AUC、AP、F1 指标等；
3. **结果文件**：测试结果保存至：`outputs/[数据集]/results/[核心参数组合]/IPNet-[版本].csv`，包含以下核心字段：
   - Training_Date：实验运行日期
   - Task_Type：任务类型（T/I）
   - Acc/AUC/AP/F1：测试集评估指标（百分比）
   - Time (s)：总运行时间
   - Seed：随机种子
   - Best_Model_Path：最优模型的参数字典保存路径

# 模型结构

IPNet 主要由 4 个核心模块构成，实现「交互模式建模 - 上下文特征融合 - 链接预测」的端到端流程：

1. **InteractionSequenceEncoder**：对节点交互序列进行时间 + 位置编码；
2. **SequenceFeatureAggregator**：通过 RNN 捕捉交互序列特征；
3. **ContextEncoder/ContextualFeatureAggregator**：建模时序上下文特征（支持注意力聚合）；
4. **MergeLayer**：融合源 / 目标节点特征，输出链接预测分数。

# 论文引用

如果该代码对你的研究有帮助，请引用：

```bibtex
@inproceedings{10.1145/3746252.3761063,
    author = {Zhang, Qingyang and Wang, Yitong and Lin, Xinjie},
    title = {IPNet: An Interaction Pattern-aware Neural Network for Temporal Link Prediction},
    year = {2025},
    doi = {10.1145/3746252.3761063},
    booktitle = {Proceedings of the 34th ACM International Conference on Information and Knowledge Management},
    pages = {4160–4169},
    numpages = {10}
}
```

