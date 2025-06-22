# MiniGPT 项目

MiniGPT 是一个简洁易用的微型 GPT 语言模型实现，适合学习和实验 Transformer 架构与文本生成。

## 🚀 快速开始

```bash
cd MiniGPT
pip install -r requirements.txt
python -m scripts.train --config small --dataset ag_news
python -m scripts.inference --model_path output/checkpoints/best_model.pth
python -m scripts.dataset_manager
```

## 📁 项目结构

```
MiniGPT/
├── core/         # 核心模型
├── data/         # 数据处理
├── scripts/      # 训练、推理、数据集管理
├── utils/        # 工具函数
├── configs/      # 配置文件
├── output/       # 训练输出
├── datasets/     # 数据集
├── README.md
└── requirements.txt
```

## 🧠 主要功能
- 支持多种真实开源数据集和模型配置
- 一站式训练、推理、数据集管理
- 训练历史与模型参数可视化

## 常用命令
- 训练模型：  `python -m scripts.train --config small --dataset ag_news`
- 推理生成：  `python -m scripts.inference --model_path output/checkpoints/best_model.pth`
- 数据集管理：`python -m scripts.dataset_manager`

## 依赖环境
- Python 3.7+
- PyTorch 1.8+
- 详见 requirements.txt

## 贡献与许可
欢迎贡献代码！请 Fork 后提交 Pull Request。
本项目采用 MIT 许可证。
如有问题请提交 Issue。 