import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os
import json
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import numpy as np
import argparse
import traceback

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from core.mini_gpt import MiniGPT, count_parameters
from data.tokenizer import SimpleTokenizer, CharTokenizer, BPETokenizer
from data.datasets import DatasetLoader
from utils.utils import visualize_attention_weights, plot_parameter_distribution


class Trainer:
    """MiniGPT训练器"""
    
    def __init__(self, model: MiniGPT, tokenizer, train_dataloader: DataLoader,
                 val_dataloader: Optional[DataLoader] = None,
                 learning_rate: float = 1e-4, weight_decay: float = 0.01,
                 device: str = 'auto'):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = self._get_device(device)
        
        # 将模型移到设备
        self.model.to(self.device)
        
        # 优化器和学习率调度器
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=len(train_dataloader)
        )
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        
    def _get_device(self, device: str) -> torch.device:
        """获取训练设备"""
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)
    
    def train_epoch(self) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_dataloader)
        
        progress_bar = tqdm(self.train_dataloader, desc="训练中")
        
        for batch_idx, (input_ids, target_ids) in enumerate(progress_bar):
            # 移到设备
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            logits, loss = self.model(input_ids, target_ids)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # 优化器步进
            self.optimizer.step()
            self.scheduler.step()
            
            # 记录损失
            total_loss += loss.item()
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.6f}'
            })
            
            # 记录学习率
            self.learning_rates.append(self.scheduler.get_last_lr()[0])
        
        return total_loss / num_batches
    
    def validate(self) -> float:
        """验证模型"""
        if self.val_dataloader is None:
            return 0.0
            
        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.val_dataloader)
        
        with torch.no_grad():
            for input_ids, target_ids in tqdm(self.val_dataloader, desc="验证中"):
                input_ids = input_ids.to(self.device)
                target_ids = target_ids.to(self.device)
                
                logits, loss = self.model(input_ids, target_ids)
                total_loss += loss.item()
        
        return total_loss / num_batches
    
    def train(self, num_epochs: int, save_dir: str = 'output/checkpoints',
              save_every: int = 5, early_stopping_patience: int = 10) -> Dict:
        """完整训练循环"""
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"开始训练，设备: {self.device}")
        print(f"模型参数数量: {count_parameters(self.model):,}")
        print(f"训练样本数量: {len(self.train_dataloader.dataset)}")
        if self.val_dataloader:
            print(f"验证样本数量: {len(self.val_dataloader.dataset)}")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 50)
            
            # 训练
            start_time = time.time()
            train_loss = self.train_epoch()
            train_time = time.time() - start_time
            
            # 验证
            val_loss = self.validate()
            
            # 记录历史
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # 打印结果
            print(f"训练损失: {train_loss:.4f}")
            print(f"验证损失: {val_loss:.4f}")
            print(f"训练时间: {train_time:.2f}秒")
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_model(os.path.join(save_dir, 'best_model.pth'))
                print("保存最佳模型")
            else:
                patience_counter += 1
            
            # 定期保存检查点
            if (epoch + 1) % save_every == 0:
                checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch + 1}.pth')
                self.save_model(checkpoint_path)
                print(f"保存检查点: {checkpoint_path}")
            
            # 早停
            if patience_counter >= early_stopping_patience:
                print(f"早停触发，{early_stopping_patience}个epoch没有改善")
                break
        
        # 保存训练历史
        self.save_training_history(save_dir)
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
            'best_val_loss': best_val_loss
        }
    
    def save_model(self, filepath: str):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'tokenizer': self.tokenizer,
            'model_config': {
                'vocab_size': self.model.vocab_size,
                'd_model': self.model.d_model,
                'n_layers': len(self.model.transformer_blocks),
                'n_heads': self.model.transformer_blocks[0].attention.n_heads,
                'd_ff': self.model.transformer_blocks[0].feed_forward.linear1.out_features,
                'max_len': self.model.position_encoding.pe.size(0),
                'dropout': self.model.dropout.p
            }
        }, filepath)
    
    def load_model(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.tokenizer = checkpoint['tokenizer']
        print(f"模型已从 {filepath} 加载")
    
    def save_training_history(self, save_dir: str):
        """保存训练历史"""
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates
        }
        
        with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=2)
    
    def plot_training_history(self, save_dir: str = 'output/plots'):
        """Plot training history (all labels/titles in English)"""
        os.makedirs(save_dir, exist_ok=True)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss curve
        ax1.plot(self.train_losses, label='Train Loss', color='blue')
        if self.val_losses:
            ax1.plot(self.val_losses, label='Validation Loss', color='red')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Learning rate curve
        ax2.plot(self.learning_rates, color='green')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_sample(self, prompt: str, max_length: int = 50, 
                       temperature: float = 1.0, top_k: int = 50) -> str:
        """生成示例文本"""
        self.model.eval()
        
        # 编码提示（不添加EOS token）
        prompt_ids = self.tokenizer.encode(prompt, add_eos=False)
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long).to(self.device)
        
        # 计算需要生成的新token数量
        tokens_to_generate = max_length - len(prompt_ids)
        
        if tokens_to_generate <= 0:
            # 增加生成长度
            max_length = len(prompt_ids) + 20
            tokens_to_generate = 20
        
        # 生成
        with torch.no_grad():
            generated_ids = self.model.generate(
                prompt_tensor, 
                max_length=max_length,
                temperature=temperature,
                top_k=top_k
            )
        
        # 解码
        generated_text = self.tokenizer.decode(generated_ids[0].tolist())
        
        return generated_text


def create_trainer_from_config(config: Dict) -> Trainer:
    """从配置创建训练器"""
    
    # 创建分词器
    if config['tokenizer_type'] == 'word':
        tokenizer = SimpleTokenizer(vocab_size=config['vocab_size'])
    elif config['tokenizer_type'] == 'char':
        tokenizer = CharTokenizer(vocab_size=config['vocab_size'])
    elif config['tokenizer_type'] == 'bpe':
        tokenizer = BPETokenizer(vocab_size=config['vocab_size'])
    else:
        raise ValueError(f"不支持的分词器类型: {config['tokenizer_type']}")
    
    # 创建数据集
    dataset_name = config.get('dataset', 'ag_news')  # 默认使用AG News数据集
    
    # 使用DatasetLoader加载真实数据集
    loader = DatasetLoader()
    
    if dataset_name == 'curated':
        # 使用精选数据集（混合多个数据集）
        from data.datasets import create_curated_dataset
        all_texts = create_curated_dataset()
    elif dataset_name in ['tiny_shakespeare', 'tiny_stories', 'ag_news']:
        # 使用单个真实数据集
        all_texts = loader.load_dataset(dataset_name, config.get('max_samples', None))
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")
    
    # 分割训练和验证集
    split_idx = int(len(all_texts) * 0.8)
    train_texts = all_texts[:split_idx]
    val_texts = all_texts[split_idx:]
    
    # 训练分词器
    tokenizer.fit(train_texts + val_texts)
    
    # 创建简单的数据加载器
    from torch.utils.data import Dataset, DataLoader
    
    class SimpleTextDataset(Dataset):
        def __init__(self, texts, tokenizer, max_length):
            self.texts = texts
            self.tokenizer = tokenizer
            self.max_length = max_length
            
            # 编码所有文本
            self.encoded_texts = []
            for text in texts:
                encoded = tokenizer.encode(text, max_length=max_length)
                if len(encoded) > 2:  # 至少要有开始、结束和内容token
                    self.encoded_texts.append(encoded)
        
        def __len__(self):
            return len(self.encoded_texts)
        
        def __getitem__(self, idx):
            encoded = self.encoded_texts[idx]
            # 创建输入和目标（用于语言模型训练）
            input_ids = encoded[:-1]
            target_ids = encoded[1:]
            return torch.tensor(input_ids, dtype=torch.long), torch.tensor(target_ids, dtype=torch.long)
    
    # 创建数据集和数据加载器
    train_dataset = SimpleTextDataset(train_texts, tokenizer, config['max_length'])
    val_dataset = SimpleTextDataset(val_texts, tokenizer, config['max_length'])
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        drop_last=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        drop_last=True
    )
    
    # 创建模型
    vocab_size = len(tokenizer.word2idx) if hasattr(tokenizer, 'word2idx') else len(tokenizer.char2idx)
    model = MiniGPT(
        vocab_size=vocab_size,
        d_model=config['d_model'],
        n_layers=config['n_layers'],
        n_heads=config['n_heads'],
        d_ff=config['d_ff'],
        max_len=config['max_length'],
        dropout=config['dropout']
    )
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        device=config['device']
    )
    
    return trainer


def load_config(config_name: str) -> Dict:
    """
    从配置文件加载配置
    
    Args:
        config_name: 配置文件名（不包含.json扩展名）
        
    Returns:
        配置字典
    """
    config_path = os.path.join(project_root, 'configs', f'{config_name}.json')
    
    if os.path.exists(config_path):
        print(f"📁 从配置文件加载: {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        # 合并模型配置和训练配置
        config = {
            **config_data['model_config'],
            **config_data['training_config']
        }
        
        print(f"✅ 配置加载成功: {config_data['name']}")
        print(f"📝 描述: {config_data['description']}")
        print(f"💡 用途: {config_data['use_case']}")
        print(f"📊 预估参数数量: {config_data['estimated_params']:,}")
        
        return config
    else:
        print(f"⚠️  配置文件不存在: {config_path}")
        print("🔧 使用默认配置...")
        return None


def get_available_configs() -> List[str]:
    """获取可用的配置文件列表"""
    config_dir = os.path.join(project_root, 'configs')
    if not os.path.exists(config_dir):
        return []
    
    configs = []
    for file in os.listdir(config_dir):
        if file.endswith('.json'):
            configs.append(file[:-5])  # 移除.json扩展名
    
    return sorted(configs)


def get_available_datasets() -> List[str]:
    """获取可用的真实开源数据集列表"""
    return [
        'ag_news',          # AG News新闻数据集
        'tiny_shakespeare', # 莎士比亚文本
        'tiny_stories',     # 小故事数据集
        'curated'           # 精选混合数据集
    ]


def train_model(config: Dict, config_name: str):
    """
    使用给定配置训练模型
    
    Args:
        config: 训练配置
        config_name: 配置名称
    """
    print(f"\n🚀 MiniGPT训练配置: {config_name}")
    print("=" * 60)
    print(json.dumps(config, indent=2, ensure_ascii=False))
    print("=" * 60)
    
    # 创建训练器
    trainer = create_trainer_from_config(config)
    
    # 获取训练参数
    num_epochs = config.get('num_epochs', 10)
    save_every = config.get('save_every', 2)
    early_stopping_patience = config.get('early_stopping_patience', 3)
    
    # 开始训练
    print(f"\n🔥 开始训练...")
    print(f"📊 训练轮数: {num_epochs}")
    print(f"💾 保存频率: 每{save_every}轮")
    print(f"⏹️  早停耐心值: {early_stopping_patience}")
    
    history = trainer.train(
        num_epochs=num_epochs,
        save_dir='output/checkpoints',
        save_every=save_every,
        early_stopping_patience=early_stopping_patience
    )
    
    # 绘制训练历史
    trainer.plot_training_history()
    
    # 生成示例
    print("\n" + "="*50)
    print("生成示例测试")
    print("="*50)

    # 针对不同数据集选择不同的测试prompt
    dataset_name = config.get('dataset', 'ag_news')
    if dataset_name == 'ag_news':
        prompts = [
            "The president announced",
            "Stock market rises",
            "Scientists discovered",
            "In a recent study",
            "Technology news"
        ]
    elif dataset_name == 'tiny_shakespeare':
        prompts = [
            "To be, or not to be",
            "My lord",
            "What say you",
            "love and honor",
            "O Romeo"
        ]
    elif dataset_name == 'tiny_stories':
        prompts = [
            "Once upon a time",
            "There was a little cat",
            "The boy found a magic stone",
            "In a small village",
            "The adventure begins"
        ]
    elif dataset_name == 'curated':
        prompts = [
            "To be, or not to be",
            "The president announced",
            "Once upon a time",
            "Technology news",
            "love and honor"
        ]
    else:
        prompts = [
            "world",
            "machine learning",
            "artificial intelligence",
            "technology",
            "business"
        ]

    for prompt in prompts:
        print(f"\n--- 测试提示: '{prompt}' ---")
        try:
            prompt_ids = trainer.tokenizer.encode(prompt, add_eos=False)
            if 1 in prompt_ids:
                print("⚠️  警告：提示词中包含 <UNK>，建议换词或增大词表！")
                print(f"   编码结果: {prompt_ids}")

            generated = trainer.generate_sample(
                prompt, 
                max_length=len(prompt_ids) + 20,
                temperature=0.8,
                top_k=20
            )
            print(f"最终结果: '{generated}'")
        except Exception as e:
            print(f"❌ 生成失败: {e}")
        print("-" * 50)


def main():
    parser = argparse.ArgumentParser(description='MiniGPT 训练脚本')
    parser.add_argument('--config', type=str, default='small', 
                       help='配置文件名称 (small/medium/large)')
    parser.add_argument('--dataset', type=str, default='sample',
                       help='数据集名称 (sample/ag_news/tiny_shakespeare/tiny_stories/curated)')
    parser.add_argument('--force', action='store_true',
                       help='强制使用默认配置，忽略配置文件')
    parser.add_argument('--help-configs', action='store_true',
                       help='显示可用的配置文件')
    parser.add_argument('--help-datasets', action='store_true',
                       help='显示可用的数据集')
    
    args = parser.parse_args()
    
    # 显示帮助信息
    if args.help_configs:
        print("📋 可用的配置文件:")
        configs = get_available_configs()
        for config in configs:
            print(f"   - {config}")
        return
    
    if args.help_datasets:
        print("📋 可用的数据集:")
        datasets = get_available_datasets()
        for dataset in datasets:
            print(f"   - {dataset}")
        return
    
    # 验证数据集
    available_datasets = get_available_datasets()
    if args.dataset not in available_datasets:
        print(f"❌ 未找到数据集: {args.dataset}")
        print("📋 可用的数据集:")
        for dataset in available_datasets:
            print(f"   - {dataset}")
        exit(1)
    
    print(f"🚀 开始训练 MiniGPT")
    print(f"📊 数据集: {args.dataset}")
    print(f"⚙️  配置: {args.config}")
    print("=" * 50)
    
    # 加载配置
    if not args.force:
        config = load_config(args.config)
        if config is None:
            print("❌ 配置文件加载失败")
            exit(1)
    else:
        # 强制使用默认配置
        print("🔧 使用默认配置...")
        configs = get_available_configs()
        if args.config in configs:
            config = load_config(args.config)
            if config is None:
                print("❌ 默认配置加载失败")
                exit(1)
        else:
            print(f"❌ 未找到默认配置: {args.config}")
            print("📋 可用的配置:")
            for name in configs:
                print(f"   - {name}")
            exit(1)
    
    # 添加数据集信息到配置中
    config['dataset'] = args.dataset
    config_name = f"{args.config} + {args.dataset}"
    
    # 开始训练
    try:
        train_model(config, config_name)
        print("✅ 训练完成!")
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main() 