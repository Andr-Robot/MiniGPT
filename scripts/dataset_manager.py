#!/usr/bin/env python3
"""
MiniGPT 数据集管理脚本
用于查看、下载和管理训练数据集
"""

import os
import sys
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from data.datasets import DatasetLoader


def list_datasets():
    """列出所有可用的数据集"""
    loader = DatasetLoader()
    datasets = loader.list_available_datasets()
    
    print("📚 可用数据集列表:")
    print("=" * 80)
    
    for name, config in datasets.items():
        info = loader.get_dataset_info(name)
        status = "✅ 已下载" if info['downloaded'] else "❌ 未下载"
        size_info = f"({info['file_size']})" if info['downloaded'] else f"({config['size']})"
        
        print(f"\n🔹 {name.upper()}")
        print(f"   描述: {config['description']}")
        print(f"   状态: {status} {size_info}")
        print(f"   类型: {config['type']}")
        
        if info['downloaded']:
            print(f"   路径: {info['filepath']}")


def download_dataset(dataset_name: str, force: bool = False):
    """下载指定数据集"""
    loader = DatasetLoader()
    
    if dataset_name not in loader.list_available_datasets():
        print(f"❌ 未知数据集: {dataset_name}")
        return
    
    try:
        print(f"📥 开始下载数据集: {dataset_name}")
        filepath = loader.download_dataset(dataset_name, force_download=force)
        print(f"✅ 下载完成: {filepath}")
        
        # 显示数据集信息
        info = loader.get_dataset_info(dataset_name)
        print(f"📊 文件大小: {info['file_size']}")
        
    except Exception as e:
        print(f"❌ 下载失败: {e}")


def preview_dataset(dataset_name: str, num_samples: int = 5):
    """预览数据集内容"""
    loader = DatasetLoader()
    
    if dataset_name not in loader.list_available_datasets():
        print(f"❌ 未知数据集: {dataset_name}")
        return
    
    try:
        print(f"👀 预览数据集: {dataset_name}")
        print("=" * 80)
        
        texts = loader.load_dataset(dataset_name, max_samples=num_samples)
        
        for i, text in enumerate(texts, 1):
            print(f"\n📄 样本 {i}:")
            print(f"长度: {len(text)} 字符")
            print(f"内容: {text[:200]}{'...' if len(text) > 200 else ''}")
            print("-" * 40)
        
        print(f"\n📊 总样本数: {len(texts)}")
        
    except Exception as e:
        print(f"❌ 预览失败: {e}")


def test_dataset_loading(dataset_name: str):
    """测试数据集加载"""
    loader = DatasetLoader()
    
    if dataset_name not in loader.list_available_datasets():
        print(f"❌ 未知数据集: {dataset_name}")
        return
    
    try:
        print(f"🧪 测试数据集加载: {dataset_name}")
        print("=" * 80)
        
        # 加载数据集
        texts = loader.load_dataset(dataset_name)
        
        # 统计信息
        total_chars = sum(len(text) for text in texts)
        avg_length = total_chars / len(texts) if texts else 0
        
        print(f"📊 数据集统计:")
        print(f"   样本数量: {len(texts):,}")
        print(f"   总字符数: {total_chars:,}")
        print(f"   平均长度: {avg_length:.1f} 字符")
        print(f"   最短样本: {min(len(text) for text in texts)} 字符")
        print(f"   最长样本: {max(len(text) for text in texts)} 字符")
        
        # 显示前几个样本的预览
        print(f"\n📄 前3个样本预览:")
        for i, text in enumerate(texts[:3], 1):
            print(f"   样本 {i}: {text[:100]}{'...' if len(text) > 100 else ''}")
        
        print("✅ 数据集加载测试完成")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")


def create_training_config(dataset_name: str):
    """为指定数据集创建训练配置"""
    loader = DatasetLoader()
    
    if dataset_name not in loader.list_available_datasets():
        print(f"❌ 未知数据集: {dataset_name}")
        return
    
    # 根据数据集类型生成配置
    configs = {
        'tiny_shakespeare': {
            'dataset': 'tiny_shakespeare',
            'tokenizer_type': 'char',
            'vocab_size': 100,
            'max_samples': 2000,
            'batch_size': 32,
            'max_length': 128,
            'd_model': 256,
            'n_layers': 6,
            'n_heads': 8,
            'd_ff': 1024,
            'dropout': 0.1,
            'learning_rate': 1e-4,
            'weight_decay': 0.01,
            'device': 'auto'
        },
        'tiny_stories': {
            'dataset': 'tiny_stories',
            'tokenizer_type': 'word',
            'vocab_size': 2000,
            'max_samples': 5000,
            'batch_size': 32,
            'max_length': 128,
            'd_model': 256,
            'n_layers': 6,
            'n_heads': 8,
            'd_ff': 1024,
            'dropout': 0.1,
            'learning_rate': 1e-4,
            'weight_decay': 0.01,
            'device': 'auto'
        },
        'wikitext_103': {
            'dataset': 'wikitext_103',
            'tokenizer_type': 'word',
            'vocab_size': 5000,
            'max_samples': 10000,
            'batch_size': 32,
            'max_length': 128,
            'd_model': 256,
            'n_layers': 6,
            'n_heads': 8,
            'd_ff': 1024,
            'dropout': 0.1,
            'learning_rate': 1e-4,
            'weight_decay': 0.01,
            'device': 'auto'
        },
        'ag_news': {
            'dataset': 'ag_news',
            'tokenizer_type': 'word',
            'vocab_size': 5000,
            'max_samples': 5000,
            'batch_size': 32,
            'max_length': 128,
            'd_model': 256,
            'n_layers': 6,
            'n_heads': 8,
            'd_ff': 1024,
            'dropout': 0.1,
            'learning_rate': 1e-4,
            'weight_decay': 0.01,
            'device': 'auto'
        }
    }
    
    config = configs.get(dataset_name, {})
    
    print(f"⚙️ 为数据集 {dataset_name} 生成的训练配置:")
    print("=" * 80)
    
    import json
    print(json.dumps(config, indent=2, ensure_ascii=False))
    
    # 保存配置到文件
    config_file = f"config_{dataset_name}.json"
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 配置已保存到: {config_file}")
    print(f"🚀 使用方法: python train.py --config {config_file}")


def main():
    parser = argparse.ArgumentParser(description='MiniGPT 数据集管理工具')
    parser.add_argument('action', choices=['list', 'download', 'preview', 'test', 'config'],
                       help='要执行的操作')
    parser.add_argument('--dataset', type=str, help='数据集名称')
    parser.add_argument('--force', action='store_true', help='强制重新下载')
    parser.add_argument('--samples', type=int, default=5, help='预览样本数量')
    
    args = parser.parse_args()
    
    if args.action == 'list':
        list_datasets()
    elif args.action == 'download':
        if not args.dataset:
            print("❌ 请指定要下载的数据集名称")
            return
        download_dataset(args.dataset, args.force)
    elif args.action == 'preview':
        if not args.dataset:
            print("❌ 请指定要预览的数据集名称")
            return
        preview_dataset(args.dataset, args.samples)
    elif args.action == 'test':
        if not args.dataset:
            print("❌ 请指定要测试的数据集名称")
            return
        test_dataset_loading(args.dataset)
    elif args.action == 'config':
        if not args.dataset:
            print("❌ 请指定要生成配置的数据集名称")
            return
        create_training_config(args.dataset)


if __name__ == "__main__":
    main() 