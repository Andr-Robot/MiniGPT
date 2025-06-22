import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
import json
import os
import sys
from collections import defaultdict

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from core.mini_gpt import MiniGPT, count_parameters


def analyze_model_parameters(model: MiniGPT) -> Dict:
    """Analyze model parameter distribution"""
    param_info = {
        'total_params': 0,
        'trainable_params': 0,
        'param_groups': defaultdict(int),
        'layer_params': {}
    }
    
    for name, param in model.named_parameters():
        num_params = param.numel()
        param_info['total_params'] += num_params
        
        if param.requires_grad:
            param_info['trainable_params'] += num_params
        
        # 按参数类型分组
        if 'weight' in name:
            param_info['param_groups']['weights'] += num_params
        elif 'bias' in name:
            param_info['param_groups']['biases'] += num_params
        else:
            param_info['param_groups']['others'] += num_params
        
        # 按层分组
        layer_name = name.split('.')[0]
        if layer_name not in param_info['layer_params']:
            param_info['layer_params'][layer_name] = 0
        param_info['layer_params'][layer_name] += num_params
    
    return param_info


def visualize_attention_weights(model: MiniGPT, input_text: str, tokenizer, 
                               layer_idx: int = 0, head_idx: int = 0) -> None:
    """Visualize attention weights"""
    model.eval()
    
    # 编码输入（不添加EOS token）
    input_ids = tokenizer.encode(input_text, add_eos=False)
    input_tensor = torch.tensor([input_ids], dtype=torch.long)
    
    # 获取注意力权重
    with torch.no_grad():
        # 前向传播到指定层
        token_embeddings = model.token_embedding(input_tensor) * (model.d_model ** 0.5)
        x = model.position_encoding(token_embeddings.transpose(0, 1)).transpose(0, 1)
        x = model.dropout(x)
        
        # 通过指定层
        for i, transformer_block in enumerate(model.transformer_blocks):
            if i == layer_idx:
                # 获取注意力权重
                attn_output, attention_weights = transformer_block.attention(x)
                break
            x = transformer_block(x)
    
    # 提取指定头的注意力权重
    attention_matrix = attention_weights[0, head_idx].cpu().numpy()
    
    # 获取token
    tokens = [str(tokenizer.idx2word.get(idx, f'<{idx}>')) for idx in input_ids]
    
    # 绘制热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention_matrix, 
                xticklabels=tokens, 
                yticklabels=tokens,
                cmap='Blues',
                annot=True,
                fmt='.2f',
                cbar_kws={'label': 'Attention Weight'})
    
    plt.title(f'Layer {layer_idx} Head {head_idx} Attention Weights')
    plt.xlabel('Key')
    plt.ylabel('Query')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


def plot_parameter_distribution(model: MiniGPT, save_path: Optional[str] = None):
    """Plot parameter distribution (all info in English)"""
    param_info = analyze_model_parameters(model)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Parameter type distribution
    param_groups = param_info['param_groups']
    ax1.pie(param_groups.values(), labels=param_groups.keys(), autopct='%1.1f%%')
    ax1.set_title('Parameter Type Distribution')
    
    # Layer parameter distribution
    layer_params = param_info['layer_params']
    layers = list(layer_params.keys())
    params = list(layer_params.values())
    ax2.bar(layers, params)
    ax2.set_title('Parameter Count per Layer')
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Parameter Count')
    ax2.tick_params(axis='x', rotation=45)
    
    # Weight and bias value histogram
    weight_params = []
    bias_params = []
    for name, param in model.named_parameters():
        if 'weight' in name:
            weight_params.extend(param.data.cpu().numpy().flatten())
        elif 'bias' in name:
            bias_params.extend(param.data.cpu().numpy().flatten())
    ax3.hist(weight_params, bins=50, alpha=0.7, label='Weights', density=True)
    ax3.hist(bias_params, bins=50, alpha=0.7, label='Biases', density=True)
    ax3.set_title('Parameter Value Distribution')
    ax3.set_xlabel('Value')
    ax3.set_ylabel('Density')
    ax3.legend()
    
    # Parameter statistics (English)
    stats_text = f"""
    Total parameters: {param_info['total_params']:,}
    Trainable parameters: {param_info['trainable_params']:,}
    Weight parameters: {param_groups['weights']:,}
    Bias parameters: {param_groups['biases']:,}
    """
    ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes, fontsize=12,
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat'))
    ax4.set_title('Parameter Statistics')
    ax4.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def compute_model_complexity(model: MiniGPT, input_size: Tuple[int, int]) -> Dict:
    """计算模型复杂度（FLOPs）"""
    batch_size, seq_len = input_size
    
    # 计算每层的FLOPs
    flops_info = {}
    
    # 嵌入层
    embedding_flops = batch_size * seq_len * model.d_model
    flops_info['embedding'] = embedding_flops
    
    # 位置编码
    pos_encoding_flops = batch_size * seq_len * model.d_model
    flops_info['position_encoding'] = pos_encoding_flops
    
    # Transformer层
    transformer_flops = 0
    for i, transformer_block in enumerate(model.transformer_blocks):
        # 自注意力
        # Q, K, V 投影
        qkv_flops = 3 * batch_size * seq_len * model.d_model * model.d_model
        # 注意力计算
        attention_flops = batch_size * model.transformer_blocks[0].attention.n_heads * seq_len * seq_len * (model.d_model // model.transformer_blocks[0].attention.n_heads)
        # 输出投影
        output_flops = batch_size * seq_len * model.d_model * model.d_model
        
        # 前馈网络
        ff_flops = 2 * batch_size * seq_len * model.d_model * transformer_block.feed_forward.linear1.out_features
        
        layer_flops = qkv_flops + attention_flops + output_flops + ff_flops
        transformer_flops += layer_flops
        flops_info[f'transformer_layer_{i}'] = layer_flops
    
    flops_info['transformer_total'] = transformer_flops
    
    # 输出层
    output_flops = batch_size * seq_len * model.d_model * model.vocab_size
    flops_info['output_projection'] = output_flops
    
    # 总FLOPs
    total_flops = sum(flops_info.values())
    flops_info['total'] = total_flops
    
    return flops_info


def analyze_gradient_flow(model: MiniGPT, dataloader) -> Dict:
    """分析梯度流"""
    model.train()
    
    # 获取一个批次
    input_ids, target_ids = next(iter(dataloader))
    
    # 前向传播
    logits, loss = model(input_ids, target_ids)
    
    # 反向传播
    loss.backward()
    
    # 收集梯度信息
    grad_info = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_mean = param.grad.mean().item()
            grad_std = param.grad.std().item()
            
            grad_info[name] = {
                'norm': grad_norm,
                'mean': grad_mean,
                'std': grad_std
            }
    
    return grad_info


def save_model_analysis(model: MiniGPT, save_dir: str = 'analysis'):
    """保存模型分析结果"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 参数分析
    param_info = analyze_model_parameters(model)
    
    with open(os.path.join(save_dir, 'parameter_analysis.json'), 'w') as f:
        json.dump(param_info, f, indent=2, default=str)
    
    # 复杂度分析
    complexity_info = compute_model_complexity(model, (1, 64))
    
    with open(os.path.join(save_dir, 'complexity_analysis.json'), 'w') as f:
        json.dump(complexity_info, f, indent=2, default=str)
    
    # 绘制参数分布
    plot_parameter_distribution(model, os.path.join(save_dir, 'parameter_distribution.png'))
    
    print(f"模型分析结果已保存到 {save_dir}")


def compare_models(model_configs: List[Dict], save_dir: str = 'comparison'):
    """比较不同配置的模型"""
    os.makedirs(save_dir, exist_ok=True)
    
    comparison_data = []
    
    for config in model_configs:
        # 创建模型
        model = MiniGPT(**config)
        
        # 分析参数
        param_info = analyze_model_parameters(model)
        
        # 计算复杂度
        complexity_info = compute_model_complexity(model, (1, 64))
        
        comparison_data.append({
            'config': config,
            'total_params': param_info['total_params'],
            'total_flops': complexity_info['total'],
            'param_info': param_info,
            'complexity_info': complexity_info
        })
    
    # 绘制比较图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 参数数量比较
    model_names = [f"Model_{i}" for i in range(len(comparison_data))]
    param_counts = [data['total_params'] for data in comparison_data]
    ax1.bar(model_names, param_counts)
    ax1.set_title('Model Parameter Count Comparison')
    ax1.set_ylabel('Parameter Count')
    ax1.tick_params(axis='x', rotation=45)
    
    # FLOPs比较
    flops_counts = [data['total_flops'] for data in comparison_data]
    ax2.bar(model_names, flops_counts)
    ax2.set_title('Model FLOPs Comparison')
    ax2.set_ylabel('FLOPs')
    ax2.tick_params(axis='x', rotation=45)
    
    # 参数类型分布比较
    param_types = ['weights', 'biases']
    for i, data in enumerate(comparison_data):
        param_groups = data['param_info']['param_groups']
        weights = param_groups.get('weights', 0)
        biases = param_groups.get('biases', 0)
        ax3.bar([f"{model_names[i]}_W", f"{model_names[i]}_B"], [weights, biases], 
                alpha=0.7, label=f"Model_{i}")
    ax3.set_title('Parameter Type Distribution Comparison')
    ax3.set_ylabel('Parameter Count')
    ax3.legend()
    
    # 配置参数比较
    d_models = [data['config']['d_model'] for data in comparison_data]
    n_layers = [data['config']['n_layers'] for data in comparison_data]
    ax4.scatter(d_models, n_layers, s=100, alpha=0.7)
    ax4.set_xlabel('d_model')
    ax4.set_ylabel('n_layers')
    ax4.set_title('Model Configuration Comparison')
    
    for i, (d_model, n_layer) in enumerate(zip(d_models, n_layers)):
        ax4.annotate(f"Model_{i}", (d_model, n_layer), xytext=(5, 5), 
                    textcoords='offset points')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 保存比较数据
    with open(os.path.join(save_dir, 'model_comparison.json'), 'w') as f:
        json.dump(comparison_data, f, indent=2, default=str)
    
    return comparison_data


def create_model_configs() -> List[Dict]:
    """创建不同的模型配置用于比较"""
    configs = [
        {
            'vocab_size': 1000,
            'd_model': 64,
            'n_layers': 2,
            'n_heads': 4,
            'd_ff': 256,
            'max_len': 128,
            'dropout': 0.1
        },
        {
            'vocab_size': 1000,
            'd_model': 128,
            'n_layers': 4,
            'n_heads': 8,
            'd_ff': 512,
            'max_len': 128,
            'dropout': 0.1
        },
        {
            'vocab_size': 1000,
            'd_model': 256,
            'n_layers': 6,
            'n_heads': 8,
            'd_ff': 1024,
            'max_len': 128,
            'dropout': 0.1
        },
        {
            'vocab_size': 1000,
            'd_model': 512,
            'n_layers': 8,
            'n_heads': 16,
            'd_ff': 2048,
            'max_len': 128,
            'dropout': 0.1
        }
    ]
    
    return configs


if __name__ == "__main__":
    # 测试工具函数
    print("测试模型分析工具:")
    print("=" * 50)
    
    # 创建测试模型
    model = MiniGPT(
        vocab_size=1000,
        d_model=128,
        n_layers=4,
        n_heads=8,
        d_ff=512
    )
    
    # 分析参数
    param_info = analyze_model_parameters(model)
    print("参数分析结果:")
    print(f"总参数数量: {param_info['total_params']:,}")
    print(f"可训练参数: {param_info['trainable_params']:,}")
    print(f"参数类型分布: {dict(param_info['param_groups'])}")
    
    # 计算复杂度
    complexity_info = compute_model_complexity(model, (1, 64))
    print(f"\n模型复杂度:")
    print(f"总FLOPs: {complexity_info['total']:,}")
    
    # 保存分析结果
    save_model_analysis(model)
    
    # 模型比较
    print("\n模型比较:")
    configs = create_model_configs()
    comparison_data = compare_models(configs)
    
    print("模型比较完成！") 