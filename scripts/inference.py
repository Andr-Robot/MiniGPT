import torch
import json
import os
from typing import List, Dict, Optional
import argparse
import sys

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from core.mini_gpt import MiniGPT
from data.tokenizer import SimpleTokenizer, CharTokenizer, BPETokenizer
from scripts.train import Trainer
from utils.utils import visualize_attention_weights, plot_parameter_distribution


class MiniGPTDemo:
    """MiniGPT演示类"""
    
    def __init__(self, model_path: str, device: str = 'auto'):
        self.device = self._get_device(device)
        self.model, self.tokenizer = self._load_model(model_path)
        self.model.to(self.device)
        
    def _get_device(self, device: str) -> torch.device:
        """获取设备"""
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)
    
    def _load_model(self, model_path: str) -> tuple:
        """加载模型和分词器"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 创建模型
        model_config = checkpoint['model_config']
        model = MiniGPT(**model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 加载分词器
        tokenizer = checkpoint['tokenizer']
        
        print(f"模型已从 {model_path} 加载")
        print(f"模型配置: {model_config}")
        
        return model, tokenizer
    
    def generate_text(self, prompt: str, max_length: int = 50, 
                     temperature: float = 1.0, top_k: int = 50, 
                     top_p: float = 0.9) -> str:
        """生成文本"""
        self.model.eval()
        
        # 编码提示（不添加EOS token）
        prompt_ids = self.tokenizer.encode(prompt, add_eos=False)
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long).to(self.device)
        
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
    
    def interactive_generation(self):
        """交互式文本生成"""
        print("欢迎使用MiniGPT文本生成演示！")
        print("输入 'quit' 退出，输入 'help' 查看帮助")
        print("-" * 50)
        
        while True:
            try:
                user_input = input("\n请输入提示文本: ").strip()
                
                if user_input.lower() == 'quit':
                    print("再见！")
                    break
                elif user_input.lower() == 'help':
                    self._show_help()
                    continue
                elif user_input.lower() == 'config':
                    self._show_config()
                    continue
                elif user_input.lower() == 'analyze':
                    self._analyze_model()
                    continue
                elif not user_input:
                    continue
                
                # 获取生成参数
                max_length = int(input("最大生成长度 (默认50): ") or 50)
                temperature = float(input("温度参数 (默认1.0): ") or 1.0)
                top_k = int(input("Top-k参数 (默认50): ") or 50)
                
                # 生成文本
                print("\n生成中...")
                generated_text = self.generate_text(
                    user_input, 
                    max_length=max_length,
                    temperature=temperature,
                    top_k=top_k
                )
                
                print(f"\n原始提示: {user_input}")
                print(f"生成结果: {generated_text}")
                
            except KeyboardInterrupt:
                print("\n\n再见！")
                break
            except Exception as e:
                print(f"错误: {e}")
    
    def batch_generation(self, prompts: List[str], **kwargs) -> List[str]:
        """批量生成文本"""
        results = []
        
        for i, prompt in enumerate(prompts):
            print(f"生成 {i+1}/{len(prompts)}: {prompt}")
            generated = self.generate_text(prompt, **kwargs)
            results.append(generated)
            print(f"结果: {generated}\n")
        
        return results
    
    def _show_help(self):
        """显示帮助信息"""
        help_text = """
可用命令:
- quit: 退出程序
- help: 显示此帮助信息
- config: 显示模型配置
- analyze: 分析模型参数和结构

生成参数说明:
- max_length: 最大生成长度，控制生成文本的长度
- temperature: 温度参数，控制生成的随机性 (0.1-2.0)
  - 较低值: 更确定性，重复性更高
  - 较高值: 更随机，创造性更强
- top_k: Top-k采样，限制候选token数量
        """
        print(help_text)
    
    def _show_config(self):
        """显示模型配置"""
        config = {
            'vocab_size': self.model.vocab_size,
            'd_model': self.model.d_model,
            'n_layers': len(self.model.transformer_blocks),
            'n_heads': self.model.transformer_blocks[0].attention.n_heads,
            'd_ff': self.model.transformer_blocks[0].feed_forward.linear1.out_features,
            'max_len': self.model.position_encoding.pe.size(0),
            'dropout': self.model.dropout.p,
            'device': str(self.device)
        }
        
        print("模型配置:")
        for key, value in config.items():
            print(f"  {key}: {value}")
    
    def _analyze_model(self):
        """分析模型"""
        print("分析模型参数和结构...")
        
        # 计算参数数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"总参数数量: {total_params:,}")
        print(f"可训练参数: {trainable_params:,}")
        
        # 绘制参数分布
        plot_parameter_distribution(self.model)
    
    def visualize_attention(self, text: str, layer_idx: int = 0, head_idx: int = 0):
        """可视化注意力权重"""
        visualize_attention_weights(self.model, text, self.tokenizer, layer_idx, head_idx)
    
    def save_generations(self, prompts: List[str], outputs: List[str], 
                        filepath: str = 'generations.json'):
        """保存生成结果"""
        results = []
        for prompt, output in zip(prompts, outputs):
            results.append({
                'prompt': prompt,
                'generated_text': output
            })
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"生成结果已保存到 {filepath}")


def main():
    parser = argparse.ArgumentParser(description='MiniGPT推理/生成程序')
    parser.add_argument('--model_path', type=str, default='output/checkpoints/best_model.pth',
                       help='模型文件路径（默认output/checkpoints/best_model.pth）')
    parser.add_argument('--mode', type=str, default='interactive',
                       choices=['interactive', 'batch'],
                       help='运行模式')
    parser.add_argument('--prompts', type=str, nargs='+', default=[],
                       help='批量生成时的提示文本')
    parser.add_argument('--max_length', type=int, default=50,
                       help='最大生成长度')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='温度参数')
    parser.add_argument('--top_k', type=int, default=50,
                       help='Top-k参数')
    parser.add_argument('--output_file', type=str, default='generations.json',
                       help='输出文件路径')
    
    args = parser.parse_args()
    
    # 检查模型文件是否存在
    if not os.path.exists(args.model_path):
        print(f"❌ 未找到模型文件: {args.model_path}")
        print("请先完成模型训练，确保 best_model.pth 已生成！")
        return
    
    try:
        # 创建演示实例
        demo = MiniGPTDemo(args.model_path)
        
        if args.mode == 'interactive':
            # 交互式模式
            demo.interactive_generation()
        else:
            # 批量模式
            if not args.prompts:
                print("批量模式需要提供提示文本，使用 --prompts 参数")
                return
            
            outputs = demo.batch_generation(
                args.prompts,
                max_length=args.max_length,
                temperature=args.temperature,
                top_k=args.top_k
            )
            
            # 保存结果
            demo.save_generations(args.prompts, outputs, args.output_file)
    
    except Exception as e:
        print(f"错误: {e}")
        print("请检查模型文件路径是否正确")


if __name__ == "__main__":
    main() 