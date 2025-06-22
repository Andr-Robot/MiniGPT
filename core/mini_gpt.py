import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # 线性变换层
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """缩放点积注意力"""
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        return output, attention_weights
    
    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()
        
        # 线性变换
        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)
        
        # 重塑为多头
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # 注意力计算
        output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # 重塑回原始形状
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        # 输出投影
        output = self.w_o(output)
        
        return output, attention_weights


class FeedForward(nn.Module):
    """前馈神经网络"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class LayerNorm(nn.Module):
    """层归一化"""
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        return self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta


class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TransformerBlock(nn.Module):
    """Transformer块"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # 自注意力 + 残差连接
        attn_output, _ = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈网络 + 残差连接
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class MiniGPT(nn.Module):
    """微型GPT模型"""
    
    def __init__(self, vocab_size: int, d_model: int = 256, n_layers: int = 6, 
                 n_heads: int = 8, d_ff: int = 1024, max_len: int = 512, 
                 dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # 词嵌入
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_encoding = PositionalEncoding(d_model, max_len)
        
        # Transformer层
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # 输出层
        self.output_projection = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                
    def create_causal_mask(self, seq_len: int) -> torch.Tensor:
        """创建因果掩码（用于自回归生成）"""
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        return ~mask
        
    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播
        
        Args:
            x: 输入序列 [batch_size, seq_len]
            targets: 目标序列 [batch_size, seq_len] (可选，用于训练)
            
        Returns:
            logits: 输出logits [batch_size, seq_len, vocab_size]
            loss: 损失值 (如果提供了targets)
        """
        batch_size, seq_len = x.size()
        
        # 词嵌入 + 位置编码
        token_embeddings = self.token_embedding(x) * math.sqrt(self.d_model)
        x = self.position_encoding(token_embeddings.transpose(0, 1)).transpose(0, 1)
        x = self.dropout(x)
        
        # 创建因果掩码
        causal_mask = self.create_causal_mask(seq_len).to(x.device)
        
        # 通过Transformer层
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, causal_mask)
            
        # 输出投影
        logits = self.output_projection(x)
        
        # 计算损失（如果提供了targets）
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
            
        return logits, loss
    
    def generate(self, prompt: torch.Tensor, max_length: int = 100, 
                temperature: float = 1.0, top_k: int = 50) -> torch.Tensor:
        """
        生成文本
        
        Args:
            prompt: 起始序列 [batch_size, prompt_len]
            max_length: 最大生成长度
            temperature: 温度参数
            top_k: top-k采样参数
            
        Returns:
            生成的序列 [batch_size, generated_length]
        """
        self.eval()
        with torch.no_grad():
            batch_size = prompt.size(0)
            generated = prompt.clone()
            
            # 计算需要生成的新token数量
            tokens_to_generate = max_length - prompt.size(1)
            
            for _ in range(tokens_to_generate):
                # 获取最后一个token的预测
                logits, _ = self(generated)
                next_token_logits = logits[:, -1, :] / temperature
                
                # 避免生成PAD token (ID: 0) 和特殊token
                next_token_logits[:, 0] = -float('inf')  # PAD token
                if next_token_logits.size(-1) > 1:
                    next_token_logits[:, 1] = -float('inf')  # UNK token
                
                # Top-k采样
                if top_k > 0:
                    # 确保top_k不超过词汇表大小
                    actual_top_k = min(top_k, next_token_logits.size(-1))
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, actual_top_k)
                    next_token_logits = torch.full_like(next_token_logits, -float('inf'))
                    next_token_logits.scatter_(1, top_k_indices, top_k_logits)
                
                # 采样下一个token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # 检查是否生成了EOS token，如果是则停止生成
                if next_token[0].item() == 3:  # EOS token ID
                    generated = torch.cat([generated, next_token], dim=1)
                    break
                
                # 添加到生成序列
                generated = torch.cat([generated, next_token], dim=1)
                
        return generated


def count_parameters(model: nn.Module) -> int:
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 测试模型
    vocab_size = 1000
    d_model = 256
    n_layers = 6
    n_heads = 8
    
    model = MiniGPT(vocab_size, d_model, n_layers, n_heads)
    print(f"模型参数数量: {count_parameters(model):,}")
    
    # 测试前向传播
    batch_size, seq_len = 4, 32
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    logits, loss = model(x, targets)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {logits.shape}")
    print(f"损失值: {loss.item():.4f}")
    
    # 测试生成
    prompt = torch.randint(0, vocab_size, (1, 10))
    generated = model.generate(prompt, max_length=20)
    print(f"生成序列形状: {generated.shape}") 