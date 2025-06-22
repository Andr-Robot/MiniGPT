import re
from typing import List, Dict, Tuple, Optional
from collections import Counter
import pickle
import os


class SimpleTokenizer:
    """简单的分词器实现"""
    
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.word2idx = {}
        self.idx2word = {}
        self.special_tokens = {
            '<PAD>': 0,  # 填充token
            '<UNK>': 1,  # 未知token
            '<BOS>': 2,  # 序列开始token
            '<EOS>': 3,  # 序列结束token
        }
        
    def fit(self, texts: List[str]):
        """从文本列表构建词汇表"""
        # 统计词频
        word_counts = Counter()
        
        for text in texts:
            # 简单的分词：按空格分割并转换为小写
            words = re.findall(r'\b\w+\b', text.lower())
            word_counts.update(words)
        
        # 选择最常见的词
        most_common = word_counts.most_common(self.vocab_size - len(self.special_tokens))
        
        # 构建词汇表
        self.word2idx = {**self.special_tokens}
        self.idx2word = {idx: word for word, idx in self.special_tokens.items()}
        
        for i, (word, _) in enumerate(most_common):
            idx = i + len(self.special_tokens)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            
    def encode(self, text: str, max_length: Optional[int] = None, add_eos: bool = True) -> List[int]:
        """将文本编码为token ID序列"""
        # 分词
        words = re.findall(r'\b\w+\b', text.lower())
        
        # 转换为ID
        token_ids = [self.word2idx.get(word, self.special_tokens['<UNK>']) for word in words]
        
        # 添加特殊token
        token_ids = [self.special_tokens['<BOS>']] + token_ids
        if add_eos:
            token_ids = token_ids + [self.special_tokens['<EOS>']]
        
        # 截断或填充
        if max_length is not None:
            if len(token_ids) > max_length:
                token_ids = token_ids[:max_length]
            else:
                token_ids.extend([self.special_tokens['<PAD>']] * (max_length - len(token_ids)))
                
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """将token ID序列解码为文本"""
        # 移除特殊token
        words = []
        for token_id in token_ids:
            if token_id == self.special_tokens['<BOS>']:
                continue  # 跳过开始token
            elif token_id == self.special_tokens['<EOS>']:
                break    # 遇到结束token就停止
            elif token_id == self.special_tokens['<PAD>']:
                continue  # 跳过填充token
            elif token_id == self.special_tokens['<UNK>']:
                words.append('<UNK>')  # 保留UNK token
            else:
                words.append(self.idx2word.get(token_id, '<UNK>'))
            
        return ' '.join(words)
    
    def save(self, filepath: str):
        """保存分词器"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'vocab_size': self.vocab_size,
                'word2idx': self.word2idx,
                'idx2word': self.idx2word,
                'special_tokens': self.special_tokens
            }, f)
    
    def load(self, filepath: str):
        """加载分词器"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.vocab_size = data['vocab_size']
            self.word2idx = data['word2idx']
            self.idx2word = data['idx2word']
            self.special_tokens = data['special_tokens']


class CharTokenizer:
    """字符级分词器"""
    
    def __init__(self, vocab_size: int = 128):
        self.vocab_size = vocab_size
        self.char2idx = {}
        self.idx2char = {}
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,
            '<EOS>': 3,
        }
        
    def fit(self, texts: List[str]):
        """从文本列表构建字符词汇表"""
        # 统计字符频次
        char_counts = Counter()
        
        for text in texts:
            char_counts.update(text)
        
        # 选择最常见的字符
        most_common = char_counts.most_common(self.vocab_size - len(self.special_tokens))
        
        # 构建词汇表
        self.char2idx = {**self.special_tokens}
        self.idx2char = {idx: char for char, idx in self.special_tokens.items()}
        
        for i, (char, _) in enumerate(most_common):
            idx = i + len(self.special_tokens)
            self.char2idx[char] = idx
            self.idx2char[idx] = char
            
    def encode(self, text: str, max_length: Optional[int] = None, add_eos: bool = True) -> List[int]:
        """将文本编码为字符ID序列"""
        # 字符级编码
        char_ids = [self.char2idx.get(char, self.special_tokens['<UNK>']) for char in text]
        
        # 添加特殊token
        char_ids = [self.special_tokens['<BOS>']] + char_ids
        if add_eos:
            char_ids = char_ids + [self.special_tokens['<EOS>']]
        
        # 截断或填充
        if max_length is not None:
            if len(char_ids) > max_length:
                char_ids = char_ids[:max_length]
            else:
                char_ids.extend([self.special_tokens['<PAD>']] * (max_length - len(char_ids)))
                
        return char_ids
    
    def decode(self, char_ids: List[int]) -> str:
        """将字符ID序列解码为文本"""
        # 移除特殊token
        chars = []
        for char_id in char_ids:
            if char_id == self.special_tokens['<BOS>']:
                continue  # 跳过开始token
            elif char_id == self.special_tokens['<EOS>']:
                break    # 遇到结束token就停止
            elif char_id == self.special_tokens['<PAD>']:
                continue  # 跳过填充token
            elif char_id == self.special_tokens['<UNK>']:
                chars.append('<UNK>')  # 保留UNK token
            else:
                chars.append(self.idx2char.get(char_id, '<UNK>'))
            
        return ''.join(chars)
    
    def save(self, filepath: str):
        """保存分词器"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'vocab_size': self.vocab_size,
                'char2idx': self.char2idx,
                'idx2char': self.idx2char,
                'special_tokens': self.special_tokens
            }, f)
    
    def load(self, filepath: str):
        """加载分词器"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.vocab_size = data['vocab_size']
            self.char2idx = data['char2idx']
            self.idx2char = data['idx2char']
            self.special_tokens = data['special_tokens']


class BPETokenizer:
    """简易BPE分词器（纯Python实现，适合教学和实验）"""
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.bpe_codes = []  # BPE合并规则（按顺序）
        self.token2idx = {}
        self.idx2token = {}
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,
            '<EOS>': 3,
        }

    def get_vocab(self, texts):
        """统计初始词汇表（以字符为基本单位）"""
        vocab = {}
        for text in texts:
            # 以空格分词，每个词末尾加特殊符号（BPE论文做法）
            for word in text.strip().split():
                word = tuple(word) + ('</w>',)
                vocab[word] = vocab.get(word, 0) + 1
        return vocab

    def get_stats(self, vocab):
        """统计所有相邻符号对的频率"""
        pairs = {}
        for word, freq in vocab.items():
            for i in range(len(word) - 1):
                pair = (word[i], word[i+1])
                pairs[pair] = pairs.get(pair, 0) + freq
        return pairs

    def merge_vocab(self, pair, vocab):
        """将词表中所有出现pair的地方合并为新符号"""
        new_vocab = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        for word, freq in vocab.items():
            word_str = ' '.join(word)
            new_word = tuple(word_str.replace(bigram, replacement).split(' '))
            new_vocab[new_word] = freq
        return new_vocab

    def fit(self, texts, verbose=False):
        """训练BPE合并规则和词表"""
        vocab = self.get_vocab(texts)
        bpe_codes = []
        # 统计初始符号表
        symbols = set()
        for word in vocab:
            symbols.update(word)
        # 预留特殊token
        symbols = set(list(symbols) + list(self.special_tokens.keys()))
        # BPE主循环
        while len(symbols) < self.vocab_size:
            pairs = self.get_stats(vocab)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            vocab = self.merge_vocab(best, vocab)
            bpe_codes.append(best)
            # 更新符号表
            symbols.add(''.join(best))
            if verbose:
                print(f"合并: {best}, 新符号: {''.join(best)}, 当前词表大小: {len(symbols)}")
        self.bpe_codes = bpe_codes
        # 构建token2idx/idx2token
        tokens = set()
        for word in vocab:
            tokens.update(word)
        tokens = list(tokens) + list(self.special_tokens.keys())
        self.token2idx = {tok: idx for idx, tok in enumerate(tokens)}
        self.idx2token = {idx: tok for tok, idx in self.token2idx.items()}

    def bpe(self, word):
        """对单个词应用BPE合并规则，返回token序列"""
        word = tuple(word) + ('</w>',)
        pairs = [(word[i], word[i+1]) for i in range(len(word)-1)]
        word = list(word)
        for merge in self.bpe_codes:
            i = 0
            while i < len(word) - 1:
                if (word[i], word[i+1]) == merge:
                    word[i:i+2] = [''.join(merge)]
                i += 1
        # 移除结尾标记
        if word[-1] == '</w>':
            word = word[:-1]
        return word

    def encode(self, text, max_length=None, add_eos=True):
        """将文本编码为BPE token id序列"""
        tokens = [self.special_tokens['<BOS>']]
        for word in text.strip().split():
            bpe_tokens = self.bpe(word)
            for tok in bpe_tokens:
                tokens.append(self.token2idx.get(tok, self.special_tokens['<UNK>']))
        if add_eos:
            tokens.append(self.special_tokens['<EOS>'])
        if max_length is not None:
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
            else:
                tokens.extend([self.special_tokens['<PAD>']] * (max_length - len(tokens)))
        return tokens

    def decode(self, token_ids):
        """将BPE token id序列解码为文本"""
        words = []
        for tid in token_ids:
            if tid == self.special_tokens['<BOS>']:
                continue
            elif tid == self.special_tokens['<EOS>']:
                break
            elif tid == self.special_tokens['<PAD>']:
                continue
            elif tid == self.special_tokens['<UNK>']:
                words.append('<UNK>')
            else:
                tok = self.idx2token.get(tid, '<UNK>')
                words.append(tok)
        # 合并BPE分词结果
        text = ''.join(words).replace('</w>', ' ').strip()
        return text


def create_sample_data() -> List[str]:
    """创建示例训练数据"""
    return [
        "hello world this is a test",
        "machine learning is fascinating",
        "deep learning models are powerful",
        "artificial intelligence is the future",
        "neural networks can learn patterns",
        "transformer models are revolutionary",
        "attention mechanisms are important",
        "natural language processing is complex",
        "computer vision helps machines see",
        "reinforcement learning teaches agents",
        "the quick brown fox jumps over the lazy dog",
        "machine learning algorithms improve over time",
        "deep neural networks have many layers",
        "attention is all you need",
        "language models can generate text",
        "computer science is fundamental",
        "data science involves statistics and programming",
        "artificial neural networks mimic the brain",
        "supervised learning uses labeled data",
        "unsupervised learning finds hidden patterns"
    ]


if __name__ == "__main__":
    # 测试分词器
    sample_texts = create_sample_data()
    
    print("测试词级分词器:")
    word_tokenizer = SimpleTokenizer(vocab_size=50)
    word_tokenizer.fit(sample_texts)
    
    test_text = "hello world machine learning"
    encoded = word_tokenizer.encode(test_text)
    decoded = word_tokenizer.decode(encoded)
    
    print(f"原始文本: {test_text}")
    print(f"编码结果: {encoded}")
    print(f"解码结果: {decoded}")
    print(f"词汇表大小: {len(word_tokenizer.word2idx)}")
    print(f"词汇表内容: {word_tokenizer.word2idx}")
    
    print("\n测试字符级分词器:")
    char_tokenizer = CharTokenizer(vocab_size=50)
    char_tokenizer.fit(sample_texts)
    
    test_text = "hello world"
    encoded = char_tokenizer.encode(test_text)
    decoded = char_tokenizer.decode(encoded)
    
    print(f"原始文本: {test_text}")
    print(f"编码结果: {encoded}")
    print(f"解码结果: {decoded}")
    print(f"字符表大小: {len(char_tokenizer.char2idx)}")
    print(f"字符表内容: {char_tokenizer.char2idx}")

    print("\n测试BPE分词器:")
    bpe_tokenizer = BPETokenizer(vocab_size=150)
    bpe_tokenizer.fit(sample_texts, verbose=True)
    test_text = "hello world machine learning"
    encoded = bpe_tokenizer.encode(test_text)
    decoded = bpe_tokenizer.decode(encoded)
    print(f"原始文本: {test_text}")
    print(f"编码结果: {encoded}")
    print(f"解码结果: {decoded}")
    print(f"BPE词表大小: {len(bpe_tokenizer.token2idx)}")
    # 只显示前30项，避免输出过长
    print(f"BPE词表内容(前30项): {dict(list(bpe_tokenizer.token2idx.items())[:30])}") 