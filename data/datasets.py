"""
MiniGPT 数据集模块
包含多个适合本地训练的小型数据集
"""

import os
import requests
import zipfile
import json
import csv
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import random
from tqdm import tqdm

# 数据集配置
DATASET_CONFIGS = {
    'tiny_shakespeare': {
        'url': 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt',
        'filename': 'tinyshakespeare.txt',
        'description': '莎士比亚作品的小型版本，约1MB，适合字符级训练',
        'type': 'text',
        'size': '~1MB'
    },
    'tiny_stories': {
        'url': 'https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt',
        'filename': 'tinystories.txt',
        'description': 'AI生成的简单故事，适合语言模型训练',
        'type': 'text',
        'size': '~27MB'
    },
    'wikitext_103': {
        'url': 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip',
        'filename': 'wikitext-103-raw.zip',
        'description': '维基百科文章的小型版本，适合词级训练',
        'type': 'text',
        'size': '~50MB'
    },
    'ag_news': {
        'url': 'https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv',
        'filename': 'ag_news.csv',
        'description': '新闻分类数据集，包含新闻标题和内容',
        'type': 'csv',
        'size': '~12MB'
    }
}


class DatasetLoader:
    """数据集加载器"""
    
    def __init__(self, data_dir: str = 'datasets'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
    
    def download_dataset(self, dataset_name: str, force_download: bool = False) -> str:
        """下载数据集"""
        if dataset_name not in DATASET_CONFIGS:
            raise ValueError(f"未知数据集: {dataset_name}")
        
        config = DATASET_CONFIGS[dataset_name]
        filepath = self.data_dir / config['filename']
        
        if filepath.exists() and not force_download:
            print(f"数据集 {dataset_name} 已存在: {filepath}")
            return str(filepath)
        
        print(f"下载数据集 {dataset_name}...")
        print(f"描述: {config['description']}")
        print(f"大小: {config['size']}")
        
        try:
            import os
            os.environ["http_proxy"] = "http://127.0.0.1:1087"  # 替换实际代理
            os.environ["https_proxy"] = "http://127.0.0.1:1087"
            response = requests.get(config['url'], stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            print(f"数据集下载完成: {filepath}")
            return str(filepath)
            
        except Exception as e:
            print(f"下载失败: {e}")
            raise
    
    def load_tiny_shakespeare(self, max_samples: Optional[int] = None) -> List[str]:
        """加载Tiny Shakespeare数据集"""
        filepath = self.download_dataset('tiny_shakespeare')
        
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # 按行分割，过滤空行
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        if max_samples:
            lines = lines[:max_samples]
        
        print(f"加载了 {len(lines)} 行文本")
        return lines
    
    def load_tiny_stories(self, max_samples: Optional[int] = None) -> List[str]:
        """加载Tiny Stories数据集"""
        filepath = self.download_dataset('tiny_stories')
        
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # 按故事分割（每个故事以换行符分隔）
        stories = [story.strip() for story in text.split('\n\n') if story.strip()]
        
        if max_samples:
            stories = stories[:max_samples]
        
        print(f"加载了 {len(stories)} 个故事")
        return stories
    
    def load_wikitext_103(self, max_samples: Optional[int] = None) -> List[str]:
        """加载WikiText-103数据集"""
        filepath = self.download_dataset('wikitext_103')
        
        # 解压文件
        extract_dir = self.data_dir / 'wikitext-103-raw'
        if not extract_dir.exists():
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)
        
        # 读取训练文件
        train_file = extract_dir / 'wiki.train.raw'
        if not train_file.exists():
            raise FileNotFoundError(f"训练文件不存在: {train_file}")
        
        with open(train_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # 按段落分割
        paragraphs = [para.strip() for para in text.split('\n\n') if para.strip()]
        
        if max_samples:
            paragraphs = paragraphs[:max_samples]
        
        print(f"加载了 {len(paragraphs)} 个段落")
        return paragraphs
    
    def load_ag_news(self, max_samples: Optional[int] = None) -> List[str]:
        """加载AG News数据集"""
        filepath = self.download_dataset('ag_news')
        
        texts = []
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 3:  # 确保有足够的列
                    # 组合标题和内容
                    title = row[1].strip()
                    content = row[2].strip()
                    combined_text = f"{title}. {content}"
                    texts.append(combined_text)
        
        if max_samples:
            texts = texts[:max_samples]
        
        print(f"加载了 {len(texts)} 条新闻")
        return texts
    
    def load_dataset(self, dataset_name: str, max_samples: Optional[int] = None) -> List[str]:
        """通用数据集加载函数"""
        if dataset_name == 'tiny_shakespeare':
            return self.load_tiny_shakespeare(max_samples)
        elif dataset_name == 'tiny_stories':
            return self.load_tiny_stories(max_samples)
        elif dataset_name == 'wikitext_103':
            return self.load_wikitext_103(max_samples)
        elif dataset_name == 'ag_news':
            return self.load_ag_news(max_samples)
        else:
            raise ValueError(f"不支持的数据集: {dataset_name}")
    
    def list_available_datasets(self) -> Dict[str, Dict]:
        """列出可用的数据集"""
        return DATASET_CONFIGS
    
    def get_dataset_info(self, dataset_name: str) -> Dict:
        """获取数据集信息"""
        if dataset_name not in DATASET_CONFIGS:
            raise ValueError(f"未知数据集: {dataset_name}")
        
        config = DATASET_CONFIGS[dataset_name]
        filepath = self.data_dir / config['filename']
        
        info = config.copy()
        info['downloaded'] = filepath.exists()
        info['filepath'] = str(filepath) if filepath.exists() else None
        
        if filepath.exists():
            info['file_size'] = f"{filepath.stat().st_size / 1024 / 1024:.1f}MB"
        
        return info


def create_mixed_dataset(datasets: List[str], max_samples_per_dataset: Optional[int] = None) -> List[str]:
    """创建混合数据集"""
    loader = DatasetLoader()
    all_texts = []
    
    for dataset_name in datasets:
        print(f"加载数据集: {dataset_name}")
        texts = loader.load_dataset(dataset_name, max_samples_per_dataset)
        all_texts.extend(texts)
    
    # 随机打乱
    random.shuffle(all_texts)
    
    print(f"混合数据集总样本数: {len(all_texts)}")
    return all_texts


def create_curated_dataset() -> List[str]:
    """创建精选数据集（组合多个小数据集）"""
    datasets = ['tiny_shakespeare', 'ag_news']
    return create_mixed_dataset(datasets, max_samples_per_dataset=1000)


if __name__ == "__main__":
    # 测试数据集加载
    loader = DatasetLoader()
    
    print("可用数据集:")
    for name, config in loader.list_available_datasets().items():
        print(f"  {name}: {config['description']} ({config['size']})")
    
    print("\n测试加载Tiny Shakespeare数据集:")
    texts = loader.load_dataset('tiny_shakespeare', max_samples=10)
    for i, text in enumerate(texts[:3]):
        print(f"样本 {i+1}: {text[:100]}...")
    
    print("\n测试加载AG News数据集:")
    news_texts = loader.load_dataset('ag_news', max_samples=5)
    for i, text in enumerate(news_texts[:3]):
        print(f"新闻 {i+1}: {text[:100]}...") 