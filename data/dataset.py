#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MRT-12 数据集管理模块
数据加载、词表构建、Lazy Dataset 实现

Author: 罗兵 (Luo Bing)
Email: <2712179753@qq.com>
WeChat: 18368870543
Douyin: 1918705950
"""

import os
import json
import re
import torch
from torch.utils.data import Dataset
from collections import Counter
import logging

logger = logging.getLogger('MRT_Data')

# ============================================================
# 1. 深度清洗逻辑 (Vortex Cleaning)
# ============================================================

def clean_chinese_text(text):
    """
    1. 移除Wiki标记 [1], [2]
    2. 只保留中文、常用标点
    3. 过滤高数字占比句子 (防止年份、坐标污染流形)
    """
    # 移除引用
    text = re.sub(r'\[\d+\]', '', text)
    # 统计数字比例
    nums = len(re.findall(r'\d', text))
    if len(text) > 0 and (nums / len(text)) > 0.15:
        return ""
    
    # 仅保留汉字及核心标点：。！？，、
    text = re.sub(r'[^\u4e00-\u9fa5。！？，、]', '', text)
    return text.strip()

# ============================================================
# 2. 数据加载 (128G RAM 全量承载)
# ============================================================

def load_data_final(filepath, max_sentences=50000000):
    """
    全量读取 JSONL 或 TXT。
    凭借 128G 内存，我们可以直接将数千万句字符串存入 Python 列表。
    """
    print(f">>> [Data] 正在全量载入语料: {filepath}")
    sentences = []
    
    if not os.path.exists(filepath):
        logger.error(f"找不到数据文件: {filepath}")
        return ["这是一个占位句子，请检查数据路径。"]

    with open(filepath, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if len(sentences) >= max_sentences:
                break
            
            try:
                # 兼容 JSONL 和纯 TXT
                if filepath.endswith(".jsonl"):
                    data = json.loads(line)
                    raw_text = data.get('text', '') or data.get('content', '')
                else:
                    raw_text = line
                
                # 分句处理 (按常用终结符切分)
                parts = re.split(r'([。！？])', raw_text)
                for i in range(0, len(parts)-1, 2):
                    sent = parts[i] + parts[i+1]
                    clean_sent = clean_chinese_text(sent)
                    # 长度过滤：太短的没逻辑，太长的炸显存
                    if 15 <= len(clean_sent) <= 256:
                        sentences.append(clean_sent)
                        
            except:
                continue
            
            if idx % 1000000 == 0 and idx > 0:
                print(f"    已扫描 {idx} 行，当前内存持有 {len(sentences)} 条有效句...")

    print(f">>> [Data] 加载完毕！最终有效句子数: {len(sentences)}")
    return sentences

# ============================================================
# 3. 词表持久化 (一次构建，永久复用)
# ============================================================

def build_or_load_vocab(sentences, vocab_file="mrt_vocab.json", max_vocab_size=20000):
    """
    检查本地是否存在词表。
    若不存在，利用全量数据构建 Top N 词表并保存。
    """
    if os.path.exists(vocab_file):
        print(f">>> [Vocab] 发现本地词表 {vocab_file}，正在秒速加载...")
        with open(vocab_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            w2i = data["w2i"]
            i2w = {int(k): v for k, v in data["i2w"].items()}
        return w2i, i2w

    print(f">>> [Vocab] 未发现词表。开始全量扫描 {len(sentences)} 条句子构建词表...")
    counter = Counter()
    for s in sentences:
        counter.update(list(s))
    
    # 选取高频词，保留 <pad> 和 <unk>
    most_common = [w for w, c in counter.most_common(max_vocab_size)]
    vocab = ["<pad>", "<unk>"] + most_common
    
    w2i = {w: i for i, w in enumerate(vocab)}
    i2w = {i: w for i, w in enumerate(vocab)}
    
    print(f">>> [Vocab] 词表构建完成，大小: {len(w2i)}。正在保存到本地...")
    with open(vocab_file, "w", encoding="utf-8") as f:
        json.dump({"w2i": w2i, "i2w": i2w}, f, ensure_ascii=False, indent=2)
    
    return w2i, i2w

# ============================================================
# 4. PyTorch Dataset 适配
# ============================================================

class WikiDataset(Dataset):
    """
    Lazy Dataset: 内存只存原始字符串。
    在 __getitem__ 时才转为 ID，极大缓解内存压力。
    """
    def __init__(self, sentences, w2i, max_len=256):
        self.sentences = sentences
        self.w2i = w2i
        self.max_len = max_len
        self.unk_id = w2i.get("<unk>", 1)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        s = self.sentences[idx]
        ids = [self.w2i.get(c, self.unk_id) for c in s]
        # 截断
        return torch.tensor(ids[:self.max_len], dtype=torch.long)

def pad_collate_fn(batch):
    """
    动态 Padding：以当前 Batch 中最长的句子为基准进行对齐。
    """
    return pad_sequence(batch, batch_first=True, padding_value=0)

# ============================================================
# 5. 兼容性函数 (保持与旧代码的接口一致)
# ============================================================

def simple_tokenizer(text: str) -> list:
    """简单分词器：按字符分割（兼容旧接口）"""
    return list(text)

def pad_sequence(sequences, batch_first=False, padding_value=0):
    """序列填充函数（兼容旧接口）"""
    max_len = max(len(seq) for seq in sequences)
    if batch_first:
        out_tensor = torch.full((len(sequences), max_len), padding_value, dtype=sequences[0].dtype)
        for i, seq in enumerate(sequences):
            out_tensor[i, :len(seq)] = seq
    else:
        out_tensor = torch.full((max_len, len(sequences)), padding_value, dtype=sequences[0].dtype)
        for i, seq in enumerate(sequences):
            out_tensor[:len(seq), i] = seq
    return out_tensor

# 为保持向后兼容性，保留部分旧接口
class VocabManager:
    """词汇表管理器（兼容旧接口）"""
    def __init__(self, special_tokens=None):
        self.special_tokens = special_tokens or ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
        self.token2id = {}
        self.id2token = {}
        self.vocab_size = 0
        
        for token in self.special_tokens:
            self._add_token(token)
    
    def _add_token(self, token):
        if token not in self.token2id:
            self.token2id[token] = self.vocab_size
            self.id2token[self.vocab_size] = token
            self.vocab_size += 1
    
    def encode(self, tokens):
        return [self.token2id.get(token, self.token2id["<UNK>"]) for token in tokens]
    
    def decode(self, ids):
        return [self.id2token.get(id, "<UNK>") for id in ids]
    
    def build_from_sentences(self, sentences, min_freq=2):
        counter = Counter()
        for sent in sentences:
            counter.update(sent)
        for token, freq in counter.items():
            if freq >= min_freq and token not in self.special_tokens:
                self._add_token(token)
    
    def save(self, filepath):
        vocab_data = {"token2id": self.token2id, "special_tokens": self.special_tokens}
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
    
    def load(self, filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)
        self.token2id = vocab_data["token2id"]
        self.special_tokens = vocab_data["special_tokens"]
        self.vocab_size = len(self.token2id)
        self.id2token = {v: k for k, v in self.token2id.items()}

def build_or_load_vocab_old(sentences, vocab_path="vocab.json", min_freq=2):
    """构建或加载词汇表（兼容旧接口）"""
    vocab_manager = VocabManager()
    
    if os.path.exists(vocab_path):
        print(f"Loading vocabulary from {vocab_path}")
        vocab_manager.load(vocab_path)
    else:
        print("Building vocabulary from sentences...")
        tokenized_sents = [simple_tokenizer(sent) for sent in sentences]
        vocab_manager.build_from_sentences(tokenized_sents, min_freq=min_freq)
        vocab_manager.save(vocab_path)
        print(f"Vocabulary built with {vocab_manager.vocab_size} tokens")
    
    def encode_fn(text):
        tokens = simple_tokenizer(text)
        return vocab_manager.encode(tokens)
    
    return vocab_manager, encode_fn
