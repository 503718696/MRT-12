#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MRT-12 数据清洗模块
文本预处理和质量控制

Author: 罗兵 (Luo Bing)
Email: <2712179753@qq.com>
WeChat: 18368870543
Douyin: 1918705950
"""

import re
import unicodedata
from typing import List, Callable

class DataCleaner:
    """数据清洗器：文本预处理和质量控制"""
    
    def __init__(self):
        # 中文标点符号映射
        self.chinese_punctuation = {
            '，': ',',
            '。': '.',
            '！': '!',
            '？': '?',
            '；': ';',
            '：': ':',
            '“': '"',
            '”': '"',
            '‘': "'",
            '’': "'",
            '（': '(',
            '）': ')',
            '【': '[',
            '】': ']',
            '《': '<',
            '》': '>'
        }
    
    def normalize_unicode(self, text: str) -> str:
        """Unicode标准化"""
        return unicodedata.normalize('NFKC', text)
    
    def remove_control_chars(self, text: str) -> str:
        """移除控制字符"""
        return ''.join(char for char in text if ord(char) >= 32 or char in '\n\r\t')
    
    def standardize_punctuation(self, text: str) -> str:
        """标准化标点符号"""
        for chinese, english in self.chinese_punctuation.items():
            text = text.replace(chinese, english)
        return text
    
    def remove_extra_whitespace(self, text: str) -> str:
        """移除多余空白字符"""
        # 替换多个空格为单个空格
        text = re.sub(r'\s+', ' ', text)
        # 移除行首行尾空白
        return text.strip()
    
    def clean_numbers(self, text: str, strategy='mask') -> str:
        """
        数字处理策略
        Args:
            text: 输入文本
            strategy: 处理策略 ('keep', 'mask', 'remove')
        Returns:
            处理后的文本
        """
        if strategy == 'keep':
            return text
        elif strategy == 'mask':
            # 将数字替换为特殊标记
            return re.sub(r'\d+', '[NUM]', text)
        elif strategy == 'remove':
            # 完全移除数字
            return re.sub(r'\d+', '', text)
        else:
            return text
    
    def filter_by_length(self, text: str, min_chars=1, max_chars=1000) -> bool:
        """按长度过滤文本"""
        char_count = len(text.strip())
        return min_chars <= char_count <= max_chars
    
    def filter_by_quality(self, text: str, min_alpha_ratio=0.5) -> bool:
        """
        按质量过滤文本
        Args:
            text: 输入文本
            min_alpha_ratio: 最小字母字符比例
        Returns:
            是否通过质量检查
        """
        if not text.strip():
            return False
        
        # 计算字母字符比例
        alpha_chars = sum(1 for c in text if c.isalpha() or c.isspace())
        alpha_ratio = alpha_chars / len(text)
        
        return alpha_ratio >= min_alpha_ratio
    
    def clean_text(self, text: str, 
                   normalize_unicode=True,
                   remove_controls=True,
                   standardize_punct=True,
                   clean_numbers_strategy='mask',
                   remove_whitespace=True) -> str:
        """
        综合文本清洗
        """
        if normalize_unicode:
            text = self.normalize_unicode(text)
        
        if remove_controls:
            text = self.remove_control_chars(text)
        
        if standardize_punct:
            text = self.standardize_punctuation(text)
        
        text = self.clean_numbers(text, clean_numbers_strategy)
        
        if remove_whitespace:
            text = self.remove_extra_whitespace(text)
        
        return text

def create_cleaning_pipeline(min_length=10, max_length=512, 
                           min_alpha_ratio=0.3) -> Callable[[str], str]:
    """
    创建清洗流水线
    Args:
        min_length: 最小文本长度
        max_length: 最大文本长度
        min_alpha_ratio: 最小字母比例
    Returns:
        清洗函数
    """
    cleaner = DataCleaner()
    
    def pipeline(text: str) -> str:
        # 基础清洗
        cleaned = cleaner.clean_text(text)
        
        # 长度过滤
        if not cleaner.filter_by_length(cleaned, min_length, max_length):
            return ""
        
        # 质量过滤
        if not cleaner.filter_by_quality(cleaned, min_alpha_ratio):
            return ""
        
        return cleaned
    
    return pipeline

def batch_clean_texts(texts: List[str], 
                     cleaning_func: Callable[[str], str] = None) -> List[str]:
    """
    批量清洗文本
    Args:
        texts: 文本列表
        cleaning_func: 清洗函数
    Returns:
        清洗后的文本列表
    """
    if cleaning_func is None:
        cleaning_func = create_cleaning_pipeline()
    
    cleaned_texts = []
    for text in texts:
        cleaned = cleaning_func(text)
        if cleaned:  # 只保留非空文本
            cleaned_texts.append(cleaned)
    
    return cleaned_texts
