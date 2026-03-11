# MRT-12 数据集文件

## 📊 包含的数据集

### zhwiki_dataset.jsonl (主训练语料)
- **文件大小**: 2.3 GB
- **语言**: 中文
- **来源**: 中文维基百科
- **格式**: JSONL (每行一个 JSON 对象)
- **用途**: MRT-12 世界模型预训练

#### 数据示例
```json
{"text": "人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。"}
```

## 🎯 使用方式

### 在训练中加载
```python
from data.dataset import load_data_final

# 加载数据
sentences = load_data_final('data/zhwiki_dataset.jsonl', max_sentences=50000000)
print(f"加载完成：{len(sentences)} 条句子")
```

### 构建词表
```python
from data.dataset import build_or_load_vocab

# 构建或加载词表
w2i, i2w = build_or_load_vocab(sentences, vocab_file='data/mrt_vocab.json')
print(f"词表大小：{len(w2i)} tokens")
```

## 📈 数据统计

- **总句数**: ~50,000,000 条（实际数量取决于分句逻辑）
- **平均句长**: ~50-100 字符
- **总字符数**: ~2.5B+ 字符
- **推荐配置**: 
  - 训练 batch_size: 4-8（根据 GPU 显存）
  - 梯度累积：32 步
  - 等效 batch: 128-256

## 🔧 数据处理流程

1. **加载**: `load_data_final()` 读取 JSONL 文件
2. **清洗**: 自动移除引用标记、过滤高数字占比句子
3. **分句**: 按标点符号分割为独立句子
4. **编码**: 懒加载 tokenization，节省内存
5. **填充**: 动态 padding，批次内对齐

## 💡 最佳实践

### 内存优化
- ✅ 使用懒加载 tokenization（在 `__getitem__` 时编码）
- ✅ 词表持久化到磁盘（避免重复构建）
- ✅ 减少 num_workers（如设置为 8）降低内存压力

### 性能提升
- ✅ 启用 `pin_memory=True` 加速 CPU→GPU 传输
- ✅ 使用 `persistent_workers=True` 减少进程创建开销
- ✅ 合理设置 `num_workers=8-16`（根据系统内存）

## ⚠️ 注意事项

1. **文件完整性**: 确保文件完整复制（2.3GB）
2. **编码格式**: UTF-8 编码，避免中文乱码
3. **磁盘空间**: 建议保留至少 5GB 空闲空间用于数据处理
4. **备份建议**: 原始数据只读，避免意外修改

## 🔗 相关资源

- **备用数据**: `zhwiki_sentences.txt` (2.0GB)
- **大型语料**: `zh_corpus_huge.txt` (5.1GB)
- **逻辑数据**: `logic_library.json` (1.9 万条黄金数据)
- **微型数据**: `micro_dataset.jsonl` (15.6MB，快速验证用)

---

<p align="center">
  Built by 罗兵 (Luo Bing) for advancing geometric AI research | Email: <2712179753@qq.com> | WeChat: 18368870543 | Douyin: 1918705950
</p>
