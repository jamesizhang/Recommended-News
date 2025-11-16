# 测试文件说明

本目录包含项目的单元测试文件。

## 测试文件列表

- `test_news4.py` - 测试 news4.py 中的 groupby 相关功能
- `test_word2vec.py` - 测试 Word2Vec 相关功能

## 运行测试

### 运行所有测试
```bash
cd /root/news/Recommended-News-main
python -m unittest discover tests -v
```

### 运行单个测试文件
```bash
# 运行 test_news4.py
python -m unittest tests.test_news4 -v

# 运行 test_word2vec.py
python -m unittest tests.test_word2vec -v
```

### 运行单个测试用例
```bash
# 运行 test_news4.py 中的特定测试
python -m unittest tests.test_news4.TestNews4GroupbyFix.test_groupby_apply_reset_index -v
```

## 测试覆盖

### test_news4.py
- groupby().apply().reset_index() 的修复
- Series 和 DataFrame 的处理
- 字典创建功能
- 边界情况处理

### test_word2vec.py
- Word2Vec 基本功能
- 新版本访问方式 (w2v.wv[key])
- 不同长度序列的处理
- 短序列与 window 的关系
- 字典创建逻辑

