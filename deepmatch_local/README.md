# 本地可修改的 DeepMatch 版本

这是一个本地可修改的 DeepMatch 版本，主要修改了 YouTubeDNN 模型以支持多个 item features。

## 主要修改

1. **支持多个 item features**：原始版本只支持 1 个 item feature（item_id），现在可以支持多个（如 item_id + category_id）

2. **文件结构**：
   ```
   deepmatch_local/
   ├── deepmatch/
   │   ├── __init__.py
   │   ├── inputs.py
   │   ├── utils.py
   │   ├── layers/
   │   │   ├── __init__.py
   │   │   └── core.py
   │   └── models/
   │       ├── __init__.py
   │       └── youtubednn.py  # 修改版，支持多个 item features
   └── README.md
   ```

## 使用方法

在代码开头添加：
```python
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'deepmatch_local'))

from deepmatch.models import YoutubeDNN
```

## 修改说明

在 `youtubednn.py` 中：
- 移除了 `if len(item_feature_columns) > 1: raise ValueError(...)` 的限制
- 当有多个 item features 时，会将它们的 embeddings concat 起来
- 第一个 item feature 仍然作为主 feature（用于负采样）

## 注意事项

- 这个版本基于 deepmatch 库修改
- 确保已安装 deepctr 和 tensorflow
- 可以自由修改代码以满足需求

