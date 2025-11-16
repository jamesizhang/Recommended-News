# Transformer推荐模型评估指标说明

## 当前代码中的评估方法

根据 `news3_transformer.py` 中的实现，当前主要使用 **Hit@K (Hit Rate@K)** 指标：

```python
def metrics_recall(user_recall_items_dict, trn_last_click_df, topk=50):
    """评估召回效果"""
    # 检查用户最后一次点击的物品是否在召回的前K个物品中
    for k in range(10, topk + 1, 10):
        hit_num = 0
        for user in valid_users:
            recall_item_ids = [item[0] for item in item_list[:k]]
            if last_click_item in set(recall_item_ids):
                hit_num += 1
        hit_rate = hit_num / user_num
        print(f"k={k}: hit_num={hit_num}, hit_rate={hit_rate}")
```

## 推荐的评估指标

对于Transformer直接输出作为推荐，建议使用以下指标组合：

### 1. **Hit@K (Hit Rate@K)** - 命中率 ✅ 已实现
- **定义**：真实物品是否出现在前K个推荐中
- **计算**：`Hit@K = (命中用户数) / (总用户数)`
- **特点**：只关注是否命中，不考虑排序位置
- **常用K值**：Hit@1, Hit@5, Hit@10, Hit@20, Hit@50
- **注意**：Hit@1 通常太严格，建议至少看 Hit@5

### 2. **NDCG@K (Normalized Discounted Cumulative Gain@K)** - 推荐添加
- **定义**：考虑排序位置的归一化折损累积增益
- **优势**：不仅看是否命中，还考虑命中物品的排序位置（位置越靠前，得分越高）
- **公式**：
  ```
  DCG@K = Σ(rel_i / log2(i+2))  # i是位置，rel_i是相关性（0或1）
  IDCG@K = 理想情况下的DCG（相关物品都排在前面）
  NDCG@K = DCG@K / IDCG@K
  ```
- **常用K值**：NDCG@5, NDCG@10, NDCG@20

### 3. **Precision@K** - 推荐添加
- **定义**：前K个推荐中有多少个是相关的
- **计算**：`Precision@K = (命中物品数) / K`
- **特点**：关注推荐的精确度
- **常用K值**：Precision@5, Precision@10

### 4. **MRR (Mean Reciprocal Rank)** - 推荐添加
- **定义**：平均倒数排名，关注第一个相关物品的位置
- **计算**：`MRR = (1/N) * Σ(1/rank_i)`，其中rank_i是第一个相关物品的位置
- **特点**：如果第一个推荐就命中，MRR=1；如果第5个才命中，MRR=0.2
- **适用场景**：特别关注Top1推荐质量的场景

### 5. **MAP@K (Mean Average Precision@K)** - 可选
- **定义**：平均精确率均值
- **特点**：综合考虑多个相关物品的情况

## 针对Transformer的评估建议

### 召回阶段评估
- **主要指标**：Hit@10, Hit@20, Hit@50
- **辅助指标**：NDCG@10, NDCG@20（考虑排序质量）
- **不建议只用Hit@1**：太严格，容易低估模型效果

### 排序阶段评估
- **主要指标**：NDCG@5, NDCG@10（排序质量最重要）
- **辅助指标**：Hit@5, Precision@5, MRR（关注Top1质量）

### 端到端评估（召回+排序）
- **主要指标**：Hit@5, NDCG@5（最终推荐效果）
- **辅助指标**：Hit@10, NDCG@10, MRR

## 实现建议

可以在 `news3_transformer.py` 中添加更完整的评估函数：

```python
def evaluate_transformer_recommendation(user_recall_items_dict, trn_last_click_df, k_list=[1, 5, 10, 20, 50]):
    """
    评估Transformer推荐效果，使用多个指标
    """
    last_click_item_dict = dict(zip(trn_last_click_df['user_id'], trn_last_click_df['click_article_id']))
    valid_users = set(user_recall_items_dict.keys()) & set(last_click_item_dict.keys())
    
    metrics = {
        'hit_rate': {k: [] for k in k_list},
        'ndcg': {k: [] for k in k_list},
        'precision': {k: [] for k in k_list},
        'mrr': []
    }
    
    for user in valid_users:
        recall_items = user_recall_items_dict[user]
        recall_item_ids = [item[0] if isinstance(item, (list, tuple)) else item for item in recall_items]
        true_item = last_click_item_dict[user]
        
        # 计算每个K的指标
        for k in k_list:
            top_k_items = recall_item_ids[:k]
            
            # Hit@K
            hit = 1 if true_item in top_k_items else 0
            metrics['hit_rate'][k].append(hit)
            
            # Precision@K
            precision = hit / k
            metrics['precision'][k].append(precision)
            
            # NDCG@K
            if true_item in top_k_items:
                rank = top_k_items.index(true_item) + 1
                dcg = 1.0 / np.log2(rank + 1)
                idcg = 1.0 / np.log2(2)  # 理想情况：相关物品排第一
                ndcg = dcg / idcg
            else:
                ndcg = 0.0
            metrics['ndcg'][k].append(ndcg)
        
        # MRR
        if true_item in recall_item_ids:
            rank = recall_item_ids.index(true_item) + 1
            mrr = 1.0 / rank
        else:
            mrr = 0.0
        metrics['mrr'].append(mrr)
    
    # 计算平均值
    results = {}
    for k in k_list:
        results[f'Hit@{k}'] = np.mean(metrics['hit_rate'][k])
        results[f'NDCG@{k}'] = np.mean(metrics['ndcg'][k])
        results[f'Precision@{k}'] = np.mean(metrics['precision'][k])
    results['MRR'] = np.mean(metrics['mrr'])
    
    return results
```

## 总结

- **不要只用Top1 (Hit@1)**：太严格，建议至少看 Hit@5, Hit@10
- **推荐使用指标组合**：Hit@K + NDCG@K + MRR
- **根据场景选择K值**：
  - 召回阶段：K=10, 20, 50
  - 排序阶段：K=5, 10
  - 最终推荐：K=5（用户通常只看前5个推荐）

