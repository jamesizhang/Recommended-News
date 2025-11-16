"""
快速检查数据泄露的脚本
"""
import pandas as pd
import numpy as np

save_path = '../temp_results/'

# 读取训练数据
trn_df = pd.read_csv(save_path + 'trn_user_item_feats_df.csv')

print("=" * 60)
print("数据泄露检查")
print("=" * 60)

# 1. 检查特征与标签的关联度
print("\n1. 特征与标签的相关性（绝对值）:")
lgb_cols = ['sim0', 'time_diff0', 'word_diff0','sim_max', 'sim_min', 'sim_sum',
            'sim_mean', 'score','click_size', 'time_diff_mean', 'active_level',
            'click_environment','click_deviceGroup', 'click_os', 'click_country',
            'click_region','click_referrer_type', 'user_time_hob1', 'user_time_hob2',
            'words_hbo', 'category_id', 'created_at_ts','words_count']

correlations = {}
for col in lgb_cols:
    if col in trn_df.columns:
        corr = abs(trn_df[col].corr(trn_df['label']))
        correlations[col] = corr

# 按相关性排序
sorted_corr = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
print("\n相关性最高的特征:")
for col, corr in sorted_corr[:10]:
    print(f"  {col}: {corr:.4f}")

# 2. 检查是否有特征能完美区分标签
print("\n2. 检查是否有特征能完美区分标签:")
for col in lgb_cols:
    if col in trn_df.columns:
        # 检查正负样本的分布是否完全分离
        pos_values = set(trn_df[trn_df['label'] == 1][col].dropna().unique())
        neg_values = set(trn_df[trn_df['label'] == 0][col].dropna().unique())
        
        # 如果正负样本的值完全不重叠，可能是泄露
        if len(pos_values.intersection(neg_values)) == 0 and len(pos_values) > 0 and len(neg_values) > 0:
            print(f"  ⚠️  {col}: 正负样本值完全不重叠！")
            print(f"     正样本唯一值数量: {len(pos_values)}, 负样本唯一值数量: {len(neg_values)}")

# 3. 检查 score 字段
print("\n3. 检查 score 字段:")
if 'score' in trn_df.columns:
    print(f"  score 与 label 的相关性: {abs(trn_df['score'].corr(trn_df['label'])):.4f}")
    print(f"  正样本 score 均值: {trn_df[trn_df['label']==1]['score'].mean():.4f}")
    print(f"  负样本 score 均值: {trn_df[trn_df['label']==0]['score'].mean():.4f}")
    print(f"  正样本 score 范围: [{trn_df[trn_df['label']==1]['score'].min():.4f}, {trn_df[trn_df['label']==1]['score'].max():.4f}]")
    print(f"  负样本 score 范围: [{trn_df[trn_df['label']==0]['score'].min():.4f}, {trn_df[trn_df['label']==0]['score'].max():.4f}]")

print("\n" + "=" * 60)
print("检查完成！如果看到相关性接近1.0或值完全不重叠的特征，就是泄露源。")
print("=" * 60)

