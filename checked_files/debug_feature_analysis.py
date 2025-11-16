"""
分析用户特征对模型性能的影响
"""
import pandas as pd
import numpy as np

data_path = '../data_raw/'

# 加载数据
all_click = pd.read_csv(data_path + 'train_click_log.csv', nrows=10000)

user_feature_cols = ['click_environment', 'click_deviceGroup', 'click_os', 
                     'click_country', 'click_region', 'click_referrer_type']

print("=" * 80)
print("用户特征统计分析")
print("=" * 80)

for col in user_feature_cols:
    if col in all_click.columns:
        print(f"\n{col}:")
        print(f"  唯一值数量: {all_click[col].nunique()}")
        print(f"  缺失值数量: {all_click[col].isna().sum()}")
        print(f"  前5个最常见的值:")
        print(all_click[col].value_counts().head())
        
        # 检查用户特征的一致性（每个用户的特征值是否一致）
        user_mode = all_click.groupby('user_id')[col].agg(lambda x: x.value_counts().index[0] if len(x) > 0 else x.iloc[0]).reset_index()
        user_mode.columns = ['user_id', col]
        merged = all_click.merge(user_mode, on='user_id', how='left', suffixes=('', '_mode'))
        inconsistent = (merged[col] != merged[col + '_mode']).sum()
        print(f"  用户特征不一致的记录数: {inconsistent} / {len(all_click)} ({inconsistent/len(all_click)*100:.2f}%)")

print("\n" + "=" * 80)
print("可能导致召回率降低的原因:")
print("=" * 80)
print("1. 特征基数过高：如果特征的唯一值很多，embedding矩阵会很大，但参数可能不足")
print("2. 特征噪声：如果用户特征在不同点击间变化很大，可能引入噪声")
print("3. 特征分布不均：某些特征值出现频率很低，embedding学习不充分")
print("4. 模型容量不足：增加了6个特征但没有增加模型容量")
print("5. 特征交互冲突：用户特征可能与历史序列特征存在竞争关系")
print("6. 数据量不足：小数据集（10000条）可能不足以学习复杂的特征组合")

