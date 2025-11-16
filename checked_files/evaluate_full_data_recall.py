"""
评估全量数据的召回率（非采样数据）
检查最后一次点击是否在召回结果中
"""
import pickle
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

save_path = '../temp_results/'
data_path = '../data_raw/'

print("=" * 80)
print("全量数据召回率评估（非采样数据）")
print("=" * 80)

# 1. 加载全量数据的最后一次点击
print("\n1. 加载全量测试数据...")
try:
    # 加载训练集和测试集
    trn_click = pd.read_csv(data_path + 'train_click_log.csv')
    tst_click = pd.read_csv(data_path + 'testA_click_log.csv')
    all_click_df = pd.concat([trn_click, tst_click], ignore_index=True)
    all_click_df = all_click_df.drop_duplicates(['user_id', 'click_article_id', 'click_timestamp'])
    
    print(f"  全量数据: {len(all_click_df)} 条记录")
    print(f"  用户数: {all_click_df['user_id'].nunique()}")
    
    # 获取最后一次点击
    def get_hist_and_last_click(all_click):
        all_click = all_click.sort_values(['user_id', 'click_timestamp'])
        click_list = []
        for user_id, hist_click in tqdm(all_click.groupby('user_id'), desc="处理用户"):
            click_list.append({
                'user_id': user_id,
                'click_article_id': hist_click['click_article_id'].tolist(),
                'click_timestamp': hist_click['click_timestamp'].tolist()
            })
        
        last_click_list = []
        for click in click_list:
            if len(click['click_article_id']) > 1:
                last_click_list.append({
                    'user_id': click['user_id'],
                    'click_article_id': click['click_article_id'][-1],
                    'click_timestamp': click['click_timestamp'][-1]
                })
        
        return pd.DataFrame(last_click_list)
    
    trn_last_click_df = get_hist_and_last_click(all_click_df)
    print(f"  最后一次点击数据: {len(trn_last_click_df)} 用户")
    
except Exception as e:
    print(f"  ✗ 加载数据失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 2. 检查召回结果文件
print("\n2. 检查召回结果文件...")
recall_files = {
    'itemcf_sim_itemcf_recall': 'itemcf_u2i_dict.pkl',
    'embedding_sim_item_recall': 'embedding_u2i_dict.pkl',
    'youtubednn_recall': 'youtube_u2i_dict.pkl',
    'youtubednn_usercf_recall': 'youtube_u2u_sim.pkl',
    'cold_start_recall': 'cold_start_user_items_dict.pkl',
    'final_combined_recall': 'final_recall_items_dict.pkl'
}

file_info = {}
for method, filename in recall_files.items():
    filepath = os.path.join(save_path, filename)
    if os.path.exists(filepath):
        file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
        file_info[method] = {'path': filepath, 'size_mb': file_size}
        print(f"  ✓ {method}: {file_size:.2f} MB")
    else:
        print(f"  ✗ {method}: 文件不存在")

# 3. 加载并评估召回结果
print("\n3. 加载并评估召回结果...")
def metrics_recall(user_recall_items_dict, trn_last_click_df, topk_list=[10, 20, 50, 100, 150]):
    """评估召回率"""
    last_click_item_dict = dict(zip(trn_last_click_df['user_id'], trn_last_click_df['click_article_id']))
    
    results = {}
    for k in topk_list:
        hit_num = 0
        valid_user_num = 0
        total_recall_num = 0
        
        for user, item_list in user_recall_items_dict.items():
            if user not in last_click_item_dict:
                continue
            valid_user_num += 1
            
            # 获取前k个召回的结果
            if isinstance(item_list, list):
                tmp_recall_items = [x[0] if isinstance(x, tuple) else x for x in item_list[:k]]
                total_recall_num += len(tmp_recall_items)
            elif isinstance(item_list, dict):
                tmp_recall_items = list(item_list.keys())[:k]
                total_recall_num += len(tmp_recall_items)
            else:
                continue
            
            if last_click_item_dict[user] in tmp_recall_items:
                hit_num += 1
        
        if valid_user_num > 0:
            hit_rate = hit_num / valid_user_num
            avg_recall_num = total_recall_num / valid_user_num if valid_user_num > 0 else 0
            results[k] = {
                'hit_num': hit_num, 
                'hit_rate': hit_rate, 
                'valid_user_num': valid_user_num,
                'avg_recall_num': avg_recall_num
            }
        else:
            results[k] = {'hit_num': 0, 'hit_rate': 0.0, 'valid_user_num': 0, 'avg_recall_num': 0}
    
    return results

# 评估各路召回
all_results = {}
for method, file_info_item in file_info.items():
    if method == 'final_combined_recall':
        continue  # 最后单独处理
    
    filepath = file_info_item['path']
    try:
        recall_dict = pickle.load(open(filepath, 'rb'))
        print(f"\n  {method}:")
        print(f"    召回用户数: {len(recall_dict)}")
        
        # 检查是否是采样数据（用户数较少）
        if len(recall_dict) < 50000:
            print(f"    ⚠️  警告: 召回用户数较少，可能是采样数据")
        
        results = metrics_recall(recall_dict, trn_last_click_df)
        all_results[method] = results
        
        for k, res in results.items():
            print(f"    Top-{k:3d}: hit_num={res['hit_num']:5d}, hit_rate={res['hit_rate']:.5f}, "
                  f"valid_users={res['valid_user_num']}, avg_recall={res['avg_recall_num']:.1f}")
    except Exception as e:
        print(f"  ✗ {method}: 加载失败 - {e}")

# 4. 评估合并后的召回结果
print("\n4. 合并后的召回率评估:")
if 'final_combined_recall' in file_info:
    filepath = file_info['final_combined_recall']['path']
    try:
        final_recall_data = pickle.load(open(filepath, 'rb'))
        
        # 检查数据格式并转换
        if isinstance(final_recall_data, dict):
            first_user = list(final_recall_data.keys())[0]
            first_value = final_recall_data[first_user]
            
            if isinstance(first_value, list):
                final_recall_dict = final_recall_data
            elif isinstance(first_value, dict):
                final_recall_dict = {}
                for user, item_dict in final_recall_data.items():
                    final_recall_dict[user] = sorted(item_dict.items(), key=lambda x: x[1], reverse=True)
            else:
                print(f"  ✗ 未知的数据格式: {type(first_value)}")
                final_recall_dict = {}
        else:
            print(f"  ✗ 未知的数据格式: {type(final_recall_data)}")
            final_recall_dict = {}
        
        if len(final_recall_dict) > 0:
            print(f"  ✓ 加载成功: {len(final_recall_dict)} 用户")
            
            if len(final_recall_dict) < 50000:
                print(f"  ⚠️  警告: 召回用户数较少，可能是采样数据")
            
            combined_results = metrics_recall(final_recall_dict, trn_last_click_df)
            all_results['final_combined_recall'] = combined_results
            
            print("\n  合并后的召回率:")
            for k, res in combined_results.items():
                print(f"    Top-{k:3d}: hit_num={res['hit_num']:5d}, hit_rate={res['hit_rate']:.5f}, "
                      f"valid_users={res['valid_user_num']}, avg_recall={res['avg_recall_num']:.1f}")
        else:
            print("  ✗ 数据格式不正确")
    except Exception as e:
        print(f"  ✗ 加载失败: {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"  ✗ 文件不存在")

# 5. 对比总结
print("\n5. 召回率对比总结 (Top-50):")
print(f"  {'召回方法':<30} {'命中数':<10} {'召回率':<12} {'有效用户':<10} {'平均召回数':<10}")
print("  " + "-" * 80)
for method, results in all_results.items():
    if 50 in results:
        res = results[50]
        print(f"  {method:<30} {res['hit_num']:<10} {res['hit_rate']:<12.5f} "
              f"{res['valid_user_num']:<10} {res['avg_recall_num']:<10.1f}")

# 6. 检查是否是采样数据
print("\n6. 数据检查:")
print(f"  全量数据用户数: {all_click_df['user_id'].nunique()}")
print(f"  最后一次点击用户数: {len(trn_last_click_df)}")
for method, results in all_results.items():
    if 50 in results:
        recall_user_num = results[50]['valid_user_num']
        coverage = recall_user_num / len(trn_last_click_df) * 100
        print(f"  {method}: 召回用户数={recall_user_num}, 覆盖率={coverage:.2f}%")
        if coverage < 50:
            print(f"    ⚠️  警告: 覆盖率较低，可能是采样数据")

print("\n" + "=" * 80)
print("提示: 如果召回用户数远少于全量用户数，说明使用的是采样数据")
print("需要运行全量数据的召回才能获得准确的召回率评估")
print("=" * 80)

