"""
检查 Transformer 召回提交文件是否覆盖所有测试集用户
"""
import pandas as pd
import os

data_path = '../data_raw/'
save_path = '../temp_results/'

print("=" * 80)
print("检查 Transformer 召回提交文件覆盖率")
print("=" * 80)

# 1. 读取测试集用户
print("\n1. 读取测试集用户...")
try:
    test_click = pd.read_csv(os.path.join(data_path, 'testA_click_log.csv'))
    test_users = set(test_click['user_id'].unique())
    print(f"  测试集总用户数: {len(test_users)}")
    print(f"  测试集总记录数: {len(test_click)}")
except Exception as e:
    print(f"  ✗ 读取测试集失败: {e}")
    test_users = set()

# 2. 读取提交文件
print("\n2. 读取提交文件...")
submission_files = []
for file in os.listdir(save_path):
    if file.startswith('transformer_recall') and file.endswith('.csv') and 'sample' not in file:
        submission_files.append(file)

if not submission_files:
    print(f"  ✗ 未找到提交文件（在 {save_path} 目录下）")
    print(f"  当前目录下的文件: {os.listdir(save_path)[:10]}")
else:
    # 使用最新的提交文件（按修改时间排序）
    submission_files_with_time = []
    for file in submission_files:
        file_path = os.path.join(save_path, file)
        mtime = os.path.getmtime(file_path)
        submission_files_with_time.append((mtime, file))
    
    submission_files_with_time.sort(reverse=True)  # 最新的在前
    submission_file = submission_files_with_time[0][1]
    submission_path = os.path.join(save_path, submission_file)
    print(f"  找到提交文件: {submission_file} (最新)")
    print(f"  所有提交文件: {[f[1] for f in submission_files_with_time[:5]]}")
    
    try:
        submission_df = pd.read_csv(submission_path)
        submission_users = set(submission_df['user_id'].unique())
        print(f"  提交文件中的用户数: {len(submission_users)}")
        print(f"  提交文件总行数: {len(submission_df)}")
        
        # 3. 比较用户集合
        print("\n3. 用户覆盖率分析：")
        print(f"  测试集用户数: {len(test_users)}")
        print(f"  提交文件用户数: {len(submission_users)}")
        
        # 检查缺失的用户
        missing_users = test_users - submission_users
        extra_users = submission_users - test_users
        
        print(f"\n  缺失的用户数: {len(missing_users)}")
        if len(missing_users) > 0:
            print(f"  ⚠️  警告：{len(missing_users)} 个测试集用户未在提交文件中")
            if len(missing_users) <= 20:
                print(f"     缺失用户示例: {sorted(list(missing_users))[:20]}")
            else:
                print(f"     缺失用户示例（前20个）: {sorted(list(missing_users))[:20]}")
                print(f"     缺失用户示例（后20个）: {sorted(list(missing_users))[-20:]}")
        
        print(f"\n  额外的用户数: {len(extra_users)}")
        if len(extra_users) > 0:
            print(f"  ⚠️  警告：{len(extra_users)} 个用户只在提交文件中，不在测试集中")
            if len(extra_users) <= 20:
                print(f"     额外用户示例: {sorted(list(extra_users))[:20]}")
        
        # 覆盖率
        if len(test_users) > 0:
            coverage = len(submission_users & test_users) / len(test_users) * 100
            print(f"\n  用户覆盖率: {coverage:.2f}%")
            print(f"  覆盖的用户数: {len(submission_users & test_users)}")
        
        # 4. 检查每个用户是否有足够的文章
        print("\n4. 检查每个用户是否有足够的文章（topk=5）：")
        required_cols = ['article_1', 'article_2', 'article_3', 'article_4', 'article_5']
        missing_cols = [col for col in required_cols if col not in submission_df.columns]
        if missing_cols:
            print(f"  ✗ 提交文件缺少列: {missing_cols}")
        else:
            # 检查是否有空值
            users_with_missing = []
            for idx, row in submission_df.iterrows():
                missing_articles = [col for col in required_cols if pd.isna(row[col])]
                if missing_articles:
                    users_with_missing.append(row['user_id'])
            
            if len(users_with_missing) > 0:
                print(f"  ⚠️  警告：{len(users_with_missing)} 个用户的文章数不足5个")
                if len(users_with_missing) <= 20:
                    print(f"     文章不足的用户示例: {users_with_missing[:20]}")
            else:
                print(f"  ✓ 所有用户都有5篇文章")
        
        # 5. 统计信息
        print("\n5. 提交文件统计：")
        print(f"  总用户数: {len(submission_df)}")
        print(f"  唯一用户数: {submission_df['user_id'].nunique()}")
        print(f"  列名: {submission_df.columns.tolist()}")
        print(f"\n  前5行数据：")
        print(submission_df.head())
        
    except Exception as e:
        print(f"  ✗ 读取提交文件失败: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 80)
print("检查完成")
print("=" * 80)

