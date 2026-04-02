import pandas as pd

# 读取原始数据
df = pd.read_csv('data/gossipcop/train_gossipcop.csv')
print(f'原始数据: realnews={len(df[df["label"]==0])}, fakenews={len(df[df["label"]==1])}')

# 分离数据
realnews = df[df['label'] == 0]
fakenews = df[df['label'] == 1]

# 过采样 realnews 到与 fakenews 相同数量
realnews_oversampled = realnews.sample(n=len(fakenews), replace=True, random_state=42)

# 合并
balanced_df = pd.concat([fakenews, realnews_oversampled], ignore_index=True)

# 打乱顺序
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

# 保存
balanced_df.to_csv('data/gossipcop/train_gossipcop_balanced.csv', index=False)

print(f'平衡后: realnews={len(balanced_df[balanced_df["label"]==0])}, fakenews={len(balanced_df[balanced_df["label"]==1])}')
print('✅ 平衡数据集已创建')
