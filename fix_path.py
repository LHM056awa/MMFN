with open('gossipcop_dataset.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 修改为使用平衡数据集
content = content.replace('train_gossipcop.csv', 'train_gossipcop_balanced.csv')

with open('gossipcop_dataset.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('✅ 已切换到平衡数据集')
