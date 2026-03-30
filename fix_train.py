import re

with open('trainMMFN.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. 添加类别权重
if 'weight=torch.tensor' not in content:
    content = content.replace(
        'loss_f_rumor = torch.nn.CrossEntropyLoss()',
        'loss_f_rumor = torch.nn.CrossEntropyLoss(weight=torch.tensor([4.0, 1.0]).cuda())'
    )
    print('✅ 已添加类别权重: realnews x4, fakenews x1')

# 2. 降低学习率
content = content.replace('lr=1e-3', 'lr=5e-4')
content = content.replace("'lr': 1e-5", "'lr': 5e-6")
print('✅ 学习率: 1e-3 -> 5e-4')

# 3. 增加 batch_size
content = content.replace('batch_size=2', 'batch_size=4')
print('✅ batch_size: 2 -> 4')

# 4. 增加 patience
content = content.replace('patience = 15', 'patience = 30')
print('✅ patience: 15 -> 30')

with open('trainMMFN.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('\n✅ 所有修改完成！')
