with open('trainMMFN.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 修改权重
content = content.replace(
    'weight=torch.tensor([5.0, 1.0]).cuda()',
    'weight=torch.tensor([6.0, 1.0]).cuda()'
)

with open('trainMMFN.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('✅ 权重改为 6:1')
