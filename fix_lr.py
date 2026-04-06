with open('trainMMFN.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('lr=5e-4', 'lr=3e-4')
content = content.replace("'lr': 5e-6", "'lr': 3e-6")

with open('trainMMFN.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('✅ 学习率降低到 3e-4')
