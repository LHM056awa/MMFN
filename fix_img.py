import pandas as pd

# 1. 查看 CSV 结构
df = pd.read_csv('data/gossipcop/train_gossipcop.csv')
print('列名:', list(df.columns))

# 找到包含图片的列
img_col = None
for col in ['image', 'images']:
    if col in df.columns:
        img_col = col
        break

print(f'将使用图片列: {img_col}')

# 2. 修复 gossipcop_dataset.py
with open('gossipcop_dataset.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 替换错误的索引访问为列名访问
old_code = 'images_name = str(gc.iloc[i, 8])'
new_code = f'images_name = str(gc.iloc[i]["{img_col}"])'

if old_code in content:
    content = content.replace(old_code, new_code)
    with open('gossipcop_dataset.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print(f'✅ 已修复: {old_code} -> {new_code}')
else:
    print('未找到需要修改的代码，尝试其他方式...')
    # 尝试其他可能的代码模式
    patterns = [
        ('gc.iloc[i, 8]', f'gc.iloc[i]["{img_col}"]'),
        ('gc.iloc[i, 7]', f'gc.iloc[i]["{img_col}"]'),
        ('gc["image"][i]', f'gc["{img_col}"][i]'),
    ]
    for old, new in patterns:
        if old in content:
            content = content.replace(old, new)
            with open('gossipcop_dataset.py', 'w', encoding='utf-8') as f:
                f.write(content)
            print(f'✅ 已修复: {old} -> {new}')
            break

print('\n修复完成！')
