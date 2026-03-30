# env_test.py
import torch
import transformers
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import jieba
import nltk
import sklearn
import matplotlib
import tqdm

print('='*60)
print('🎉 MMFN 项目 - 最终验证')
print('='*60)

print(f'\n📦 核心框架:')
print(f'  ✅ PyTorch: {torch.__version__}')
print(f'  ✅ Transformers: {transformers.__version__}')

print(f'\n🚀 GPU 状态:')
print(f'  ✅ CUDA 可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  ✅ CUDA 版本: {torch.version.cuda}')
    print(f'  ✅ GPU 名称: {torch.cuda.get_device_name(0)}')
    print(f'  ✅ 显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')
    # 测试 CUDA 运算
    x = torch.tensor([1.0, 2.0]).cuda()
    y = x * 2
    print(f'  ✅ CUDA 运算测试: {y.cpu().numpy()}')

print(f'\n📊 数据处理库:')
print(f'  ✅ NumPy: {np.__version__}')
print(f'  ✅ Pandas: {pd.__version__}')
print(f'  ✅ Scikit-learn: {sklearn.__version__}')

print(f'\n🖼️  图像处理库:')
print(f'  ✅ OpenCV: {cv2.__version__}')
print(f'  ✅ Pillow: {Image.__version__}')

print(f'\n📝 文本处理库:')
print(f'  ✅ Jieba: {jieba.__version__}')
print(f'  ✅ NLTK: {nltk.__version__}')

print(f'\n🔧 工具库:')
print(f'  ✅ Matplotlib: {matplotlib.__version__}')
print(f'  ✅ TQDM: {tqdm.__version__}')

print('\n' + '='*60)
print('✅ 所有依赖检查通过！环境配置完成')
print('='*60)