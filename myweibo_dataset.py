import torch
import pandas as pd
from PIL import Image
import os
from transformers import BertTokenizer, AutoFeatureExtractor
import clip

# 加载模型
token = BertTokenizer.from_pretrained('./bert-base-chinese')
feature_extractor = AutoFeatureExtractor.from_pretrained("./swin-base-patch4-window7-224")
clip_model, preprocess = clip.load("ViT-B/32", device="cpu")


class WeiboDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, img_root='data/weibo/'):
        self.data = pd.read_csv(data_path)
        self.data = self.data.fillna('')
        self.img_root = img_root

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        sent = str(row['text']) if 'text' in row else ''
        label = int(row['label']) if 'label' in row else 0

        # 直接从 CSV 的 image 列获取图片路径
        img_path = os.path.join(self.img_root, row['image'])

        # 加载图片，如果不存在就用空白图片
        if os.path.exists(img_path):
            img = Image.open(img_path).convert('RGB')
        else:
            img = Image.new('RGB', (224, 224), color='gray')

        # Swin 特征
        image_swin = feature_extractor(images=img, return_tensors="pt")['pixel_values'].squeeze(0)
        # CLIP 图像特征
        image_clip = preprocess(img)
        # CLIP 文本特征
        text_clip = clip.tokenize([sent], truncate=True).squeeze(0)

        return (sent, image_swin, image_clip, text_clip), label


def collate_fn(data):
    sents = [i[0][0] for i in data]
    image_swins = [i[0][1] for i in data]
    image_clips = [i[0][2] for i in data]
    text_clips = [i[0][3] for i in data]
    labels = [i[1] for i in data]

    # 文本处理
    text_data = token.batch_encode_plus(
        batch_text_or_text_pairs=sents,
        truncation=True,
        padding='max_length',
        max_length=300,
        return_tensors='pt'
    )

    input_ids = text_data['input_ids']
    attention_mask = text_data['attention_mask']
    token_type_ids = text_data.get('token_type_ids', None)

    # 图像处理
    image_swin = torch.stack(image_swins)
    image_clip = torch.stack(image_clips)
    text_clip = torch.stack(text_clips)
    labels = torch.LongTensor(labels)

    return input_ids, attention_mask, token_type_ids, image_swin, image_clip, text_clip, labels