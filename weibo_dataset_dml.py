import os
import torch
import torch.utils.data as data
import data.util as util
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import pandas
from PIL import Image
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer
import clip

# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 优先使用 DML，否则回退到 CPU
try:
    import torch_directml
    device = torch_directml.device()
    print(f"✅ 数据集使用 DML 设备: {device}")
except:
    device = "cpu"
    print("⚠️ DML 不可用，数据集使用 CPU")

# 加载 CLIP 模型到 DML 设备
clipmodel, preprocess = clip.load('ViT-B/32', device)
for param in clipmodel.parameters():
    param.requires_grad = False

# 定义 Swin Transformer 的图像预处理
# Swin 要求：224x224，归一化均值/标准差为 ImageNet 标准
swin_transform = transforms.Compose([
    transforms.Resize(256),               # 先缩放到 256
    transforms.CenterCrop(224),           # 再中心裁剪到 224
    transforms.ToTensor(),                # 转为 [0,1] 的 tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 加载 BERT tokenizer（仍然使用 transformers）
token = BertTokenizer.from_pretrained('./data/bert-chinese', local_files_only=True)


def read_img(imgs, root_path, LABLEF):
    GT_path = imgs[np.random.randint(0, len(imgs))]
    GT_path = GT_path.split('/')[-1]
    GT_path = os.path.join(root_path, LABLEF, GT_path)

    try:
        img_GT = util.read_img(GT_path)          # numpy 数组 (H,W,C)
        img_pro = Image.open(GT_path).convert('RGB')
    except Exception as e:
        img_GT = np.zeros((224, 224, 3))
        img_pro = Image.new('RGB', (224, 224), (255, 255, 255)).convert('RGB')
        print("找不到图片，使用默认图:", GT_path)
    return img_GT, img_pro


class weibo_dataset(data.Dataset):
    def __init__(self, is_train=True):
        super(weibo_dataset, self).__init__()
        self.label_dict = []
        self.swin_transform = swin_transform    # 替换原来的 feature_extractor
        self.preprocess = preprocess
        self.local_path = os.path.join(current_dir, "weibo")

        csv_path = os.path.join(
            self.local_path,
            '{}_weibo_final3.csv'.format('train' if is_train else 'test')
        )
        gc = pandas.read_csv(csv_path)

        for i in tqdm(range(len(gc))):
            images_name = str(gc.iloc[i, 1])
            label = int(gc.iloc[i, 2])
            content = str(gc.iloc[i, 4])
            sum_content = str(gc.iloc[i, 4])
            has_image = gc.iloc[i, 6]
            record = {
                'images': images_name,
                'label': label,
                'content': content,
                'sum_content': sum_content,
                'has_image': has_image
            }
            self.label_dict.append(record)

    def __getitem__(self, item):
        record = self.label_dict[item]
        images, label, content, sum_content, has_image = (
            record['images'],
            record['label'],
            record['content'],
            record['sum_content'],
            record['has_image']
        )

        if label == 0:
            LABLEF = 'rumor_images'
        else:
            LABLEF = 'nonrumor_images'

        imgs = images.split('|')
        if has_image:
            img_GT, img_pro = read_img(imgs, self.local_path, LABLEF)
        else:
            img_GT = np.zeros((224, 224, 3))
            img_pro = Image.new('RGB', (224, 224), (255, 255, 255)).convert('RGB')

        sent = content
        # 使用 torchvision 预处理图片
        # 将 numpy 数组转为 PIL Image（因为 transforms 要求 PIL 或 tensor）
        image_pil = Image.fromarray(img_GT.astype('uint8')).convert('RGB')
        image_swin = self.swin_transform(image_pil).unsqueeze(0)  # [1,3,224,224]
        image_clip = self.preprocess(img_pro)
        text_clip = sum_content
        return (sent, image_swin, image_clip, text_clip), label

    def __len__(self):
        return len(self.label_dict)

    def to_tensor(self, img):
        img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        return img_t


def collate_fn(data):
    sents = [i[0][0] for i in data]
    image = [i[0][1] for i in data]
    imageclip = [i[0][2] for i in data]
    textclip = [i[0][3] for i in data]
    labels = [i[1] for i in data]

    data = token.batch_encode_plus(
        batch_text_or_text_pairs=sents,
        truncation=True,
        padding='max_length',
        max_length=300,
        return_tensors='pt',
        return_length=True
    )

    textclip = clip.tokenize(textclip, truncate=True)
    input_ids = data['input_ids']
    attention_mask = data['attention_mask']
    token_type_ids = data['token_type_ids']

    # 拼接 batch：注意 image 已经是 [batch,1,3,224,224]，需要去掉第1维
    image = torch.cat(image, dim=0)          # 原 list of [1,3,224,224] → [batch,3,224,224]
    imageclip = torch.stack(imageclip)       # [batch,3,224,224]（CLIP 预处理后）
    labels = torch.LongTensor(labels)

    return input_ids, attention_mask, token_type_ids, image, imageclip, textclip, labels


if __name__ == "__main__":
    print("当前文件所在目录:", current_dir)
    dataset = weibo_dataset(is_train=True)
    print("数据集根目录:", dataset.local_path)
    print("CSV 文件存在:", os.path.exists(os.path.join(dataset.local_path, 'train_weibo_final3.csv')))
    print("rumor_images 存在:", os.path.exists(os.path.join(dataset.local_path, 'rumor_images')))
    print("nonrumor_images 存在:", os.path.exists(os.path.join(dataset.local_path, 'nonrumor_images')))
    try:
        sample, label = dataset[0]
        print("样本加载成功，标签:", label)
        sent, image_swin, image_clip, text_clip = sample
        print("文本内容长度:", len(sent))
        print("Swin 特征形状:", image_swin.shape)
        print("CLIP 图像特征形状:", image_clip.shape)
        print("CLIP 文本 token 数量:", len(text_clip))
    except Exception as e:
        print("样本加载失败:", e)