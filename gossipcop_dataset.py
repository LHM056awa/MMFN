import torch
import torch.utils.data as data
import data.util as util
import torchvision.transforms.functional as F
import pandas
from PIL import Image
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer, AutoFeatureExtractor
import clip
import os
# Determine whether to use CUDA (GPU) or CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP model and its preprocessing function
clipmodel, preprocess = clip.load('ViT-B/32', device)

# Freeze the parameters of the CLIP model
for param in clipmodel.parameters():
    param.requires_grad = False

# Load a feature extractor from the transformers library
feature_extractor = AutoFeatureExtractor.from_pretrained("./swin-base-patch4-window7-224")
token = BertTokenizer.from_pretrained('bert-base-chinese')


def read_img(img_path, root_path, img_dir):
    full_path = os.path.join(root_path, img_dir, img_path)
    try:
        img = Image.open(full_path).convert('RGB')
        img = img.resize((224, 224), Image.Resampling.LANCZOS)
        img_GT = np.array(img).astype(np.float32) / 255.0
        img_pro = img
        return img_GT, img_pro
    except Exception as e:
        print(f"图片加载失败: {full_path}")
        img_GT = np.zeros((224, 224, 3), dtype=np.float32)
        img_pro = Image.new('RGB', (224, 224), (255, 255, 255))
        return img_GT, img_pro


class gossipcop_dataset(data.Dataset):
    def __init__(self, is_train=True):
        super(gossipcop_dataset, self).__init__()
        self.label_dict = []
        self.swin = feature_extractor
        self.preprocess = preprocess
        self.local_path = r"E:\MMFN-master\data\gossipcop"
        # Read CSV file to populate label_dict
        if is_train:
            self.img_dir = "gossip_train"
            csv_path = os.path.join(self.local_path, "train_gossipcop.csv")
        else:
            self.img_dir = "gossip_test"
            csv_path = os.path.join(self.local_path, "test_gossipcop.csv")

        print(f'加载数据: {csv_path}')
        gc = pandas.read_csv(csv_path)

        # gc = gc[:1000]
        # Populate label_dict with records from the CSV file
        for i in tqdm(range(len(gc))):
            images_name = str(gc.iloc[i]["image"])
            label = int(gc.iloc[i]["label"])
            content = str(gc.iloc[i]["content"])
            sum_content = str(gc.iloc[i]["content"])
            has_image = gc.iloc[i]["has_image"]
            record = {}
            record['images'] = images_name
            record['label'] = label
            record['content'] = content
            record['sum_content'] = sum_content
            record['has_image'] = has_image
            self.label_dict.append(record)
        assert len(self.label_dict) != 0, 'Error: GT path is empty.'

    def __getitem__(self, item):
        record = self.label_dict[item]
        images_name = record['images']
        label = record['label']
        content = record['content']
        sum_content = record['sum_content']
        has_image = record['has_image']
        if label == 0:
            LABLEF = 'rumor_images'
        else:
            LABLEF = 'nonrumor_images'
        imgs = images_name.split('|')
        if has_image:
             img_GT, img_pro = read_img(images_name, self.local_path, self.img_dir)
        else:
            img_GT = np.zeros((224, 224, 3))
            img_pro = Image.new('RGB', (224, 224), (255, 255, 255)).convert('RGB')
            # raise IOError("Load {} Error {}".format(imgs, record['images']))
        sent = content
        image_swin = self.swin(img_GT, return_tensors="pt").pixel_values
        image_clip = self.preprocess(img_pro)
        text_clip = sum_content
        return (sent, image_swin, image_clip, text_clip), label

    def __len__(self):
        return len(self.label_dict)

    def to_tensor(self, img):
        img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        return img_t

    pass



def collate_fn(data):
    sents = [i[0][0] for i in data]
    image = [i[0][1] for i in data]
    imageclip = [i[0][2] for i in data]
    textclip = [i[0][3] for i in data]
    labels = [i[1] for i in data]

    data = token.batch_encode_plus(batch_text_or_text_pairs=sents,
                                   truncation=True,
                                   padding='max_length',
                                   max_length=300,
                                   return_tensors='pt',
                                   return_length=True)

    textclip = clip.tokenize(textclip, truncate=True)
    input_ids = data['input_ids']
    attention_mask = data['attention_mask']
    token_type_ids = data['token_type_ids']
    image = torch.stack(image).squeeze()
    imageclip = torch.stack(imageclip)
    labels = torch.LongTensor(labels)
    return input_ids, attention_mask, token_type_ids, image, imageclip, textclip, labels

