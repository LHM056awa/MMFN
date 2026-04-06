import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, SwinModel


class MultiModal(nn.Module):
    def __init__(self, num_labels=2):
        super(MultiModal, self).__init__()
        # 文本编码器
        self.bert = BertModel.from_pretrained('./bert-base-chinese')
        # 图像编码器
        self.swin = SwinModel.from_pretrained("C:/Users/yuki3/Desktop/DWMF_yyt-master/swin-base-patch4-window7-224")
        # 将图像特征从 1024 映射到 768
        self.image_proj = nn.Linear(1024, 768)

        # 动态权重层（让模型自己学习文本和图像的重要性）
        self.weight_layer = nn.Sequential(
            nn.Linear(768 * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Softmax(dim=1)
        )

        # 分类器
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids, image_swin, image_clip=None, text_clip=None,
                labels=None):
        # 文本特征
        text_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        text_features = text_outputs.pooler_output  # [batch, 768]

        # 图像特征
        image_outputs = self.swin(pixel_values=image_swin)
        image_features = image_outputs.pooler_output  # [batch, 1024]

        # 映射图像特征到 768
        image_proj = self.image_proj(image_features)  # [batch, 768]

        # 动态计算权重
        concat_features = torch.cat([text_features, image_proj], dim=1)  # [batch, 1536]
        weights = self.weight_layer(concat_features)  # [batch, 2]
        text_weight = weights[:, 0:1]  # [batch, 1]
        image_weight = weights[:, 1:2]  # [batch, 1]

        # 加权融合
        combined = text_weight * text_features + image_weight * image_proj  # [batch, 768]

        logits = self.classifier(combined)

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return loss, logits

        return logits