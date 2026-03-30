import torch
import torch.nn as nn
from transformers import BertModel, SwinModel

class MultiModal(nn.Module):
    def __init__(self, num_labels=2):
        super(MultiModal, self).__init__()
        # 文本编码器
        self.bert = BertModel.from_pretrained('./bert-base-chinese')
        # 图像编码器
        self.swin = SwinModel.from_pretrained("./swin-base-patch4-window7-224")
        # 新增：图像特征映射层
        self.image_proj = nn.Linear(1024, 768)
        # 分类器（维度改为768）
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids, image_swin, image_clip, text_clip, labels=None):
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

        # 映射图像特征到768维
        image_proj = self.image_proj(image_features)  # [batch, 768]

        # 加权融合
        alpha = 0.7   # 文本权重
        beta = 0.3    # 图像权重
        combined = alpha * text_features + beta * image_proj  # [batch, 768]
        logits = self.classifier(combined)

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return loss, logits
        return logits
