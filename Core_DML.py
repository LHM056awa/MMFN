from random import random
import torch
import torch.nn as nn
import math
import random
import torch.backends.cudnn as cudnn
import numpy as np
import copy
from transformers import BertConfig, BertModel, SwinModel

# Set a manual seed for reproducibility
manualseed = 666
random.seed(manualseed)
np.random.seed(manualseed)
torch.manual_seed(manualseed)
torch.cuda.manual_seed(manualseed)
cudnn.deterministic = True

model_name = "bert-base-chinese"
config = BertConfig.from_pretrained("./data/bert-chinese", num_labels=2)
config.output_hidden_states = False


class Transformer(nn.Module):
    def __init__(self, model_dimension, number_of_heads, number_of_layers, dropout_probability,
                 log_attention_weights=False):
        super().__init__()
        mha = MultiHeadedAttention(model_dimension, number_of_heads, dropout_probability, log_attention_weights)
        encoder_layer = EncoderLayer(model_dimension, dropout_probability, mha)
        self.encoder = Encoder(encoder_layer, number_of_layers)
        self.init_params()

    def init_params(self, default_initialization=False):
        if not default_initialization:
            for name, p in self.named_parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def forward(self, text, image):
        src1 = self.encode(text, image)
        src2 = self.encode(image, text)
        return src1, src2

    def encode(self, src1, src2):
        return self.encoder(src1, src2)


class Encoder(nn.Module):
    def __init__(self, encoder_layer, number_of_layers):
        super().__init__()
        assert isinstance(encoder_layer, EncoderLayer), f'Expected EncoderLayer got {type(encoder_layer)}.'
        self.encoder_layers = get_clones(encoder_layer, number_of_layers)
        self.norm = nn.LayerNorm(encoder_layer.model_dimension)

    def forward(self, src1, src2):
        for encoder_layer in self.encoder_layers:
            src = encoder_layer(src1, src2)
        return self.norm(src)


class EncoderLayer(nn.Module):
    def __init__(self, model_dimension, dropout_probability, multi_headed_attention):
        super().__init__()
        num_of_sublayers_encoder = 2
        self.sublayers = get_clones(SublayerLogic(model_dimension, dropout_probability), num_of_sublayers_encoder)
        self.multi_headed_attention = multi_headed_attention
        self.model_dimension = model_dimension

    def forward(self, srb1, srb2):
        encoder_self_attention = lambda srb1, srb2: self.multi_headed_attention(query=srb1, key=srb2, value=srb2)
        return self.sublayers[0](srb1, srb2, encoder_self_attention)


class SublayerLogic(nn.Module):
    def __init__(self, model_dimension, dropout_probability):
        super().__init__()
        self.norm = nn.LayerNorm(model_dimension)
        self.dropout = nn.Dropout(p=dropout_probability)

    def forward(self, srb1, srb2, sublayer_module):
        return srb1 + self.dropout(sublayer_module(self.norm(srb1), self.norm(srb2)))


class MultiHeadedAttention(nn.Module):
    def __init__(self, model_dimension, number_of_heads, dropout_probability, log_attention_weights):
        super().__init__()
        assert model_dimension % number_of_heads == 0, f'Model dimension must be divisible by the number of heads.'
        self.head_dimension = int(model_dimension / number_of_heads)
        self.number_of_heads = number_of_heads
        self.qkv_nets = get_clones(nn.Linear(model_dimension, model_dimension), 3)
        self.out_projection_net = nn.Linear(model_dimension, model_dimension)
        self.attention_dropout = nn.Dropout(p=dropout_probability)
        self.softmax = nn.Softmax(dim=-1)
        self.log_attention_weights = log_attention_weights
        self.attention_weights = None

    def attention(self, query, key, value):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dimension)
        attention_weights = self.softmax(scores)
        attention_weights = self.attention_dropout(attention_weights)
        intermediate = torch.matmul(attention_weights, value)
        return intermediate, attention_weights

    def forward(self, query, key, value):
        batch_size = query.shape[0]
        query, key, value = [net(x).view(batch_size, -1, self.number_of_heads, self.head_dimension).transpose(1, 2)
                             for net, x in zip(self.qkv_nets, (query, key, value))]
        intermediate, attn_weights = self.attention(query, key, value)
        if self.log_attention_weights:
            self.attention_weights = attn_weights
        reshaped = intermediate.transpose(1, 2).reshape(batch_size, -1,
                                                        self.number_of_heads * self.head_dimension)
        return self.out_projection_net(reshaped)


def get_clones(module, num):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num)])


class UnimodalDetection(nn.Module):
    def __init__(self, shared_dim=256, prime_dim=16):
        super(UnimodalDetection, self).__init__()
        self.text_uni = nn.Sequential(
            nn.Linear(1280, shared_dim), nn.BatchNorm1d(shared_dim), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(shared_dim, prime_dim), nn.BatchNorm1d(prime_dim), nn.ReLU())
        self.image_uni = nn.Sequential(
            nn.Linear(1536, shared_dim), nn.BatchNorm1d(shared_dim), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(shared_dim, prime_dim), nn.BatchNorm1d(prime_dim), nn.ReLU())

    def forward(self, text_enc, image_enc):
        return self.text_uni(text_enc), self.image_uni(image_enc)


class CrossModule(nn.Module):
    def __init__(self, corre_out_dim=16):
        super(CrossModule, self).__init__()
        self.corre_dim = 1024
        self.c_specific_1 = nn.Sequential(nn.Linear(self.corre_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.1))
        self.c_specific_2 = nn.Sequential(nn.Linear(self.corre_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.1))
        self.c_specific_3 = nn.Sequential(
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, corre_out_dim), nn.BatchNorm1d(corre_out_dim), nn.ReLU())

    def forward(self, text, image, text1, image1):
        out1 = self.c_specific_1(torch.cat((text, image), 1).float())
        out2 = self.c_specific_2(torch.cat((text1, image1), 1).float())
        return self.c_specific_3(torch.cat((out1, out2), 1))


class MultiModal(nn.Module):
    def __init__(self, feature_dim=48, h_dim=48):
        super(MultiModal, self).__init__()
        self.w = nn.Parameter(torch.rand(1))
        self.b = nn.Parameter(torch.rand(1))

        self.trans = Transformer(model_dimension=512, number_of_heads=8, number_of_layers=1,
                                 dropout_probability=0.1, log_attention_weights=False)

        self.t_projection_net = nn.Linear(768, 512)
        self.i_projection_net = nn.Linear(1024, 512)

        # 模型加载时不指定 device，训练时会自动移动到 DML 设备
        self.swin = SwinModel.from_pretrained("./data/swin-base-patch4-window7-224", local_files_only=True)
        for param in self.swin.parameters():
            param.requires_grad = True

        self.bert = BertModel.from_pretrained("./data/bert-chinese", local_files_only=True)
        for param in self.bert.parameters():
            param.requires_grad = True

        self.uni_repre = UnimodalDetection()
        self.cross_module = CrossModule()
        self.fine_align_proj = nn.Linear(512, 16)

        self.classifier_corre = nn.Sequential(
            nn.Linear(feature_dim, h_dim), nn.BatchNorm1d(h_dim), nn.ReLU(), nn.Linear(h_dim, 2)
        )
        self.classifier_main = nn.Sequential(
            nn.Linear(64, h_dim), nn.BatchNorm1d(h_dim), nn.ReLU(), nn.Linear(h_dim, 2)
        )

    # 消融实验方法保持不变，使用 classifier_corre
    def forward_no_unimodal(self, input_ids, attention_mask, token_type_ids, image_raw, text, image):
        BERT_feature = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        last_hidden_states = BERT_feature['last_hidden_state']
        text_raw = torch.sum(last_hidden_states, dim=1) / 300
        image_raw = self.swin(image_raw)

        text_m = self.t_projection_net(last_hidden_states)
        image_m = self.i_projection_net(image_raw.last_hidden_state)
        text_att, image_att = self.trans(text_m, image_m)
        correlation = self.cross_module(text, image, torch.sum(text_att, dim=1) / 300, torch.sum(image_att, dim=1) / 49)
        sim = torch.div(torch.sum(text * image, 1), torch.sqrt(torch.sum(torch.pow(text, 2), 1)) * torch.sqrt(torch.sum(torch.pow(image, 2), 1)))
        sim = sim * self.w + self.b
        mweight = sim.unsqueeze(1)
        correlation = correlation * mweight
        final_feature = torch.cat([correlation, correlation, correlation], 1)
        return self.classifier_corre(final_feature)

    def forward_no_image(self, input_ids, attention_mask, token_type_ids, image_raw, text, image):
        BERT_feature = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        last_hidden_states = BERT_feature['last_hidden_state']
        text_raw = torch.sum(last_hidden_states, dim=1) / 300
        image_raw = self.swin(image_raw)
        text_prime, _ = self.uni_repre(torch.cat([text_raw, text], 1), torch.cat([image_raw.pooler_output, image.flatten(1)], 1))
        text_m = self.t_projection_net(last_hidden_states)
        sim = torch.div(torch.sum(text * text, 1), torch.sqrt(torch.sum(torch.pow(text, 2), 1)) * torch.sqrt(torch.sum(torch.pow(text, 2), 1)))
        sim = sim * self.w + self.b
        mweight = sim.unsqueeze(1)
        final_feature = torch.cat([text_prime, text_prime, text_prime], 1)
        return self.classifier_corre(final_feature)

    def forward_no_text(self, input_ids, attention_mask, token_type_ids, image_raw, text, image):
        BERT_feature = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        last_hidden_states = BERT_feature['last_hidden_state']
        text_raw = torch.sum(last_hidden_states, dim=1) / 300
        image_raw = self.swin(image_raw)
        text_prime, image_prime = self.uni_repre(torch.cat([text_raw, text], 1), torch.cat([image_raw.pooler_output, image.flatten(1)], 1))
        text_m = self.t_projection_net(last_hidden_states)
        image_m = self.i_projection_net(image_raw.last_hidden_state)
        text_att, image_att = self.trans(text_m, image_m)
        sim = torch.div(torch.sum(text * text, 1), torch.sqrt(torch.sum(torch.pow(text, 2), 1)) * torch.sqrt(torch.sum(torch.pow(text, 2), 1)))
        sim = sim * self.w + self.b
        mweight = sim.unsqueeze(1)
        final_feature = torch.cat([image_prime, image_prime, image_prime], 1)
        return self.classifier_corre(final_feature)

    def forward_no_clip(self, input_ids, attention_mask, token_type_ids, image_raw, text, image):
        BERT_feature = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        last_hidden_states = BERT_feature['last_hidden_state']
        text_raw = torch.sum(last_hidden_states, dim=1) / 300
        image_raw = self.swin(image_raw)
        text = torch.ones_like(text)
        image = torch.ones_like(image)
        text_prime, image_prime = self.uni_repre(torch.cat([text_raw, text], 1), torch.cat([image_raw.pooler_output, image.flatten(1)], 1))
        text_m = self.t_projection_net(last_hidden_states)
        image_m = self.i_projection_net(image_raw.last_hidden_state)
        text_att, image_att = self.trans(text_m, image_m)
        correlation = self.cross_module(text, image, torch.sum(text_att, dim=1) / 300, torch.sum(image_att, dim=1) / 49)
        final_feature = torch.cat([text_prime, image_prime, correlation], 1)
        return self.classifier_corre(final_feature)

    def forward_no_transformer(self, input_ids, attention_mask, token_type_ids, image_raw, text, image):
        BERT_feature = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        last_hidden_states = BERT_feature['last_hidden_state']
        text_raw = torch.sum(last_hidden_states, dim=1) / 300
        image_raw = self.swin(image_raw)

        text_prime, image_prime = self.uni_repre(torch.cat([text_raw, text], 1), torch.cat([image_raw.pooler_output, image.flatten(1)], 1))
        text_m = self.t_projection_net(last_hidden_states)
        image_m = self.i_projection_net(image_raw.last_hidden_state)
        correlation = self.cross_module(text, image, text_m, image_m)
        sim = torch.div(torch.sum(text * image, 1), torch.sqrt(torch.sum(torch.pow(text, 2), 1)) * torch.sqrt(torch.sum(torch.pow(image, 2), 1)))
        sim = sim * self.w + self.b
        mweight = sim.unsqueeze(1)
        correlation = correlation * mweight
        final_feature = torch.cat([text_prime, image_prime, correlation], 1)
        return self.classifier_corre(final_feature)

    def forward_no_weight(self, input_ids, attention_mask, token_type_ids, image_raw, text, image):
        BERT_feature = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        last_hidden_states = BERT_feature['last_hidden_state']
        text_raw = torch.sum(last_hidden_states, dim=1) / 300
        image_raw = self.swin(image_raw)

        text_prime, image_prime = self.uni_repre(torch.cat([text_raw, text], 1), torch.cat([image_raw.pooler_output, image.flatten(1)], 1))
        text_m = self.t_projection_net(last_hidden_states)
        image_m = self.i_projection_net(image_raw.last_hidden_state)
        text_att, image_att = self.trans(text_m, image_m)

        correlation = self.cross_module(text, image, torch.sum(text_att, dim=1) / 300, torch.sum(image_att, dim=1) / 49)
        final_feature = torch.cat([text_prime, image_prime, correlation], 1)
        return self.classifier_corre(final_feature)

    def forward_no_crossmodule(self, input_ids, attention_mask, token_type_ids, image_raw, text, image):
        BERT_feature = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        last_hidden_states = BERT_feature['last_hidden_state']
        text_raw = torch.sum(last_hidden_states, dim=1) / 300
        image_raw = self.swin(image_raw)

        text_prime, image_prime = self.uni_repre(torch.cat([text_raw, text], 1), torch.cat([image_raw.pooler_output, image.flatten(1)], 1))
        text_m = self.t_projection_net(last_hidden_states)
        image_m = self.i_projection_net(image_raw.last_hidden_state)
        text_att, image_att = self.trans(text_m, image_m)
        sim = torch.div(torch.sum(text * image, 1), torch.sqrt(torch.sum(torch.pow(text, 2), 1)) * torch.sqrt(torch.sum(torch.pow(image, 2), 1)))
        sim = sim * self.w + self.b
        mweight = sim.unsqueeze(1)
        final_feature = torch.cat([text_prime, image_prime, text_prime], 1)
        return self.classifier_corre(final_feature)

    # 主 forward，使用多粒度对齐模块和 classifier_main
    def forward(self, input_ids, attention_mask, token_type_ids, image_raw, text, image):
        BERT_feature = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        last_hidden_states = BERT_feature['last_hidden_state']
        text_raw = torch.sum(last_hidden_states, dim=1) / 300
        image_raw = self.swin(image_raw)

        text_prime, image_prime = self.uni_repre(torch.cat([text_raw, text], 1),
                                                 torch.cat([image_raw.pooler_output, image.flatten(1)], 1))

        text_m = self.t_projection_net(last_hidden_states)      # [b, seq_len, 512]
        image_m = self.i_projection_net(image_raw.last_hidden_state)  # [b, num_patches, 512]

        text_att, image_att = self.trans(text_m, image_m)

        correlation = self.cross_module(text, image, torch.sum(text_att, dim=1) / 300, torch.sum(image_att, dim=1) / 49)

        sim = torch.div(torch.sum(text * image, 1),
                        torch.sqrt(torch.sum(torch.pow(text, 2), 1)) * torch.sqrt(torch.sum(torch.pow(image, 2), 1)))
        sim = sim * self.w + self.b
        mweight = sim.unsqueeze(1)
        correlation = correlation * mweight

        # 多粒度对齐模块
        fine_grained_sim = torch.matmul(text_m, image_m.transpose(1, 2)) / math.sqrt(512)  # [b, seq_len, num_patches]
        att_weights = torch.softmax(fine_grained_sim, dim=-1)
        fine_grained_image = torch.matmul(att_weights, image_m)  # [b, seq_len, 512]
        fine_grained_align = torch.mean(fine_grained_image, dim=1)  # [b, 512]
        fine_align_proj = self.fine_align_proj(fine_grained_align)  # [b, 16]

        final_feature = torch.cat([text_prime, image_prime, correlation, fine_align_proj], 1)  # [b, 64]
        pre_label = self.classifier_main(final_feature)
        return pre_label