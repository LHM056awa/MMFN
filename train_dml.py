import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, \
    classification_report
from transformers import logging
from torch.autograd import Variable
from torch.utils.data import DataLoader
from Core_DML import MultiModal
from tqdm import tqdm
from weibo_dataset_dml import weibo_dataset, collate_fn, clipmodel

logging.set_verbosity_warning()
logging.set_verbosity_error()

# ========== 设备设置：优先使用 DML ==========
try:
    import torch_directml
    device = torch_directml.device()
    print(f"✅ 训练使用 DirectML 设备: {device}")
except:
    device = "cpu"
    print("⚠️ DirectML 不可用，使用 CPU")

def to_var(x):
    """将张量移到指定设备，并包装为 Variable（PyTorch 新版本中 Variable 已废弃，但保留兼容）"""
    if device != "cpu":
        x = x.to(device)
    return x


def train():
    batch_size = 8          # 可根据显存调整，DML 上建议 4-8
    patience = 5
    best_loss = np.inf
    patience_counter = 0

    train_set = weibo_dataset(is_train=True)
    validate_set = weibo_dataset(is_train=False)

    # num_workers=0 避免多进程问题，DML 下建议设为 0
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=collate_fn,
        shuffle=True,
        pin_memory=False
    )
    test_loader = DataLoader(
        validate_set,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=collate_fn,
        shuffle=False,
        pin_memory=False
    )

    rumor_module = MultiModal()
    # 将模型移到 DML 设备
    rumor_module.to(device)

    loss_f_rumor = torch.nn.CrossEntropyLoss()

    # 提取 BERT 和 Swin 的参数 ID
    base_params = list(map(id, rumor_module.bert.parameters()))
    base_params += list(map(id, rumor_module.swin.parameters()))

    optim_task = torch.optim.Adam([
        {'params': filter(lambda p: p.requires_grad and id(p) not in base_params, rumor_module.parameters())},
        {'params': rumor_module.bert.parameters(), 'lr': 1e-5},
        {'params': rumor_module.swin.parameters(), 'lr': 1e-5}
    ], lr=1e-3)

    for epoch in range(50):
        print("start to train")
        rumor_module.train()
        corrects_pre_rumor = 0
        loss_total = 0
        rumor_count = 0
        tk0 = tqdm(train_loader, desc="train", smoothing=0, mininterval=1.0)

        for i, (input_ids, attention_mask, token_type_ids, image, imageclip, textclip, label) in enumerate(tk0):
            input_ids, attention_mask, token_type_ids, image, imageclip, textclip, label = (
                to_var(input_ids), to_var(attention_mask), to_var(token_type_ids),
                to_var(image), to_var(imageclip), to_var(textclip), to_var(label)
            )

            with torch.no_grad():
                # CLIP 模型已在 DML 设备上
                image_clip = clipmodel.encode_image(imageclip)
                text_clip = clipmodel.encode_text(textclip)

            pre_rumor = rumor_module(input_ids, attention_mask, token_type_ids, image, text_clip, image_clip)
            loss_rumor = loss_f_rumor(pre_rumor, label)

            optim_task.zero_grad()
            loss_rumor.backward()
            optim_task.step()

            pre_label_rumor = pre_rumor.argmax(1)
            corrects_pre_rumor += pre_label_rumor.eq(label.view_as(pre_label_rumor)).sum().item()
            loss_total += loss_rumor.item() * input_ids.shape[0]
            rumor_count += input_ids.shape[0]

        loss_rumor_train = loss_total / rumor_count
        acc_rumor_train = corrects_pre_rumor / rumor_count

        acc_rumor_test, precision_rumor_test, recall_rumor_test, f1_rumor_test, loss_rumor_test, conf_rumor = test(
            rumor_module, test_loader)

        print('-----------rumor detection----------------')
        print(
            "EPOCH = %d || acc_rumor_train = %.3f || acc_rumor_test = %.3f || loss_rumor_train = %.3f || loss_rumor_test = %.3f" %
            (epoch + 1, acc_rumor_train, acc_rumor_test, loss_rumor_train, loss_rumor_test))
        print('-----------rumor_confusion_matrix---------')
        print(conf_rumor)

        if loss_rumor_test < best_loss:
            best_loss = loss_rumor_test
            patience_counter = 0
            torch.save(rumor_module.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    rumor_module.load_state_dict(torch.load('best_model.pth'))
    return rumor_module, test_loader


def test(rumor_module, test_loader):
    rumor_module.eval()
    loss_f_rumor = torch.nn.CrossEntropyLoss()
    rumor_count = 0
    loss_total = 0
    rumor_label_all = []
    rumor_pre_label_all = []

    with torch.no_grad():
        for input_ids, attention_mask, token_type_ids, image, imageclip, textclip, label in test_loader:
            input_ids, attention_mask, token_type_ids, image, imageclip, textclip, label = (
                to_var(input_ids), to_var(attention_mask), to_var(token_type_ids),
                to_var(image), to_var(imageclip), to_var(textclip), to_var(label)
            )
            image_clip = clipmodel.encode_image(imageclip)
            text_clip = clipmodel.encode_text(textclip)

            pre_rumor = rumor_module(input_ids, attention_mask, token_type_ids, image, text_clip, image_clip)
            loss_rumor = loss_f_rumor(pre_rumor, label)
            pre_label_rumor = pre_rumor.argmax(1)

            loss_total += loss_rumor.item() * input_ids.shape[0]
            rumor_count += input_ids.shape[0]

            rumor_pre_label_all.append(pre_label_rumor.detach().cpu().numpy())
            rumor_label_all.append(label.detach().cpu().numpy())

        loss_rumor_test = loss_total / rumor_count
        rumor_pre_label_all = np.concatenate(rumor_pre_label_all, 0)
        rumor_label_all = np.concatenate(rumor_label_all, 0)

        acc_rumor_test = accuracy_score(rumor_label_all, rumor_pre_label_all)
        precision_rumor_test = precision_score(rumor_label_all, rumor_pre_label_all, average=None)
        recall_rumor_test = recall_score(rumor_label_all, rumor_pre_label_all, average=None)
        f1_rumor_test = f1_score(rumor_label_all, rumor_pre_label_all, average=None)
        conf_rumor = confusion_matrix(rumor_label_all, rumor_pre_label_all)

        classification_report_rumor = classification_report(rumor_label_all, rumor_pre_label_all,
                                                            target_names=['realnews', 'fakenews'], digits=4)

    print("Overall Accuracy:", acc_rumor_test)
    print("Precision per class:", precision_rumor_test)
    print("Recall per class:", recall_rumor_test)
    print("F1 Score per class:", f1_rumor_test)
    print("Confusion Matrix:\n", conf_rumor)
    print("Classification Report:\n", classification_report_rumor)

    return acc_rumor_test, precision_rumor_test, recall_rumor_test, f1_rumor_test, loss_rumor_test, conf_rumor


if __name__ == "__main__":
    model, test_loader = train()
    test(model, test_loader)