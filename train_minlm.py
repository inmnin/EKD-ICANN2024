from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig, BertModel
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

import pickle

from Loss.MINLM_loss import MINLMLoss as MINLM_loss
from Dataset.Normal_Dataset import Bert_Dataset
from evaluate import evaluate


def get_result_index(file_path):
    file_path += "index.txt"
    index = None
    with open(file_path, 'r') as f:
        index = f.readlines()[0].strip()
    return index

def update_result_index(file_path):
    file_path += "index.txt"
    index = None
    with open(file_path, 'r') as f:
        index = f.readlines()[0].strip()
    index = int(index)
    index += 1
    index = str(index)
    with open(file_path, 'w') as f:
        f.write(index)


def train_minlm(description="minilm",
          MAX_SEQ_LEN=150,
          BATCH_SIZE=32,
          LR=0.00001,
          EPOCHS=40,
          NUM_LABELS=2,
          # 蒸馏损失相关参数
          ALPHA_VITKD=0.00001,
          BETA_VITKD=0,
          GAMA_VITKD=0.3,
          MASK_RATE=0.3,
          REGRESSOER_DEPT=3,
          REGRESSOER_NUM_HEADS=8,
          T_KD_LAYERS=[0, 3, 6, 9, 12],
          S_KD_LAYERS=[0, 2, 4, 6, 8],
          S_first_layers=[0, 1, 2, 3, 4],
          T_first_layers=[0, 1, 2, 3, 4],
          S_last_layers=[4, 5, 6, 7, 8],
          T_last_layers=[8, 9, 10, 11, 12],
          train_type="cae",
          data_set_type="hotel",
          ):
    torch.cuda.empty_cache()
    # 老师模型
    print(data_set_type)
    tokenizer = BertTokenizer.from_pretrained("./Teacher_Model/" + data_set_type + "_Model/")
    teacher_model = BertForSequenceClassification.from_pretrained(
        "./Teacher_Model/" + data_set_type + "_Model/", num_labels=NUM_LABELS)

    # 学生模型
    student_model = BertConfig(hidden_size=312, num_hidden_layers=8,
                               num_attention_heads=4)
    student_model = BertForSequenceClassification(config=student_model)

    TRAIN_PATH = "./Data/" + data_set_type + "/train_set.txt"
    TEST_PATH = "./Data/" + data_set_type + "/test_set.txt"

    add_loss = None
    RESULT_PATH = ""

    if train_type == "minlm":
        add_loss = MINLM_loss(teacher_dims=teacher_model.config.hidden_size,
                            student_dims=student_model.config.hidden_size,
                            student_mim_layers=S_KD_LAYERS,
                            teacher_mim_layers=T_KD_LAYERS,
                            student = student_model,
                            teacher = teacher_model,
                            L = 12,
                            M = 8,
                            A_r=12,
                            alpha = ALPHA_VITKD,
                            beta = BETA_VITKD,
                            gama = GAMA_VITKD
                            )
        RESULT_PATH = "result/" + data_set_type + '/MINLM/'

    # 定义设备，模型加载到设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student_model = student_model.to(device)
    teacher_model = teacher_model.to(device)
    add_loss = add_loss.to(device)

    # 定义优化器和损失函数
    combined_params = list(add_loss.student.parameters())
    optimizer = torch.optim.AdamW(combined_params, lr=LR)
    loss_function = torch.nn.CrossEntropyLoss()

    # 训练集
    train_dataset = Bert_Dataset(
        TRAIN_PATH, tokenizer, MAX_SEQ_LEN)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 测试集
    val_dataset = Bert_Dataset(TEST_PATH, tokenizer, MAX_SEQ_LEN)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 开始训练循环
    epochs = EPOCHS
    # 统计预测集上的准确率
    val_accuracy = []

    print("training start!!!")
    for epoch in range(epochs):
        add_loss.student.train()
        add_loss.teacher.eval()

        total_loss = 0

        progress_bar = tqdm(train_dataloader,
                            desc=f"Epoch {epoch + 1}/{epochs}", position=0, leave=True)

        for batch in progress_bar:
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            loss = add_loss(input_ids=input_ids, attention_mask=attention_mask,labels=labels)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            total_loss += loss

            # 只显示学生的bar即可
            progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item())})
        val_accuracy.append(evaluate(add_loss.student, val_dataloader, device))
        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Average training loss: {avg_train_loss:.3f}")

    hyper_dic = {"DESCRIPTION": description,
                 "random seed": 2020,
                 "MAX_SEQ_LEN": MAX_SEQ_LEN,
                 "BATCH_SIZE": BATCH_SIZE,
                 "LR": LR,
                 "epochs": EPOCHS,
                 "NUM_LABELS": NUM_LABELS,
                 "ALPHA_VITKD": ALPHA_VITKD,
                 "BETA_VITKD": BETA_VITKD,
                 "GAMA_VITKD": GAMA_VITKD,
                 "MASK_RATE": MASK_RATE,
                 "REGRESSOER_DEPT": REGRESSOER_DEPT,
                 "REGRESSOER_NUM_HEADS": REGRESSOER_NUM_HEADS,
                 "S_KD_LAYERS": S_KD_LAYERS,
                 "T_KD_LAYERS": T_KD_LAYERS,
                 }
    val_accuracy.append(hyper_dic)
    # 保存训练后模型结果
    path = RESULT_PATH
    result_index = get_result_index(path)

    with open(path + result_index + '.pkl', 'wb') as f:
        pickle.dump(val_accuracy, f)

    update_result_index(path)
    print("Training complete!")
