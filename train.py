from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import random
import os
import pickle
from Dataset.Normal_Dataset import Bert_Dataset
from Loss.MAE_loss import MAEFeatureGenerator as MAE_loss
from Loss.PKD_loss import PKDLoss as PKD_loss
from Loss.TINY_loss import TINYLoss as TINY_loss
from Loss.CAE_loss import ContextAutoencoderLoss as CAE_loss


from evaluate import evaluate
def get_result_index(file_path):
    file_path += "index.txt"
    index = None
    with open(file_path,'r') as f:
        index = f.readlines()[0].strip()
    return index
def update_result_index(file_path):
    file_path += "index.txt"
    index = None
    with open(file_path,'r') as f:
        index = f.readlines()[0].strip()
    index = int(index)
    index += 1
    index = str(index)
    with open(file_path,'w') as f:
        f.write(index)

def train(description = "skip,共用一个CAE模块,第二次hotel调参",
MAX_SEQ_LEN = 150,
BATCH_SIZE = 32,
LR = 0.00001,
EPOCHS = 40,
NUM_LABELS = 2,
#蒸馏损失相关参数
ALPHA_VITKD = 0.00001,
BETA_VITKD = 0,
GAMA_VITKD = 0.3,
MASK_RATE = 0.3,
REGRESSOER_DEPT = 3,
REGRESSOER_NUM_HEADS = 8,
T_KD_LAYERS = [0,3,6,9,12],
S_KD_LAYERS = [0,2,4,6,8],
S_first_layers = [0, 1, 2, 3, 4],
T_first_layers = [0, 1, 2, 3, 4],
S_last_layers = [4, 5, 6, 7, 8],
T_last_layers = [8,9,10,11,12],
train_type = "cae",
data_set_type = "hotel",
):
    torch.cuda.empty_cache()
    # 老师模型
    print(data_set_type)
    tokenizer = BertTokenizer.from_pretrained("./Teacher_Model/"+data_set_type+"_Model/")
    teacher_model = BertForSequenceClassification.from_pretrained(
        "./Teacher_Model/"+data_set_type+"2_Model/", num_labels=NUM_LABELS)

    # 学生模型
    student_model = BertConfig(hidden_size=312, num_hidden_layers=8,
                               num_attention_heads=4)
    student_model = BertForSequenceClassification(config=student_model)

    TRAIN_PATH = "./Data/"+data_set_type+"/train_set.txt"
    TEST_PATH = "./Data/"+data_set_type+"/test_set.txt"

    add_loss = None
    RESULT_PATH = ""

    # if train_type=="cae":
    #     # 定义损失函数
    #     add_loss = CAE_loss(teacher_dims=teacher_model.config.hidden_size, student_dims=student_model.config.hidden_size,
    #                         regressor_depth=REGRESSOER_DEPT, regressor_num_heads=REGRESSOER_NUM_HEADS, mask_rate=MASK_RATE)
    #     RESULT_PATH = 'result/'+data_set_type+'/skip_CAE/'
    if train_type=="mae":
        add_loss = MAE_loss(student_hidden_size=312, teacher_hidden_size=768, num_layers=REGRESSOER_DEPT, num_attention_heads=REGRESSOER_NUM_HEADS,
                 intermediate_size=3072, max_seq_len=MAX_SEQ_LEN, init_std=0.02)
        RESULT_PATH = 'result/'+data_set_type+'/skip_MAE/'
    
    if train_type=="cae":
        add_loss = CAE_loss(student_dims=312, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=None, init_values=None, attn_head_dim=None, init_std=0.02,
                 teacher_dims=768, regressor_depth=REGRESSOER_DEPT, regressor_num_heads=REGRESSOER_NUM_HEADS,
                 regressor_layer_scale_init_value=0.1, fix_init_weight=False, mask_rate = MASK_RATE)
        RESULT_PATH = 'result/'+data_set_type+'/skip_CAE/'
        
    if train_type=="mae_first":
        add_loss = MAE_loss(student_hidden_size=312, teacher_hidden_size=768, num_layers=REGRESSOER_DEPT, num_attention_heads=REGRESSOER_NUM_HEADS,
                 intermediate_size=3072, max_seq_len=MAX_SEQ_LEN, init_std=0.02)
        RESULT_PATH = 'result/'+data_set_type+'/first_MAE/'

    if train_type=="mae_last":
        add_loss = MAE_loss(student_hidden_size=312, teacher_hidden_size=768, num_layers=REGRESSOER_DEPT, num_attention_heads=REGRESSOER_NUM_HEADS,
                 intermediate_size=3072, max_seq_len=MAX_SEQ_LEN, init_std=0.02)
        RESULT_PATH = 'result/'+data_set_type+'/last_MAE/'

    if train_type=="pkd_skip":
        add_loss = PKD_loss(teacher_dims=teacher_model.config.hidden_size, student_dims=student_model.config.hidden_size,
                            student_mim_layers=S_KD_LAYERS,teacher_mim_layers=T_KD_LAYERS)
        RESULT_PATH = 'result/'+data_set_type+'/skip_PKD/'

    if train_type =="kd":
        add_loss = PKD_loss(teacher_dims=teacher_model.config.hidden_size, student_dims=student_model.config.hidden_size,
                            student_mim_layers=S_KD_LAYERS,teacher_mim_layers=T_KD_LAYERS)
        RESULT_PATH = "result/"+data_set_type+'/KD/'
    if train_type=="baseline":
        add_loss = PKD_loss(teacher_dims=teacher_model.config.hidden_size, student_dims=student_model.config.hidden_size,
                            student_mim_layers=S_KD_LAYERS,teacher_mim_layers=T_KD_LAYERS)
        RESULT_PATH = "result/"+data_set_type+'/baseline/'

    if train_type=="tiny":
        add_loss = TINY_loss(teacher_dims=teacher_model.config.hidden_size, student_dims=student_model.config.hidden_size,
                             student_mim_layers=S_KD_LAYERS, teacher_mim_layers=T_KD_LAYERS)
        RESULT_PATH = "result/"+data_set_type+'/TINY/'
        # 确保模型输出注意力矩阵
        student_model.config.output_attentions = True
        teacher_model.config.output_attentions = True

    # 定义设备，模型加载到设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student_model = student_model.to(device)
    teacher_model = teacher_model.to(device)
    add_loss = add_loss.to(device)

    # 定义优化器和损失函数
    combined_params = list(student_model.parameters()) + list(add_loss.parameters())
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
    CE = nn.CrossEntropyLoss()
    KL = nn.KLDivLoss(reduction='batchmean')  # 实例化KL散度损失
    print("training start!!!")
    for epoch in range(epochs):
        student_model.train()
        add_loss.train()
        teacher_model.eval()

        total_loss = 0

        progress_bar = tqdm(train_dataloader,
                            desc=f"Epoch {epoch + 1}/{epochs}", position=0, leave=True)

        for batch in progress_bar:
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            pos_embed = student_model.bert.embeddings.position_embeddings.weight
            # 模型的输出(输出所有隐藏层状态)
            student_output = student_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                                           output_hidden_states=True)
            teacher_output = teacher_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                                           output_hidden_states=True)

            c_loss = 0
            kd_loss = 0
            # 获得模型的隐藏状态
            if train_type == "cae":
                for i, j in zip(S_KD_LAYERS, T_KD_LAYERS):
                    hidden_fea_s = student_output.hidden_states[i]
                    hidden_fea_t = teacher_output.hidden_states[j]

                    # 学生和老师的深层特征
                    preds_S = hidden_fea_s
                    preds_T = hidden_fea_t
                    c_loss += add_loss(preds_S, preds_T, pos_embed=pos_embed)

            elif train_type =="mae":
                s_hidden_states = [student_output.hidden_states[i] for i in S_KD_LAYERS]
                t_hidden_states = [teacher_output.hidden_states[i] for i in T_KD_LAYERS]
                c_loss += add_loss(s_hidden_states, t_hidden_states,position_ids=pos_embed[:MAX_SEQ_LEN,:])

            elif train_type =="mae_first":
                s_hidden_states = [student_output.hidden_states[i] for i in S_first_layers]
                t_hidden_states = [teacher_output.hidden_states[i] for i in T_first_layers]
                c_loss += add_loss(s_hidden_states, t_hidden_states, position_ids=pos_embed[:MAX_SEQ_LEN,:])
            elif train_type =="mae_last":
                s_hidden_states = [student_output.hidden_states[i] for i in S_last_layers]
                t_hidden_states = [teacher_output.hidden_states[i] for i in T_last_layers]
                c_loss += add_loss(s_hidden_states, t_hidden_states, position_ids=pos_embed[:MAX_SEQ_LEN,:])
            
            elif train_type == "pkd_skip":
                s_logits = student_output.logits
                t_logits = teacher_output.logits
                
                t_logits_soft = F.softmax(t_logits / BETA_VITKD, dim=1)
                s_logits_soft = F.log_softmax(s_logits / BETA_VITKD, dim=1)
                c_loss = KL(s_logits_soft, t_logits_soft)*(BETA_VITKD*BETA_VITKD)
                
                kd_loss += add_loss(student_output.hidden_states,teacher_output.hidden_states)

            elif train_type =="kd":
                s_logits = student_output.logits
                t_logits = teacher_output.logits
                
                # 蒸馏损失
                t_logits_soft = F.softmax(t_logits / BETA_VITKD, dim=1)
                s_logits_soft = F.log_softmax(s_logits / BETA_VITKD, dim=1)
                c_loss = KL(s_logits_soft, t_logits_soft)*(BETA_VITKD*BETA_VITKD)
                
            elif train_type=="tiny":
                kd_loss = add_loss(student_output.hidden_states,
                                    teacher_output.hidden_states,
                                    student_output.attentions,
                                    teacher_output.attentions
                                    )
                s_logits = student_output.logits
                t_logits = teacher_output.logits

                t_logits_soft = F.softmax(t_logits / BETA_VITKD, dim=1)
                s_logits_soft = F.log_softmax(s_logits / BETA_VITKD, dim=1)
                c_loss = KL(s_logits_soft, t_logits_soft) * (BETA_VITKD * BETA_VITKD)
            else:
                c_loss = 0
            
            loss = student_output.loss
            
            if train_type == "kd" or train_type == "pkd_skip" or train_type=="tiny":
                loss = loss*(1 - ALPHA_VITKD)
            
            # 这里要改改
            loss += ALPHA_VITKD * c_loss + GAMA_VITKD*kd_loss

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            total_loss += loss

            # 只显示学生的bar即可
            progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item())})
        val_accuracy.append(evaluate(student_model, val_dataloader, device))
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

    with open(path+result_index+'.pkl', 'wb') as f:
        pickle.dump(val_accuracy, f)

    update_result_index(path)
    print("Training complete!")
