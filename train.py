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
from Loss.MAKD_loss import MAKDLoss as MAKD_loss
from Loss.PKD_loss import PKDLoss as PKD_loss
from Loss.TINY_loss import TINYLoss as TINY_loss



from evaluate import evaluate


def train(
MAX_SEQ_LEN = 150,
BATCH_SIZE = 32,
LR = 0.00001,
EPOCHS = 40,
NUM_LABELS = 2,
#蒸馏损失相关参数
ALPHA = 0.00001,
BETA = 0,
GAMA = 0.3,
MASK_RATE = 0.3,
REGRESSOER_DEPT = 3,
REGRESSOER_NUM_HEADS = 8,
T_KD_LAYERS = [0,3,6,9,12],
S_KD_LAYERS = [0,2,4,6,8],
S_first_layers = [0, 1, 2, 3, 4],
T_first_layers = [0, 1, 2, 3, 4],
S_last_layers = [4, 5, 6, 7, 8],
T_last_layers = [8,9,10,11,12],
train_type = "makd",
data_set_type = "movie",
):
    torch.cuda.empty_cache()
    # 老师模型
    print(data_set_type)
    tokenizer = BertTokenizer.from_pretrained("./Teacher_Model/"+data_set_type+"_Model/")
    teacher_model = BertForSequenceClassification.from_pretrained(
        "./Teacher_Model/"+data_set_type+"_Model/", num_labels=NUM_LABELS)

    # 学生模型
    student_model = BertConfig(hidden_size=312, num_hidden_layers=8,
                               num_attention_heads=4)
    student_model = BertForSequenceClassification(config=student_model)

    TRAIN_PATH = "./Data/"+data_set_type+"/train_set.txt"
    TEST_PATH = "./Data/"+data_set_type+"/test_set.txt"

    add_loss = None

    if train_type=="makd":
        add_loss = MAKD_loss(student_hidden_size=312, teacher_hidden_size=768, num_layers=REGRESSOER_DEPT, num_attention_heads=REGRESSOER_NUM_HEADS,
                 intermediate_size=3072, max_seq_len=MAX_SEQ_LEN, init_std=0.02)
        
    if train_type=="makd_first":
        add_loss = MAKD_loss(student_hidden_size=312, teacher_hidden_size=768, num_layers=REGRESSOER_DEPT, num_attention_heads=REGRESSOER_NUM_HEADS,
                 intermediate_size=3072, max_seq_len=MAX_SEQ_LEN, init_std=0.02)

    if train_type=="makd_last":
        add_loss = MAKD_loss(student_hidden_size=312, teacher_hidden_size=768, num_layers=REGRESSOER_DEPT, num_attention_heads=REGRESSOER_NUM_HEADS,
                 intermediate_size=3072, max_seq_len=MAX_SEQ_LEN, init_std=0.02)

    if train_type=="pkd_skip":
        add_loss = PKD_loss(teacher_dims=teacher_model.config.hidden_size, student_dims=student_model.config.hidden_size,
                            student_mim_layers=S_KD_LAYERS,teacher_mim_layers=T_KD_LAYERS)

    if train_type =="kd":
        add_loss = PKD_loss(teacher_dims=teacher_model.config.hidden_size, student_dims=student_model.config.hidden_size,
                            student_mim_layers=S_KD_LAYERS,teacher_mim_layers=T_KD_LAYERS)

    if train_type=="baseline":
        add_loss = PKD_loss(teacher_dims=teacher_model.config.hidden_size, student_dims=student_model.config.hidden_size,
                            student_mim_layers=S_KD_LAYERS,teacher_mim_layers=T_KD_LAYERS)

    if train_type=="tiny":
        add_loss = TINY_loss(teacher_dims=teacher_model.config.hidden_size, student_dims=student_model.config.hidden_size,
                             student_mim_layers=S_KD_LAYERS, teacher_mim_layers=T_KD_LAYERS)

        # Ensure that the model outputs an attention matrix
        student_model.config.output_attentions = True
        teacher_model.config.output_attentions = True

        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student_model = student_model.to(device)
    teacher_model = teacher_model.to(device)
    add_loss = add_loss.to(device)

    combined_params = list(student_model.parameters()) + list(add_loss.parameters())
    optimizer = torch.optim.AdamW(combined_params, lr=LR)
    loss_function = torch.nn.CrossEntropyLoss()

    train_dataset = Bert_Dataset(
        TRAIN_PATH, tokenizer, MAX_SEQ_LEN)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    val_dataset = Bert_Dataset(TEST_PATH, tokenizer, MAX_SEQ_LEN)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Start training cycle
    epochs = EPOCHS
    # Accuracy on the test set
    val_accuracy = []
    CE = nn.CrossEntropyLoss()
    KL = nn.KLDivLoss(reduction='batchmean')
    
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
            # Output of the model (output all hidden layer states)
            student_output = student_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                                           output_hidden_states=True)
            teacher_output = teacher_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                                           output_hidden_states=True)

            c_loss = 0
            kd_loss = 0

            if train_type =="makd":
                s_hidden_states = [student_output.hidden_states[i] for i in S_KD_LAYERS]
                t_hidden_states = [teacher_output.hidden_states[i] for i in T_KD_LAYERS]
                c_loss += add_loss(s_hidden_states, t_hidden_states,position_ids=pos_embed[:MAX_SEQ_LEN,:])

            elif train_type =="makd_first":
                s_hidden_states = [student_output.hidden_states[i] for i in S_first_layers]
                t_hidden_states = [teacher_output.hidden_states[i] for i in T_first_layers]
                c_loss += add_loss(s_hidden_states, t_hidden_states, position_ids=pos_embed[:MAX_SEQ_LEN,:])
            elif train_type =="makd_last":
                s_hidden_states = [student_output.hidden_states[i] for i in S_last_layers]
                t_hidden_states = [teacher_output.hidden_states[i] for i in T_last_layers]
                c_loss += add_loss(s_hidden_states, t_hidden_states, position_ids=pos_embed[:MAX_SEQ_LEN,:])
            
            elif train_type == "pkd_skip":
                s_logits = student_output.logits
                t_logits = teacher_output.logits
                
                t_logits_soft = F.softmax(t_logits / BETA, dim=1)
                s_logits_soft = F.log_softmax(s_logits / BETA, dim=1)
                c_loss = KL(s_logits_soft, t_logits_soft)*(BETA*BETA)
                
                kd_loss += add_loss(student_output.hidden_states,teacher_output.hidden_states)

            elif train_type =="kd":
                s_logits = student_output.logits
                t_logits = teacher_output.logits
                
                t_logits_soft = F.softmax(t_logits / BETA, dim=1)
                s_logits_soft = F.log_softmax(s_logits / BETA, dim=1)
                c_loss = KL(s_logits_soft, t_logits_soft)*(BETA*BETA)
                
            elif train_type=="tiny":
                kd_loss = add_loss(student_output.hidden_states,
                                    teacher_output.hidden_states,
                                    student_output.attentions,
                                    teacher_output.attentions
                                    )
                s_logits = student_output.logits
                t_logits = teacher_output.logits

                t_logits_soft = F.softmax(t_logits / BETA, dim=1)
                s_logits_soft = F.log_softmax(s_logits / BETA, dim=1)
                c_loss = KL(s_logits_soft, t_logits_soft) * (BETA * BETA)
            else:
                c_loss = 0
            
            loss = student_output.loss
            
            if train_type == "kd" or train_type == "pkd_skip" or train_type=="tiny":
                loss = loss*(1 - ALPHA)
            
            loss += ALPHA * c_loss + GAMA * kd_loss

            loss.backward()
            optimizer.step()

            total_loss += loss

            progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item())})
        val_accuracy.append(evaluate(student_model, val_dataloader, device))
        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Average training loss: {avg_train_loss:.3f}")

    print("Training complete!")
