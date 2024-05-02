from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import random
import os

from Dataset.Normal_Dataset import Bert_Dataset
from Loss.ekd_loss import ekdLoss as ekd_loss
from Loss.PKD_loss import PKDLoss as PKD_loss
from Loss.TINY_loss import TINYLoss as TINY_loss



from evaluate import evaluate


def train(
MAX_SEQ_LEN = 150,
BATCH_SIZE = 32,
LR = 0.00001,
EPOCHS = 40,
NUM_LABELS = 2,

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
train_type = "ekd",
data_set_type = "movie",
):
    torch.cuda.empty_cache()
    
    print(data_set_type)
    tokenizer = BertTokenizer.from_pretrained("./Teacher_Model/"+data_set_type+"_Model/")
    teacher_model = BertForSequenceClassification.from_pretrained(
        "./Teacher_Model/"+data_set_type+"_Model/", num_labels=NUM_LABELS)

    
    student_model = BertConfig(hidden_size=312, num_hidden_layers=8,
                               num_attention_heads=4)
    student_model = BertForSequenceClassification(config=student_model)

    TRAIN_PATH = "./Data/"+data_set_type+"/train_set.txt"
    TEST_PATH = "./Data/"+data_set_type+"/test_set.txt"

    add_loss = None

    if train_type=="ekd":
        add_loss = ekd_loss(student_hidden_size=312, teacher_hidden_size=768, num_layers=REGRESSOER_DEPT, num_attention_heads=REGRESSOER_NUM_HEADS,
                 intermediate_size=3072, max_seq_len=MAX_SEQ_LEN, init_std=0.02)
        
    if train_type=="ekd_first":
        add_loss = ekd_loss(student_hidden_size=312, teacher_hidden_size=768, num_layers=REGRESSOER_DEPT, num_attention_heads=REGRESSOER_NUM_HEADS,
                 intermediate_size=3072, max_seq_len=MAX_SEQ_LEN, init_std=0.02)

    if train_type=="ekd_last":
        add_loss = ekd_loss(student_hidden_size=312, teacher_hidden_size=768, num_layers=REGRESSOER_DEPT, num_attention_heads=REGRESSOER_NUM_HEADS,
                 intermediate_size=3072, max_seq_len=MAX_SEQ_LEN, init_std=0.02)
        
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

            if train_type =="ekd":
                s_hidden_states = [student_output.hidden_states[i] for i in S_KD_LAYERS]
                t_hidden_states = [teacher_output.hidden_states[i] for i in T_KD_LAYERS]
                c_loss += add_loss(s_hidden_states, t_hidden_states,position_ids=pos_embed[:MAX_SEQ_LEN,:])

            elif train_type =="ekd_first":
                s_hidden_states = [student_output.hidden_states[i] for i in S_first_layers]
                t_hidden_states = [teacher_output.hidden_states[i] for i in T_first_layers]
                c_loss += add_loss(s_hidden_states, t_hidden_states, position_ids=pos_embed[:MAX_SEQ_LEN,:])
            elif train_type =="ekd_last":
                s_hidden_states = [student_output.hidden_states[i] for i in S_last_layers]
                t_hidden_states = [teacher_output.hidden_states[i] for i in T_last_layers]
                c_loss += add_loss(s_hidden_states, t_hidden_states, position_ids=pos_embed[:MAX_SEQ_LEN,:])
        
            
            loss = student_output.loss
            
            loss += ALPHA * c_loss + GAMA * kd_loss

            loss.backward()
            optimizer.step()

            total_loss += loss

            progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item())})
        val_accuracy.append(evaluate(student_model, val_dataloader, device))
        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Average training loss: {avg_train_loss:.3f}")

    print("Training complete!")
