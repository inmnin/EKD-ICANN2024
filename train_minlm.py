from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig, BertModel
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F


from Loss.MINLM_loss import MINLMLoss as MINLM_loss
from Dataset.Normal_Dataset import Bert_Dataset
from evaluate import evaluate



def train_minlm(
          MAX_SEQ_LEN=150,
          BATCH_SIZE=32,
          LR=0.00001,
          EPOCHS=40,
          NUM_LABELS=2,
          ALPHA=0.00001,
          BETA=0,
          GAMA=0.3,
          MASK_RATE=0.3,
          REGRESSOER_DEPT=3,
          REGRESSOER_NUM_HEADS=8,
          T_KD_LAYERS=[0, 3, 6, 9, 12],
          S_KD_LAYERS=[0, 2, 4, 6, 8],
          S_first_layers=[0, 1, 2, 3, 4],
          T_first_layers=[0, 1, 2, 3, 4],
          S_last_layers=[4, 5, 6, 7, 8],
          T_last_layers=[8, 9, 10, 11, 12],
          train_type="minlmv1",
          data_set_type="movie",
          ):
    torch.cuda.empty_cache()

    print(data_set_type)
    tokenizer = BertTokenizer.from_pretrained("./Teacher_Model/" + data_set_type + "_Model/")
    teacher_model = BertForSequenceClassification.from_pretrained(
        "./Teacher_Model/" + data_set_type + "_Model/", num_labels=NUM_LABELS)

    student_model = BertConfig(hidden_size=312, num_hidden_layers=8,
                               num_attention_heads=4)
    student_model = BertForSequenceClassification(config=student_model)

    TRAIN_PATH = "./Data/" + data_set_type + "/train_set.txt"
    TEST_PATH = "./Data/" + data_set_type + "/test_set.txt"

    add_loss = None

    add_loss = MINLM_loss(teacher_dims=teacher_model.config.hidden_size,
                            student_dims=student_model.config.hidden_size,
                            student_mim_layers=S_KD_LAYERS,
                            teacher_mim_layers=T_KD_LAYERS,
                            student=student_model,
                            teacher=teacher_model,
                            L=12,
                            M=8,
                            A_r=12,
                            alpha=ALPHA,
                            beta=BETA,
                            gama=GAMA,
                            train_type=train_type
                            )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student_model = student_model.to(device)
    teacher_model = teacher_model.to(device)
    add_loss = add_loss.to(device)

    combined_params = list(add_loss.student.parameters())
    optimizer = torch.optim.AdamW(combined_params, lr=LR)
    loss_function = torch.nn.CrossEntropyLoss()


    train_dataset = Bert_Dataset(
        TRAIN_PATH, tokenizer, MAX_SEQ_LEN)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)


    val_dataset = Bert_Dataset(TEST_PATH, tokenizer, MAX_SEQ_LEN)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)


    epochs = EPOCHS

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

            loss.backward()
            optimizer.step()

            total_loss += loss

            progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item())})
        val_accuracy.append(evaluate(add_loss.student, val_dataloader, device))
        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Average training loss: {avg_train_loss:.3f}")

    print("Training complete!")
