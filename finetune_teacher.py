import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel,BertForSequenceClassification
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from tqdm import tqdm


from evaluate import evaluate as evaluate

from Dataset.Normal_Dataset import Bert_Dataset as Bert_Dataset


import numpy as np
import torch
import random
import os


if __name__ == '__main__':

    
    #On which dataset to fine-tune the teacher. 
    #The train_type can be taken as movie, data4, takeaways, shopping or hotel
    data_set_type = "movie"
    
    
    
    seed_value = 2020  
    np.random.seed(seed_value)
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)  

    torch.manual_seed(seed_value)     
    torch.cuda.manual_seed(seed_value)      
    torch.cuda.manual_seed_all(seed_value)  
    torch.backends.cudnn.deterministic = True
    
    max_len_dict = {"hotel":150,"takeaways":70,"shopping":100,"movie":100,"data4":120}
    MAX_SEQ_LEN = max_len_dict[data_set_type]
    
    BATCH_SIZE = 32
    LR = 0.000001  
    EPOCHS = 20
    NUM_LABELS = 2


    TRAIN_PATH = "Data/"+data_set_type+"/train_set.txt"
    TEST_PATH = "Data/"+data_set_type+"/test_set.txt"
    SAVE_PATH = "Teacher_Model/"+data_set_type+"_Model/"

    # Loading pre-trained BERT teacher from hugging-face
    tokenizer = BertTokenizer.from_pretrained("Pretrained_BERT")
    model = BertForSequenceClassification.from_pretrained(
        "Pretrained_BERT", num_labels=NUM_LABELS)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model =model.to(device)


    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_function = torch.nn.CrossEntropyLoss()

    train_dataset = Bert_Dataset(TRAIN_PATH
        , tokenizer, MAX_SEQ_LEN)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    val_dataset = Bert_Dataset(TEST_PATH, tokenizer, MAX_SEQ_LEN)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)


    epochs = EPOCHS
    print("training start!!!")
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        progress_bar = tqdm(train_dataloader,
                              desc=f"Epoch {epoch + 1}/{epochs}", position=0, leave=True)

        for batch in progress_bar:
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            output = model(input_ids, attention_mask=attention_mask, labels=labels)

            loss = output.loss
            loss.backward()
            optimizer.step()

            total_loss += loss

            progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item())})

        evaluate(model,val_dataloader,device)
        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Average training loss: {avg_train_loss:.3f}")


    model.save_pretrained(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)


    print("Training complete!")
