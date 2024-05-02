import torch
import numpy as np
import random
import os
from train import train
from train_minlm import train_minlm

if __name__ == '__main__':
    """
    """
    
    #The datasets are of the following types:    movie, data4, takeaways, shopping or hotel
    data_set_type = "movie"
    
    #ALPHA is the loss weight corresponding to the ekd loss in the ekd method
    ALPHA = 5



    #The train_type can be one of the following:    ekd, ekd_first, ekd_last, pkd_skip, kd, baseline, tiny, minlmv1, or minlmv2
    train_type = "ekd"

    EPOCHS = 15
    BETA = 0
    GAMA = 0
    
    MASK_RATE = 0.5
    
    #Parameters of the autoencoder
    REGRESSOER_DEPT = 3
    REGRESSOER_NUM_HEADS = 8
    """
    """

    BATCH_SIZE = 32
    LR = 0.00001
    NUM_LABELS = 2

    T_KD_LAYERS = [0, 3, 6, 9, 12]
    S_KD_LAYERS = [0, 2, 4, 6, 8]

    max_len_dict = {"hotel":150,"takeaways":70,"shopping":150,"movie":100,"data4":120}
    MAX_SEQ_LEN = max_len_dict[data_set_type]
    


    #Fixed random seed
    seed_value = 2020  
    np.random.seed(seed_value)
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)  
    torch.manual_seed(seed_value) 
    torch.cuda.manual_seed(seed_value) 
    torch.cuda.manual_seed_all(seed_value) 
    torch.backends.cudnn.deterministic = True

    #train
    if train_type == "minlmv1" or train_type == "minlmv2":
        train_minlm(
            MAX_SEQ_LEN = MAX_SEQ_LEN,
            BATCH_SIZE = BATCH_SIZE,
            LR = LR,
            EPOCHS = EPOCHS,
            NUM_LABELS = NUM_LABELS,
            ALPHA = ALPHA,
            BETA = BETA,
            GAMA = GAMA,
            MASK_RATE = MASK_RATE,
            REGRESSOER_DEPT = REGRESSOER_DEPT,
            REGRESSOER_NUM_HEADS = REGRESSOER_NUM_HEADS,
            T_KD_LAYERS = T_KD_LAYERS,
            S_KD_LAYERS = S_KD_LAYERS,
            train_type = train_type,
            data_set_type = data_set_type)
    else:  
        train(
            MAX_SEQ_LEN = MAX_SEQ_LEN,
            BATCH_SIZE = BATCH_SIZE,
            LR = LR,
            EPOCHS = EPOCHS,
            NUM_LABELS = NUM_LABELS,
            ALPHA = ALPHA,
            BETA = BETA,
            GAMA = GAMA,
            MASK_RATE = MASK_RATE,
            REGRESSOER_DEPT = REGRESSOER_DEPT,
            REGRESSOER_NUM_HEADS = REGRESSOER_NUM_HEADS,
            T_KD_LAYERS = T_KD_LAYERS,
            S_KD_LAYERS = S_KD_LAYERS,
            train_type = train_type,
            data_set_type = data_set_type)
