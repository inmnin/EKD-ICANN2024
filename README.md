# Introduction
We provide a pytorch implementation of the MAKD method.
## for the data:
All the datasets we have cleaned and segmented are uploaded in this project and users can find them in folder `Data`.
## for the teacher model weights:
The teacher model's weights fine-tuned for a particular dataset corresponds to the folder `Teacher_Model/xxx_Model/`(xxx is the name of dataset).  

# MAKD in 3 steps
## 1.Configuring the Experimental Environment:
We saved the environment configurations used in our experiments in the `requirements.txt`.

In the experiment, we used a single RTX 4090 (24GB) GPU.

Please refer to the above information to configure the environment for the experiment.  
  
## 2. Acquire the fine-tuned teacher model weights:
We provide the following two ways to acquire the teacher model weights fine-tuned on each dataset.

In order to better reproduce the experimental results, **we recommend that you choose the first way to directly use the fine-tuned teacher model weights provided by us**.

### Use the fine-tuned teacher model weights we provide (recommended)
If you wish to use the fine-tuned teacher model weights we provide, please download **https://ufile.io/a6d9dfxs**, and then replace the original folder `Teacher_Model` in the root directory with the unzipped folder `Teacher_Model`. 

**After the replacement is completed, you can move on to the next step: *"3. run MAKD method in main.py"***.


### Fine-tune the teacher model by yourself (unrecommended)
If you wish to fine-tune the teacher model by yourself, firstly, please download the base teacher model weights from **https://ufile.io/iv2e19xd**. 

After downloading, please replace the original folder `Pretrained_BERT` in the root directory with the unzipped folder `Pretrained_BERT`. Then you need to set `train_set_type` in `finetune_teacher.py` to which dataset you want to fine-tune (movie, takeaways, data4, shopping or hotel). 
```
    #The train_type can be taken as movie, data4, takeaways, shopping or hotel
    data_set_type = "movie"
```

After setting the parameters, you can run `finetune_teacher.py`. 




To ensure you obtain the fine-tuned teacher model weights for all datasets, you need to run `finetune_teacher.py` multiple times, each time setting a different `data_set_type`, covering all types(movie, takeaways, data4, shopping and hotel).

The teacher's weights fine-tuned for a particular dataset will be saved in the folder `Teacher_Model/xxx_Model/`(xxx is the `data_set_type` you have set). 


## 3. Run MAKD method in main.py:
In `main.py`, please set `data_set_type` to the dataset name (movie, takeaways, data4, shopping or hotel). 

Then set `ALPHA` according to the `data_set_type` based on the table below: 
|data_set_type:|movie|takeaways|data4|shopping|hotel|
|:---|:---|:---|:---|:---|:---|
|**ALPHA:**|5|3000|1|2.5|0.00003|
```
    #The datasets are of the following types:    movie, data4, takeaways, shopping or hotel
    data_set_type = "movie"
    
    #ALPHA is the loss weight corresponding to the MAKD loss in the MAKD method
    ALPHA = 5

```

After setting these parameters, you can just run `main.py`. 

Users can refer to the average accuracy of the predictions on the test set in the last 5 epoch. 

