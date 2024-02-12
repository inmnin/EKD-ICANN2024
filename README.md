# introduction
We provide a pytorch implementation of the MAKD method.

# For the data we used:
All the datasets we have cleaned and segmented are uploaded in this project and users can find them in folder **Data**.
# For the teacher model weights we used:
The base teacher model weights in this experiment are from *https://huggingface.co/bert-base-chinese/tree/main*. We fine-tuned this base model on different datasets independently.
The teacher's weights fine-tuned for a particular dataset will be saved in the corresponding folder **Teacher_Model/xxx_Model/**(xxx is the name of dataset).


# MAKD in 3 steps
## 1.Configuring the Experimental Environment
We saved the environment configurations used in our experiments in the requirements.txt.

We record the configuration of the devices we used in our experiment as follows for users' reference: RTX 4090(24GB) * 1,12 vCPU Intel(R) Xeon(R) Platinum 8352V CPU @ 2.10GHz.

We conduct experiments in a single GPU environment.

Please refer to the above information to configure the environment for the experiment.

## 2. Fine-tune the teacher model on different datasets:
We provide the following two ways to fine-tune the teacher model.

### Use the fine-tuned teacher model weights we provide (recommended)
If you wish to use the fine-tuned teacher model weights we provide, please download **https://ufile.io/a6d9dfxs**, and then replace the original folder **Teacher_Model**  with the unzipped folder **Teacher_Model**. 

Using the model weights we provide will help you replicate our experimental results, so we recommend using the weights we provide directly.


### Fine-tune the teacher model by yourself (unrecommended)
If you wish to fine-tune the teacher model by yourself, firstly, please download the base parameters used for the teacher model from **https://ufile.io/iv2e19xd**. (It is the same as *https://huggingface.co/bert-base-chinese/tree/main*. However, the original repository lacks maintenance, which might causes some problems when running. So we have made a copy of the model's parameters available for download) 

After downloading, please replace the original folder Pretrained_BERT with the unzipped folder Pretrained_BERT. Then you need to set train_set_type in finetune_teacher.py to which dataset you want to fine-tune. 
```
    #The train_type can be taken as movie, data4, takeaways, shopping or hotel
    data_set_type = "movie"
```

After setting the parameters, you can run finetune_teacher.py to get the teacher model fine-tuned on this dataset. 

The teacher's weights fine-tuned for a particular dataset will be saved in the folder Teacher_Model/xxx_Model/(xxx is the data_set_type you have set).


## 3. run MAKD method in main.py:
In experimenting with the MAKD method, please set **data_set_type** to the dataset name (movie,takeaways,data4,shopping or hotel) in main.py. 

Then set **ALPHA** according to the data_set_type based on the table below: 
|data_set_type:|movie|takeaways|data4|shopping|hotel|
|:---|:---|:---|:---|:---|:---|
|**ALPHA:**|5|3000|1|2.5|0.00003|
```
    #The datasets are of the following types:    movie, data4, takeaways, shopping or hotel
    data_set_type = "movie"
    
    #ALPHA is the loss weight corresponding to the MAKD loss in the MAKD method
    ALPHA = 5

```
After setting these parameters, you can just run main.py. 

Users can refer to the average accuracy of the predictions on the test set in the last 5 epoch. 

