# MAKD
We provide a pytorch version implementation of the MAKD method.

# For the fine-tune of teacher the model on different datasets:
The fine-tuned teacher model weights for different datasets are expected to be stored in different folders of Teacher_Model. Users can either fine-tune them themselves on different datasets before testing them, or using our submitted teacher model weights:https://userscloud.com/e5g1rffrq1pp. 
If you wish to use the weights we provide, please replace the original Teacher_Model folder with the unzipped Teacher_Model folder.  
If you wish to fine-tune the teacher model by yourself, the base parameters used for the teacher model are from https://huggingface.co/bert-base-chinese, You can put the model weights in the Pretrained_BERT folder after downloading them on the hugging-face platform. Then you need to set train_set_type in finetune_teacher.py to which dataset you want to fine-tune. Once this is done you can run it and the teacher model fine-tuned for a particular dataset will be used for later experiments on that dataset!

After doing the above, you can start running the MAKD method.

# For run MAKD method in main.py:
In experimenting with the MAKD method, please change train_type to "makd"(or other methods of comparison), and data_set_type to the corresponding dataset name (movie,takeaways,data4,shopping or hotel) in main.py. Then set 15 rounds as the number of training rounds, and set the ALPHA on different datasets to the best parameters we provided:
movie:5,waimai:3000,data4:1,shopping:2.5,hotel:0.00003.
Users can refer to the average of the accuracy of the predictions of the last 5 training rounds of experiments on the test set (that is, the accuracy that will be printed at runtime). 

For the data we used:
All the datasets we have cleaned and segmented are uploaded in this project and users can find them in Data.

# For the environment of the experiment:
We saved the environment configurations used in our experiments in the requirements.txt.
We record the configuration of the devices we used in our experiment as follows for users' reference: RTX 4090(24GB) * 1,12 vCPU Intel(R) Xeon(R) Platinum 8352V CPU @ 2.10GHz. 
Our experiments were done using jupter, if you want to use jupter for your experiments, please create a .ipynb file in the root directory and copy all the contents of main.py or finetune_teacher.py into it and run this ipynb file.
