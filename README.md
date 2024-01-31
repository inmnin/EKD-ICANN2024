# MAKD
We provide a pytorch version implementation of the MAKD method.
Please change train_type to mae and data_set_type to the corresponding dataset name (movie,waimai,data4,shopping or hotel) in main.py. Then set 15 rounds as the number of training rounds, and set the ALPHA_VITKD parameters on different datasets to the best parameters we provided:
movie:5,waimai:3000,data4:1,shopping:2.5,hotel:0.00003.

Please calculate the average of the accuracy of the predictions of the last 5 rounds of experiments on the test set (that is, the accuracy that will be printed at runtime). The base parameters used for the teacher model are from https://huggingface.co/bert-base-chinese, and users can either fine-tune them themselves on different datasets before testing them, or wait until the end of the anonymization period, when we will upload a link to the weights of the teacher model we used after fine-tuning them individually on all datasets. 

The fine-tuned teacher model weights for different datasets are expected to be stored in different folders of Teacher_Model. All the datasets we have cleaned and segmented are uploaded in this project and users can find them in Data.
