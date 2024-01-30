import pickle



hotel_skip_cae = "../result/hotel/skip_CAE/5.pkl"
hotel_baseline = "../result/hotel/baseline/baseline312-8-4_03.pkl"

waimai_baseline = "../result/waimai/baseline/baseline312-8-4_02.pkl"
waimai_file_cae_skip = "../result/waimai/skip_CAE/bert312_8_4-02.pkl"
waimai_file_mae_skip = "../result/waimai/skip_MAE/0.pkl"
waimai_file_pkd_skip = "../result/hotel/skip_MAE/0.pkl"

shopping_baseline = "../result/shopping/baseline/0.pkl"
shopping_skip_cae = "../result/shopping/skip_CAE/2.pkl"

shopping_skip_mae = "../result/shopping/skip_MAE/7.pkl"
shopping_first_mae = "../result/shopping/first_MAE/15.pkl"
shopping_last_mae = "../result/shopping/last_MAE/11.pkl"

shopping_skip_pkd = "../result/shopping/skip_pkd/0.pkl"


movie_skip_mae =  "../result/movie/skip_MAE/5.pkl"

movie_baseline = "../result/movie/baseline/baseline312-8-4_02.pkl"
movie_skip_cae = "../result/movie/skip_CAE/skip_CAE-07.pkl"

movie_teacher_hypc = "../Teacher_Model/movie_Model_meta/1/hyper.pkl"

path = hotel_baseline
# 以二进制读取模式打开文件
with open(path, 'rb') as file:
    # 从文件中加载对象
    data = pickle.load(file)
    print(data)
    for i in range(42):
        print(i,': ',data[i])
