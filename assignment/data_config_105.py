
import numpy as np
import matplotlib.pyplot as plt
from assignment.label_config import match
import csv


def train_data_gen(samples,good_nums,adv_nums,mode):
    # 生成随机大小的二维地图
    # map_x = 160000
    # map_y = 50000
    # map_z = 80000


    all_data = np.empty((0,10,5))
    all_pos = np.empty((0,15,10))
    all_label = np.empty((0,10))

    for i in range(samples):
        re_pos = np.zeros([3,adv_nums,good_nums])
        # 在地图上随机生成我方位置
        points_x_good = np.random.uniform(30000, 40000, size=good_nums)
        points_y_good = np.random.uniform(40000, 50000, size=good_nums)
        points_z_good = np.random.uniform(73000, 78000, size=good_nums)

        # 在地图上随机生成敌方位置

        points_x_adv = np.random.uniform(145000, 150000, size=adv_nums)
        points_y_adv = np.random.uniform(20000, 25000, size=adv_nums)
        points_z_adv = np.random.uniform(73000, 80000, size=adv_nums)
        pos_adv = np.array([points_x_adv, points_y_adv, points_z_adv]).T

        for i in range(good_nums):
            for j in range(adv_nums):
                re_pos[0,j,i] = points_x_adv[j] - points_x_good[i]
                re_pos[1,j,i] = points_y_adv[j] - points_y_good[i]
                re_pos[2,j,i] = points_z_adv[j] - points_z_good[i]
        flatten_pos = re_pos.reshape(-1,re_pos.shape[2])

        pos_good = np.array([points_x_good,points_y_good,points_z_good]).T
        data,label = match(pos_good,pos_adv)
        data = data.T
        # all_data = np.concatenate((all_data, data[np.newaxis,...]), axis=0)
        all_pos = np.concatenate((all_pos, flatten_pos[np.newaxis,...]), axis=0)

        all_label = np.concatenate((all_label, label[np.newaxis,...]), axis=0)


    flatten_data = all_pos.reshape(all_pos.shape[0],-1)

    if mode == 0:
        with open('dataset105.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(flatten_data)

        with open('dataset_label105.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(all_label)
    if mode == 1:
        with open('validset105.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(flatten_data)

        with open('validset_label105.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(all_label)
    elif mode == 2:
        with open('testset105.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(flatten_data)

        with open('testset_label105.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(all_label)

train_data_gen(20000,10,5,0) #mode==0构建训练集
train_data_gen(4000, 10, 5,1) #mode==1构建验证集

train_data_gen(2000,10,5,2) #mode==1构建测试集









