
import numpy as np
import matplotlib.pyplot as plt
from assignment.label_config import match
import csv


def train_data_gen(samples,good_nums,adv_nums,mode):
    # 生成随机大小的二维地图
    # map_x = 160000
    # map_y = 50000
    # map_z = 80000

    all_data = np.empty((0,adv_nums,good_nums))
    all_label = np.empty((0,adv_nums,good_nums))

    for i in range(samples):
        # 在地图上随机生成我方位置
        points_x_good = np.random.uniform(30000, 40000, size=good_nums)
        points_y_good = np.random.uniform(40000, 50000, size=good_nums)
        points_z_good = np.random.uniform(73000, 78000, size=good_nums)

        # 在地图上随机生成敌方位置

        points_x_adv = np.random.uniform(145000, 150000, size=adv_nums)
        points_y_adv = np.random.uniform(20000, 25000, size=adv_nums)
        points_z_adv = np.random.uniform(73000, 80000, size=adv_nums)

        # # 在地图上随机生成我方位置
        # points_x_good = np.random.uniform(0, 400, size=good_nums)
        # points_y_good = np.random.uniform(0, 400, size=good_nums)
        # points_z_good = np.random.uniform(0, 400, size=good_nums)
        #
        # # 在地图上随机生成敌方位置
        #
        # points_x_adv = np.random.uniform(400, 800, size=adv_nums)
        # points_y_adv = np.random.uniform(400, 800, size=adv_nums)
        # points_z_adv = np.random.uniform(400, 800, size=adv_nums)
        # # pos_adv = np.array([points_x_adv, points_y_adv, points_z_adv]).T


        pos_adv = np.array([points_x_adv, points_y_adv, points_z_adv]).T

        pos_good = np.array([points_x_good,points_y_good,points_z_good]).T
        data,label = match(pos_good,pos_adv)
        label_01 = np.zeros((adv_nums,good_nums))
        for gid,advid in enumerate(label):
            label_01[int(advid),gid] = 1

        all_data = np.concatenate((all_data, data[np.newaxis,...]), axis=0)
        all_label = np.append(all_label,label_01[np.newaxis,...], axis=0)
    all_label = all_label.reshape(-1,100)


    flatten_data = all_data.reshape(all_data.shape[0],-1)

    if mode == 0:
        with open('dataset.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(flatten_data)

        with open('dataset_label.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(all_label)
    if mode == 1:
        with open('validset.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(flatten_data)

        with open('validset_label.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(all_label)
    elif mode == 2:
        with open('testset.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(flatten_data)

        with open('testset_label.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(all_label)

train_data_gen(20000,10,10,0) #mode==0构建训练集
train_data_gen(4000, 10, 10,1) #mode==1构建验证集

train_data_gen(2000,10,10,2) #mode==1构建测试集









