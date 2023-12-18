import numpy as np
import math

# 用于生成输入数据和标签
def data_extract(pos_good,pos_adv):
    num_good = pos_good.shape[0]
    num_adv = pos_adv.shape[0]

    distance = np.array([])
    dis_max = -1
    for i in range(num_good):
        for j in range(num_adv):
            dis = math.sqrt((pos_good[i,0] - pos_adv[j,0]) ** 2 + (pos_good[i,1] - pos_adv[j,1]) ** 2 + (pos_good[i,2] - pos_adv[j,2]) ** 2)
            if dis > dis_max :
                dis_max = dis
            distance = np.append(distance, dis)
    distance.resize(num_good, num_adv)
    distance_not_trans = distance.copy()

    for i in range(num_good):
        for j in range(num_adv):
            distance[i,j] = dis_max - distance[i,j] +1

    return num_good,num_adv,distance,distance_not_trans

def Kuhn_Munkras(n,m,weight):
    #初始化顶标和敌方-我方匹配号
    lx = [np.max(line) for line in weight]
    ly = [0] *m

    match = [-1] *m

    for i in range(n):
        while(1):
            sx = [0]*n
            sy = [0]*m
            found,match = search_path(i,sx,sy,n,m,lx,ly,weight,match)
            # if -1 not in match:
            #     break
            if(found):
                break

            inc = 1e10
            for j in range(n):
                if sx[j]:
                    for q in range(m):
                        if sy[q]==0 and (lx[j] + ly[q] - weight[j,q])<inc:
                            inc = lx[j] + ly[q] - weight[j,q]

            if inc==0:
                print("inc == 0,没有新的相等子图")
            for c in range(n):
                if sx[c]:
                    lx[c] -= inc
            for p in range(m):
                if sy[p]:
                    ly[p] += inc

    return match

def search_path(u,sx,sy,n,m,lx,ly,weight,match):
    """ 给第u个敌方找匹配，这个过程是匈牙利匹配 """
    sx[u]=1
    for i in range(m):
        if sy[i]==0 and (lx[u] + ly[i] == weight[u,i]):
            sy[i] = 1
            if match[i] == -1:
                match[i] = u
                return True,match
            else:
                found,match = search_path(match[i],sx,sy,n,m,lx,ly,weight,match)
                if found:
                    match[i] = u
                    return True,match

    return False,match



def match(goods,advs):

    ini_num_good,num_adv,ini_distance,dis_not_trans = data_extract(goods,advs)
    dis_not_trans = dis_not_trans.T
    ini_distance = ini_distance.T #维度应该是 advnum * goodnum
    distance = ini_distance
    num_good = ini_num_good
    times = int(num_good/num_adv)
    index = [i for i in range(num_good)]
    object_set = {}
    for aid in range(num_adv):
        object_set['{}'.format(int(aid))] = []

    for i in range(times):
        # 由KM算法得到当前的匹配矩阵match
        match = Kuhn_Munkras(num_adv, num_good, distance)

        #记录该轮结果
        index_good_match = []
        for g_index,match_adv in enumerate(match):
            if match_adv != -1:
                object_set['{}'.format(match_adv)].append(index[g_index])
                index_good_match.append(g_index)

        #删除distance矩阵中相应我方数据 和 修正我方现存ID矩阵
        distance = np.delete(distance,index_good_match,axis=1)

        for index2del in sorted(index_good_match, reverse=True):
            del index[index2del]

        num_good = len(index)

    # 处理余下的我方，距离谁最近打谁
    if len(index) !=0 :
        max_indices = np.argmax(distance, axis=0)

        for g, max_advid in enumerate(max_indices):
            object_set['{}'.format(max_advid)].append(index[g])

    # 根据object_set生成adv*num 维度的标签，有配对为1，否则为0
    label_mat = np.zeros([num_adv,ini_num_good])
    for advid,goods_set in object_set.items():
        for goodid in goods_set:
            label_mat[int(advid),goodid] = 1
    label_mat = label_mat.T
    label = np.array([])
    for goodid in label_mat:
        for i in range(num_adv):
            if goodid[i] == 1:
                label = np.append(label,i)

    a = 0
    return dis_not_trans,label













