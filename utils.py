import pytz
import datetime
import pandas as pd
from time import time
from typing import List
import pandas as pd
from sklearn.metrics import homogeneity_score
import math
import numpy as np
from collections import Counter
import math
from tqdm import tqdm
import logging
import pickle
logging.basicConfig(level=logging.INFO)

def log(func):
    def wrapper(*args, **kw):
        print('开始时间: ' + datetime.datetime.now(pytz.timezone('PRC')).strftime("%Y-%m-%d_%H:%M"))
        t = time()
        res = func(*args, **kw)
        print('耗时: {} mins'.format(round((time() - t) / 60, 2)))
        print('结束时间: ' + datetime.datetime.now(pytz.timezone('PRC')).strftime("%Y-%m-%d_%H:%M"))
        return res
    return wrapper


# deepcase 输出的簇是一个由  clusters 和 labels 两列组成的 DataFrame
# 将 deepcase 输出的簇转化为我自己定义的字典类型的格式
def process_deepcase_clusters(clusters_df: pd.DataFrame):
    clusters_id_list = clusters_df.clusters.to_list()
    clusters_dict = dict()
    for i, cluster_id in enumerate(clusters_id_list):
        if cluster_id not in clusters_dict:
            clusters_dict[cluster_id] = list()
        clusters_dict[cluster_id].append(i)
    return clusters_dict



# 描述：给定聚类结果，目标平均概率，alerts_df, 初始k值 。计算需要采样数量k，
# 输入：聚类结果, 目标平均概率，alerts_df, 初始k值
# 输出：k, 当前平均概率
def cacl_k_value(clusters_dict, tar_probability, alerts_df, init_k=1):
    label_list = alerts_df.label.to_list()

    # 提前构建 labels_dict key: 簇id, value: 簇内告警label的list
    def get_labels_dict(clusters_dict, label_list):
        labels_dict = dict()
        for cluster_id in tqdm(clusters_dict.keys(), desc='labels_dict'):
            if cluster_id == -1:
                continue
            labels_dict[cluster_id] = np.take(label_list, clusters_dict[cluster_id])
        return labels_dict
    
    # 给定预先计算好的 labels_dict 和 每簇采样个数 k，计算平均概率
    def cacl_avg_probability(labels_dict, k):
        p_list = list()
        for cluster_id in labels_dict.keys():
            # labels_dict 中没有 key == -1 的情况
            # cur_label_list = np.take(label_list, clusters_dict[cluster_id])
            cur_label_list = labels_dict[cluster_id]
            # 当前簇中的告警数量
            cur_alerts_num = len(cur_label_list)
            counter_dict = Counter(cur_label_list)
            counter_list = sorted(counter_dict.items(), key=lambda item: item[0])
            # 只有一个危险级别 或 采样数量已经超过了当前簇大小
            if len(counter_list) == 1 or k >= cur_alerts_num:
                p_list.append(1)
            else:
                p = 1 - math.comb(cur_alerts_num - counter_list[-1][1], k) / math.comb(cur_alerts_num, k)
                p_list.append(p)
        return np.average(p_list)
    
    
    labels_dict = get_labels_dict(clusters_dict, label_list)
    # 根据聚类情况以及要求的平均概率，计算每簇需要采样多少个，就以10为起点
    # 低了就加，高了就减
    k = init_k
    res = cacl_avg_probability(labels_dict, k)
    if res < tar_probability: 
        while res < tar_probability:
            k += 1
            res = cacl_avg_probability(labels_dict, k)
            print(res)
        return k, res
    else:
        while res >= tar_probability:
            if k == 1:
                return 1, res
            k -= 1
            res = cacl_avg_probability(labels_dict, k)
            print(res)
        return k + 1, res


# 计算工作量减少百分比
def cacl_workload_reduction(clusters_dict, k):
    unclustered_num = len(clusters_dict[-1])
    total = 0
    manual_num = 0
    total += unclustered_num
    for cluster_id in clusters_dict.keys():
        if cluster_id == -1:
            continue
        total += len(clusters_dict[cluster_id])
        manual_num += min(k, len(clusters_dict[cluster_id]))
    return 1 - (manual_num + unclustered_num) / total



# 给定 alerts_df 和 聚类结果  计算同质性分数
def calc_homogeneity_score(alerts_df: pd.DataFrame, clusters_dic):
    
    def construct_labels_true():
        labels_true = alerts_df.label.to_list()
        to_del_indices = clusters_dic[-1]
        for idx in sorted(to_del_indices, reverse=True):
            del labels_true[idx]
        return labels_true
    
    labels_true = construct_labels_true()
    # 构造 labels_pred
    labels_pred = [0] * 500_000
    for key in clusters_dic.keys():
        for idx in clusters_dic[key]:
            labels_pred[idx] = key
    labels_pred = list(filter((-1).__ne__, labels_pred))
    return homogeneity_score(labels_true, labels_pred)
    

# 把上面那些方法综合起来
# 给定一个 clusters_dicts 的列表，计算出 workload_reduction 最大的那个，保存所有 clusters dict 的相关结果到 save_results 中
# clusters_dicts 是字典嵌套字典 key: 簇文件名 value: {key: cluster_id, value: 簇内告警id}
def comprehensive(alerts_df: pd.DataFrame, clusters_dicts: dict, tar_probability: float, save_results: str):
    results = list()
    max_reduction = 0
    champion = None
    for clusters_dict_name in tqdm(clusters_dicts.keys(), desc='cacl_clusters_dicts'):
        clusters_dict = clusters_dicts[clusters_dict_name]
        homogeneity_score = calc_homogeneity_score(alerts_df, clusters_dict)
        k, probability = cacl_k_value(clusters_dict, tar_probability, alerts_df)
        workload_reduction = cacl_workload_reduction(clusters_dict, k)
        # 把上面得到的结果拼到一个 tuple 里
        cur_res = (clusters_dict_name, homogeneity_score, k, probability, workload_reduction)
        if workload_reduction > max_reduction:
            champion = cur_res
            max_reduction = workload_reduction
        results.append(cur_res)
    logging.info('Calculation completed.')
    logging.info(f'champion: {champion}')

    if save_results:
        with open(save_results, 'wb') as f:
            pickle.dump({
                'champion': champion,
                'results': results
            }, f)

if __name__ == '__main__':
    pass    
    