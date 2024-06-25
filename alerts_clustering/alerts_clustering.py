import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import DBSCAN
from tqdm import tqdm
import pickle
import logging
import os
from typing import Union
logging.basicConfig(level=logging.INFO)
# from alerts_clustering.utils import calc_homogeneity_score, calc_avg_probability
from .utils import calc_homogeneity_score, calc_avg_probability

class AlertsClustering(object):
    def __init__(self, load_alerts: Union[str, pd.DataFrame], load_model: [str, Word2Vec], save_cluster: str, eps: float, min_sample: int):
        self.load_alerts = load_alerts
        self.load_model = load_model
        self.save_cluster = save_cluster
        self.eps = eps
        self.min_sample = min_sample
        if load_model.endswith('.pkl'):
            with open(load_model, 'rb') as f:
                self.X = pickle.load(f)
        else:
            self.align_alert()

    # 输入 w2v 模型和原始告警 dataframe，返回一个和原来 dataframe 对齐的告警 ndarray
    def align_alert(self):
        # alerts
        if isinstance(self.load_alerts, str):
            self.alerts_df = pd.read_csv(self.load_alerts)
        elif isinstance(self.load_alerts, pd.DataFrame):
            self.alerts_df = self.load_alerts
        else:
            raise(TypeError('load_alerts type error.'))
        # w2v model
        if isinstance(self.load_model, str):
            w2v_model = Word2Vec.load(self.load_model)
        elif isinstance(self.load_model, Word2Vec):
            w2v_model = self.load_model
        else:
            raise TypeError('load_model type error.')
        # 存储按顺序的告警
        res = []
        log_id_list = self.alerts_df.log_id.to_list()
        for log_id in log_id_list:
            res.append(w2v_model.wv.vectors[w2v_model.wv.key_to_index[log_id]])
        self.X = np.array(res)
    

    # 分组聚类函数
    # 输入: 告警的总dataframe, 告警的向量数组
    # 输出: 一个字典, key: 簇号, value: 属于该簇的告警 idx 数组. (未能被聚类的告警都放到簇号为 -1 的簇中) 
    def categorical_cluster(self)->dict:
        logging.info(f'eps: {self.eps}, min_sample: {self.min_sample}')
        # 开始正式编写分组聚类算法
        # 用于存储最终的聚类结果
        # key: 顺序分配的cluster_id, value: 该簇中的所有告警原始下标组成的list
        clusters_dic = dict()
        # 为每个簇分配的id，从0开始递增
        global_cluster_id = 0
        # 存储每一次聚类后未能被聚类的告警的原始id, 最后分配一个-1，存到 clusters_dic 中
        unclustered_alerts_list = list()
        # 首先获得一系列类别
        log_message_list = set(self.alerts_df.log_message.to_list())
        # print(len(log_message_list))
        # 对于每一类告警
        # 暂时先关掉进度条
        for log_message in tqdm(log_message_list, desc='clustering'):
        # for log_message in log_message_list:
            # 这个是当前类别告警的原始下标
            idx_list = self.alerts_df[self.alerts_df.log_message==log_message].index.to_list()
            # 根据这些下标获得他们对应的向量表示 (这个时候下标就变了，需要之后重新映射)
            curX = [self.X[i] for i in idx_list]
            # 对当前类别的告警做聚类
            db = DBSCAN(eps=self.eps, min_samples=self.min_sample, n_jobs=6).fit(curX)
            ser = pd.Series(db.labels_)
            clusters_id_list = ser.value_counts().sort_values(ascending=False).index.to_list()
            # 遍历每一个簇
            for cluster_id in clusters_id_list:
                # 这个是该簇在 curX 中的下标，需要映射回 X 的再用
                cur_id_list = ser[ser==cluster_id].index.to_list()
                # 映射回去
                ori_id_list = list(map(lambda x : idx_list[x], cur_id_list))
                # 可以放到最终的聚类结果里了
                # -1的簇需要单独处理
                if cluster_id == -1:
                    unclustered_alerts_list.extend(ori_id_list)
                # 正常情况下
                else:
                    clusters_dic[global_cluster_id] = ori_id_list
                    global_cluster_id += 1
        # 最后都走完了 再把 -1 扔进去
        # 目前还是决定把 -1 加进去，因为这样 clusters_dic 就可以带着全部的信息了
        clusters_dic[-1] = unclustered_alerts_list
        logging.info(f'聚类完成，有 {len(clusters_dic) - 1} 簇，{len(unclustered_alerts_list)} 个告警未被聚类')
        score = calc_homogeneity_score(self.alerts_df, clusters_dic)
        logging.info(f'簇的 homogeneity_score 为 {score}')
        k = 10
        probability = calc_avg_probability(self.alerts_df, clusters_dic, k)
        logging.info(f'每簇采样 {k} 个告警，有平均 {probability}% 的概率每一簇都能找到最高危险级别的告警')
        # 保存
        if self.save_cluster:
            logging.info('正在保存聚类结果')
            with open(os.path.join(self.save_cluster, f'{self.eps}_{self.min_sample}.clusters'), 'wb') as f:
                pickle.dump(clusters_dic, f)
            logging.info('保存聚类结果完成')
        return clusters_dic



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description = "alerts_clustering: input alerts embedding and output clutering result"
    )
    parser.add_argument('--load_alerts', help='path of alerts csv file')
    parser.add_argument('--load_model', help='path of word2vec model')
    parser.add_argument('--save_cluster', help='path of clustering result')
    parser.add_argument('--eps', type=float, default=2.0, help='eps')
    parser.add_argument('--min_sample', type=int, default=5, help='min_sample')

    args = parser.parse_args()

    ac = AlertsClustering(args.load_alerts, args.load_model, args.save_cluster, args.eps, args.min_sample)
    ac.categorical_cluster()