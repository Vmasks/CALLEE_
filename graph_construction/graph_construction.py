from tqdm import tqdm
import pandas as pd
import networkx as nx
import logging
from itertools import combinations
from collections import Counter
from gensim.models import Word2Vec
from time import time
from datetime import datetime
import socket
import re
import os
from typing import Union
logging.basicConfig(level=logging.DEBUG)


class GraphConstruction(object):
    def __init__(self, load_alerts: Union[str, pd.DataFrame], ip_sim: float, save_graph: str):
        self.load_alerts = load_alerts
        self.sim = ip_sim
        self.graph = save_graph
        self.read_data()


    def read_data(self):
        # 字段名映射，以备不时之需
        field_mapping_dict = {
            'log_id': 'log_id',
            'timestamp': 'timestamp',
            'dev_ip': 'dev_ip',
            'sip': 'sip',
            'dip': 'dip',
            'dport': 'dport',
            'rule_id': 'rule_id',
            'log_message': 'log_message',
            'payload': 'payload',
            'q_body': 'q_body',
            'r_body': 'r_body',
            'label': 'label',
        }
        # 如果是字符串，就从路径加载
        if isinstance(self.load_alerts, str):
            self.alerts_df = pd.read_csv(self.load_alerts)
        elif isinstance(self.load_alerts, pd.DataFrame):
            self.alerts_df = self.load_alerts
        else:
            raise(TypeError('load_alerts type error.'))
        # 映射字段名
        self.alerts_df = self.alerts_df.rename(columns=field_mapping_dict)
        # 做类型转换，不然保存图序列化保存的时候会保存不上
        self.alerts_df['dport'] = self.alerts_df['dport'].apply(str)
        self.alerts_df['rule_id'] = self.alerts_df['rule_id'].apply(str)
        self.alerts_df['label'] = self.alerts_df['label'].apply(str)
        logging.info(f'成功读取 {len(self.alerts_df)} 条告警')
        

    def construct_base_graph(self):
        """创建基础图，包括告警节点、ip节点、port节点、告警与ip的边、ip与port的边
        """        
        # 遍历所有告警
        for index, row in tqdm(self.alerts_df.iterrows(), desc='basic'):
            # ip 节点，以ip字符串作为id
            self.G.add_node(row['sip'], node_type='ip')
            self.G.add_node(row['dip'], node_type='ip')
            # port 节点，以 ip + ':' + port 命名
            dport = row['dip'] + ':' + row['dport']
            self.G.add_node(dport, node_type='port')
            # 添加 dip 与 dport 之间的边
            self.G.add_edge(row['dip'], dport)
            # 添加告警节点，以 log_id 作为节点 id 告警还应该有时间戳
            self.G.add_node(row['log_id'], log_message=row['log_message'], node_type='alert', timestamp=row['timestamp'])
            # 添加sip与告警之间的边
            self.G.add_edge(row['sip'], row['log_id'])
            # 添加告警与dport之间的边
            self.G.add_edge(row['log_id'], dport)
        logging.debug(f'基础图节点数量: {self.G.number_of_nodes()}')
        logging.debug(f'基础图边数量: {self.G.number_of_edges()}')


    def construct_temporal_alerts_edges(self):
        # 构建同类告警时序边
        # 好像还是有点武断，也可以再加一个时间来约束
        # 条件应该是，同一个 sip， 同种告警，目标端口相同
        hosts_group = self.alerts_df.groupby(['sip', 'log_message', 'dport'])
        logging.debug(f'有 {len(hosts_group)} 用于构建时序的候选组')
        edges_num = self.G.number_of_edges()
        for name, group in tqdm(hosts_group, desc='temporal'):
            group.sort_values('timestamp')
            id_list = group.log_id.to_list()
            time_list = group.timestamp.to_list()
            for i in range(len(id_list) - 1):
                datetime_format = "%Y-%m-%d %H:%M:%S"
                time1 = datetime.strptime(time_list[i], datetime_format)
                time2 = datetime.strptime(time_list[i+1], datetime_format)
                time_difference = time2 - time1
                time_difference_seconds = time_difference.total_seconds()
                if time_difference_seconds < 60:
                    self.G.add_edge(id_list[i], id_list[i+1])
        logging.debug(f'添加完告警时序边后新增了 {self.G.number_of_edges() - edges_num} 条边')


    def construct_ip_simalr_edges(self):
        logging.debug('开始构建ip相似边')
        edges_num = self.G.number_of_edges()
        sentences = []
        sip_list = self.alerts_df.sip.to_list()
        log_message_list = self.alerts_df.log_message.to_list()
        dip_list = self.alerts_df.dip.to_list()
        dport_list = self.alerts_df.dport.to_list()
        for sip, log_message, dip, dport in tqdm(zip(sip_list, log_message_list, dip_list, dport_list), desc='construct sentences'):
            sentences.append([sip, log_message, dip, dport])
        model = Word2Vec(vector_size=20,
                                    window=3,
                                    sg=1, #Skip-Gram
                                    hs=0,
                                    negative=10,
                                    alpha=0.03,
                                    min_alpha=0.0007,
                                    workers=10
                                )
        t = time()
        model.build_vocab(sentences, progress_per=100000)
        logging.info('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))
        # train
        t = time()
        model.train(sentences, total_examples=model.corpus_count, epochs=20, report_delay=1)
        logging.info('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))
        emb = dict()
        for ip in self.alerts_df.sip.to_list():
            try:
                emb[ip] = model.wv[ip]
            except:
                continue
        logging.info(f'共有 {len(emb.keys())} 个ip有嵌入')
        from itertools import combinations
        from scipy.spatial import distance
        res = []
        for combo in tqdm(combinations(emb.keys(), 2)):
            first = combo[0]
            second = combo[1]
            dis = distance.euclidean(emb[first], emb[second])
            if dis <= self.sim:
                res.append(combo)
        logging.info(f'共有 {len(res)} 组sip间相似度小于 {self.sim}')
        for combo in res:
            self.G.add_edge(combo[0], combo[1])
        logging.debug(f'添加完ip相似边后新增了 {self.G.number_of_edges() - edges_num} 条边')


    def construc_payload_nodes_edges(self) -> None:
        pass
        

    def construct_graph(self) -> nx.Graph:        
        # 构建图
        self.G = nx.Graph()
        self.construct_base_graph()
        self.construct_temporal_alerts_edges()
        self.construct_ip_simalr_edges()
        self.construc_payload_nodes_edges()
        logging.info(f'图中共有连通分量 {len(list(nx.connected_components(self.G)))} 个')
        # 保存图
        if self.graph:
            path = os.path.join(self.graph, f'{self.sim}.graph')
            nx.write_gml(self.G, path)
            logging.info(f'已将图保存到{path}')
        
        # 最后返回异质图，供下一步调用
        return self.G


if __name__ == '__main__':
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description = "graph_construction: input alerts csv data and output alerts HIN"
    )
    parser.add_argument('--load_alerts', help='path of input alerts csv file', required=True)
    parser.add_argument('--ip_sim', default=0.5, type=float, help='threshold for two IPs to be considered similar')
    parser.add_argument('--save_graph', help='path of output alerts HIN')
    args = parser.parse_args()
    
    gc = GraphConstruction(args.load_alerts, args.ip_sim, args.save_graph)
    gc.construct_graph()