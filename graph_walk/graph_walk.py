import networkx as nx
from tqdm import tqdm
import random
import pickle
import logging
import os
from typing import Union
logging.basicConfig(level=logging.DEBUG)

# 限制随机游走只能走向相同类型的节点
class GraphWalk(object):
    def __init__(self, load_graph:Union[str, nx.classes.graph.Graph], seq_path:str, gamma: int, walk_len: int) -> None:
        logging.info('正在加载图')
        if isinstance(load_graph, str):
            # self.G = nx.read_gml(load_graph)
            with open(load_graph, 'rb') as f:
                self.G = pickle.load(f)
        elif isinstance(load_graph, nx.classes.graph.Graph):
            self.G = load_graph
        else:
            raise(TypeError('load_graph type error.'))
        logging.info('图加载完毕')
        self.seq_path = seq_path
        self.gamma = gamma
        self.walk_len = walk_len


    # 构建 meta-path based 随机游走所需要的字典 
    def preprocess(self):
        # 只要dest是alert，都需要两层字典
        self.a_a = dict()
        self.a_p = dict()
        # 两层字典
        self.p_a = dict()
        self.a_i = dict()
        # 两层字典
        self.i_a = dict()
        self.i_i = dict()

        self.a_pay = dict()
        # 两层字典
        self.pay_a = dict()

        # 构建节点的邻居字典
        all_nodes = list(self.G.nodes())
        neigh_dic = dict()
        for node in tqdm(all_nodes, desc='neigh dict'):
            neigh_dic[node] = list(self.G.neighbors(node))
        # 遍历图中节点
        logging.debug('开始构建邻居字典')
        for node in tqdm(all_nodes, desc='auxiliary dict'):
            # 如果这是一个 ip 节点 
            # 涉及 AIA: i_a, AIIA, i_i
            if self.G.nodes[node]['node_type'] == 'ip':
                if node not in self.i_a:
                    self.i_a[node] = dict()
                if node not in self.i_i:
                    self.i_i[node] = list()
                # 所有邻居
                neighs = neigh_dic[node]
                for neigh in neighs:
                    neigh_node_type = self.G.nodes[neigh]['node_type']
                    if neigh_node_type == 'alert':
                        neigh_log_message = self.G.nodes[neigh]['log_message'] 
                        if neigh_log_message not in self.i_a[node]:
                            self.i_a[node][neigh_log_message] = list()
                        self.i_a[node][neigh_log_message].append(neigh)
                    elif neigh_node_type == 'ip':
                        self.i_i[node].append(neigh)

            # 如果这是一个 端口节点
            # 涉及 APA: p_a
            elif self.G.nodes[node]['node_type'] == 'port':
                if node not in self.p_a:
                    self.p_a[node] = dict()
                neighs = neigh_dic[node]
                for neigh in neighs:
                    neigh_node_type = self.G.nodes[neigh]['node_type']
                    if neigh_node_type == 'alert':
                        neigh_log_message = self.G.nodes[neigh]['log_message']
                        if neigh_log_message not in self.p_a[node]:
                            self.p_a[node][neigh_log_message] = list()
                        self.p_a[node][neigh_log_message].append(neigh)

            # 如果这是一个告警节点
            elif self.G.nodes[node]['node_type'] == 'alert':
                # if self.G.nodes[node]['log_message'] == 'XSS跨站脚本 xss_midvul_htmltag':
                #     print('get you.')
                #     print(self.G[node])
                #     break
                if node not in self.a_a:
                    self.a_a[node] = dict()
                if node not in self.a_p:
                    self.a_p[node] = list()
                if node not in self.a_i:
                    self.a_i[node] = list()
                if node not in self.a_pay:
                    self.a_pay[node] = list()
                neighs = neigh_dic[node]
                for neigh in neighs:
                    if self.G.nodes[neigh]['node_type'] == 'alert':
                        if self.G.nodes[neigh]['log_message'] not in self.a_a[node]:
                            self.a_a[node][self.G.nodes[neigh]['log_message']] = list()
                        self.a_a[node][self.G.nodes[neigh]['log_message']].append(neigh)
                    elif self.G.nodes[neigh]['node_type'] == 'port':
                        self.a_p[node].append(neigh)
                    elif self.G.nodes[neigh]['node_type'] == 'ip':
                        self.a_i[node].append(neigh)
                    elif self.G.nodes[neigh]['node_type'] == 'payload':
                        self.a_pay[node].append(neigh)

            # 如果这是一个 payload 节点 构建 pay_a 字典
            elif self.G.nodes[node]['node_type'] == 'payload':
                if node not in self.pay_a:
                    self.pay_a[node] = dict()
                neighs = neigh_dic[node]
                for neigh in neighs:
                    if self.G.nodes[neigh]['node_type'] == 'alert':
                        if self.G.nodes[neigh]['log_message'] not in self.pay_a[node]:
                            self.pay_a[node][self.G.nodes[neigh]['log_message']] = list()
                        self.pay_a[node][self.G.nodes[neigh]['log_message']].append(neigh)
            
        logging.debug(f'a_a: {len(self.a_a)}')
        logging.debug(f'a_p: {len(self.a_p)}')
        logging.debug(f'p_a: {len(self.p_a)}')
        logging.debug(f'a_i: {len(self.a_i)}')
        logging.debug(f'i_a: {len(self.i_a)}')
        logging.debug(f'i_i: {len(self.i_i)}') 
        logging.debug(f'pay_a: {len(self.pay_a)}')
        logging.debug(f'a_pay: {len(self.a_pay)}')

    # 限制游走，只能走到告警类型相同的告警节点上
    def get_path(self, node, metapath):
        path = []
        path.append(node)

        # 在这里获取告警类型是什么，之后这条路上只挑相同告警类型的走
        log_msg = self.G.nodes()[node]['log_message']

        if metapath == 'AA':
            for _ in range(self.walk_len): 
                try:
                    neighs = self.a_a[node][log_msg]
                except:
                    break
                if len(neighs) == 0:
                    break
                random_node = random.choice(neighs)
                path.append(random_node)
                node = random_node

        elif metapath == 'APA':
            for _ in range(self.walk_len):
                # 走一个p
                neighs = self.a_p[node]
                if len(neighs) == 0:
                    break
                random_node = random.choice(neighs)
                # path.append(random_node)
                node = random_node
                # 再走一个 a
                try:
                    neighs = self.p_a[node][log_msg]
                except:
                    break
                if len(neighs) == 0:
                    break
                random_node = random.choice(neighs)
                path.append(random_node)
                node = random_node

        elif metapath == 'AIA':
            for _ in range(self.walk_len):
                # 走 i
                neighs = self.a_i[node]
                if len(neighs) == 0:
                    break
                random_node = random.choice(neighs)
                # path.append(random_node)
                node = random_node
                # 再走一个 a
                try:
                    neighs = self.i_a[node][log_msg]
                except:
                    break
                if len(neighs) == 0:
                    break
                random_node = random.choice(neighs)
                path.append(random_node)
                node = random_node

        elif metapath == 'AIIA':
            for _ in range(self.walk_len):
                # 走 i
                neighs = self.a_i[node]
                if len(neighs) == 0:
                    break
                random_node = random.choice(neighs)
                # path.append(random_node)
                node = random_node
                # 再走一个 i
                neighs = self.i_i[node]
                if len(neighs) == 0:
                    break
                random_node = random.choice(neighs)
                # path.append(random_node)
                node = random_node
                # 再走一个 a
                try:
                    neighs = self.i_a[node][log_msg]
                except:
                    break
                if len(neighs) == 0:
                    break
                random_node = random.choice(neighs)
                path.append(random_node)
                node = random_node

        elif metapath == 'APayA':
            for _ in range(self.walk_len):
                # 走一个 Pay
                neighs = self.a_pay[node]
                if len(neighs) == 0:
                    break
                random_node = random.choice(neighs)
                # path.append(random_node)
                node = random_node
                # 再走一个A
                try:
                    neighs = self.pay_a[node][log_msg]
                except:
                    break
                random_node = random.choice(neighs)
                path.append(random_node)
                node = random_node
        return path

    def metapath_based_random_walk(self):
        self.preprocess()
        paths = []
        for node in tqdm(self.G.nodes(), desc='random walk'):
            if self.G.nodes[node]['node_type'] != 'alert':
                continue
            for i in range(self.gamma):
                paths.append(self.get_path(node, 'AA'))
                paths.append(self.get_path(node, 'APA'))
                paths.append(self.get_path(node,'AIA'))
                paths.append(self.get_path(node, 'AIIA'))
                paths.append(self.get_path(node, 'APayA'))
        logging.debug(f'共得到{len(paths)}条序列')
        # 去除长度为 1 的序列
        paths = [path for path in paths if len(path) > 1]
        # 保存序列
        if self.seq_path:
            with open(os.path.join(self.seq_path, f'{self.gamma}_{self.walk_len}.paths'), 'wb') as f:
                pickle.dump(paths, f)
            logging.info(f'保存序列完成，共写入{len(paths)}条序列')

        return paths


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description = "graph_walk: input networkx graph and output seq collected through meta-path-based random walks"
    )
    parser.add_argument('--load_graph', help='path of input graph file')
    parser.add_argument('--save_seq', help='path of output seq file')
    parser.add_argument('--gamma', default=4, type=int, help='number of samples per node')
    parser.add_argument('--walk_len', default=10, type=int, help='path length of random walk')
    args = parser.parse_args()

    gw = GraphWalk(args.load_graph, args.save_seq, args.gamma, args.walk_len)
    gw.metapath_based_random_walk()
    
    
     