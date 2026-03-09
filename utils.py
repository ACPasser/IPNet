import os
import random
import numpy as np
import pandas as pd
import networkx as nx


# Check the existence of directory(file) path, if not, create one
def check_and_make_path(file_path):
    if file_path == '':
        return
    if not os.path.exists(file_path):
        os.makedirs(file_path)


# Get networkx graph object from file path.
def get_nx_graph(file_path, nodes_set_path, sep='\t'):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Graph file not found: {file_path}")
    if not os.path.exists(nodes_set_path):
        raise FileNotFoundError(f"Nodes file not found: {nodes_set_path}")
        
    # 读取全图所有节点，将节点映射为编号
    nodes_set = pd.read_csv(nodes_set_path, names=['node'])
    if nodes_set.empty:
        raise ValueError("Nodes file is empty!")
    node2id = {node: i + 1 for i, node in enumerate(nodes_set['node'])} # 从1开始编号，0保留给padding
    
    # 加载整张图
    df = pd.read_csv(file_path, sep=sep)
    if df.empty:
        raise ValueError("Graph file is empty!")
    for col in ['from_id', 'to_id']:
        if col not in df.columns:
            raise KeyError(f"Graph file missing column: {col}")
        
    def trans_id(nid):
        if nid not in node2id:
            raise KeyError(f"Node {nid} not found in nodes set!")
        return node2id[nid]

    df[['from_id', 'to_id']] = df[['from_id', 'to_id']].apply(lambda col: col.map(trans_id))

    graph = nx.from_pandas_edgelist(df, "from_id", "to_id", edge_attr='time', create_using=nx.MultiGraph)
    graph.remove_edges_from(nx.selfloop_edges(graph))
    return graph

def add_attr(graph, min_t=0, max_t=float('inf')):
    # 节点-影响因子-计算
    for node in graph.nodes():
        neighbors = list(graph.neighbors(node))
        if len(neighbors) == 0:
            act = 0
        else:
            deg = len(neighbors)
            avg_nei_deg = sum(len(list(graph.neighbors(neighbor))) for neighbor in neighbors) / len(neighbors)
            act = deg + avg_nei_deg
        # 为节点添加属性
        graph.nodes[node]['act'] = act
    
    # 边-结构链接强度-计算
    for u, v, key in graph.edges(keys=True):
        L_uv = get_links(graph, u, v, min_t, max_t)
        L_sum_u = get_all_links(graph, u, min_t, max_t)
        L_sum_v = get_all_links(graph, v, min_t, max_t)
        
        CN_uv = len(list(nx.common_neighbors(graph, u, v)))
        D_u = len(list(graph.neighbors(u)))
        D_v = len(list(graph.neighbors(v)))
        
        sci_uv = (L_uv / L_sum_u) * (L_uv / L_sum_v) + (CN_uv / D_u) * (CN_uv / D_v)
        graph[u][v][key]['sci'] = sci_uv
    return graph
            
def get_links(graph, u, v, min_t=0, max_t=float('inf')):
    count = 0
    if graph.has_edge(u, v):
        for key in graph[u][v]:  # 遍历节点 u 和 v 之间的所有多重边
            edge_data = graph[u][v][key]
            if 'time' in edge_data:
                timestamp = edge_data['time']
                if min_t <= timestamp <= max_t:
                    count += 1
    return count

def get_all_links(graph, node, min_t=0, max_t=float('inf')):
    count = 0
    for neighbor in graph.neighbors(node):
        for key in graph[node][neighbor]:  # 遍历节点 node 和其邻居之间的所有多重边
            edge_data = graph[node][neighbor][key]
            if 'time' in edge_data:
                timestamp = edge_data['time']
                if min_t <= timestamp <= max_t:
                    count += 1
    return count

def negative_sampling(nodes, adj, num_neg, hard_ratio=0.1, seed=None):
    """
    Args:
        nodes: 节点列表
        adj: 预构建的邻接集合字典 {node: set(neighbors)}
        num_neg: 需要采样的数量
        hard_ratio: 硬负样本(2-hop)所占的比例(小数据集建议0.3~0.5)
    """
    if seed is not None:
        random.seed(seed)

    neg_edges = set()
    num_hard = int(num_neg * hard_ratio)
    node_list = list(nodes)
    
    # 尝试：根据节点的度进行加权采样（💥 别用，效果很差）
    # node_degrees = {n: len(adj[n]) for n in node_list}

    # 1. 采样硬负样本 (2-hop)
    attempts = 0
    while len(neg_edges) < num_hard and attempts < num_hard * 5:  # 尝试次数放宽到5倍
        attempts += 1
        u = random.choice(node_list)
        if not adj[u]:
            continue
        
        # 随机选u的1-hop邻居v1
        v1 = random.choice(list(adj[u]))
        if not adj[v1]:
            continue
        
        # 随机选v1的1-hop邻居v2（u的2-hop）
        v2 = random.choice(list(adj[v1]))
        if v2 != u and v2 not in adj[u] and (u, v2) not in neg_edges and (v2, u) not in neg_edges:
            neg_edges.add((u, v2))

    # 2. 补齐随机负样本
    while len(neg_edges) < num_neg:
        u = random.choice(node_list)
        v = random.choice(node_list)
        if u != v and v not in adj[u] and (u, v) not in neg_edges and (v, u) not in neg_edges:
            neg_edges.add((u, v))
    
    return list(neg_edges)


class EarlyStopMonitor(object):
    def __init__(self, max_round=3, higher_better=True, tolerance=1e-3):
        self.max_round = max_round
        self.num_round = 0

        self.epoch_count = 0
        self.best_epoch = 0

        self.last_best = None
        self.higher_better = higher_better
        self.tolerance = tolerance

    def early_stop_check(self, curr_val):
        if not self.higher_better:
            curr_val *= -1
        if self.last_best is None:
            self.last_best = curr_val
        elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
            self.last_best = curr_val
            self.num_round = 0
            self.best_epoch = self.epoch_count
        else:
            self.num_round += 1
        self.epoch_count += 1
        return self.num_round >= self.max_round