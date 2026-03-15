import logging
import os
import random

import networkx as nx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_nx_graph_from_config(config: dict):
    """
    根据配置构建 NetworkX MultiGraph 对象
    核心逻辑：读取配置指定的图文件/节点集文件 → 节点ID映射 → 构建带时间属性的无向多重图

    Args:
        config: 数据集预处理配置字典(需包含以下字段)
                - output_graph_path: 处理后的图数据文件路径
                - output_node_dir: 节点集文件所在目录
                - csv_sep: 文件分隔符
                - node_cols: 源/目标节点列名列表 [src_col, tgt_col]
                - time_col: 时间属性列名

    Returns:
        nx.MultiGraph: 带时间属性的无向多重图（已移除自环边）

    Raises:
        KeyError: 配置缺少必要字段
        FileNotFoundError: 图文件/节点集文件不存在
        ValueError: 图文件/节点集文件为空
    """
    required_config_keys = [
        "output_graph_path",  # csv格式的图数据文件
        "output_node_path",  # csv格式的节点集文件
        "csv_sep",
        "node_cols",
    ]
    missing_keys = [key for key in required_config_keys if key not in config]
    if missing_keys:
        raise KeyError(f"构建NX图缺少必要配置字段: {missing_keys}")

    graph_path = config["output_graph_path"]
    node_path = config["output_node_path"]
    csv_sep = config["csv_sep"]
    src_col, tgt_col = config["node_cols"]
    time_col = config.get("time_col", "time")  # 时间列名，默认'time'

    if not os.path.exists(graph_path):
        logger.error(f"图数据文件不存在(配置字段: output_graph_path): {graph_path}")
        raise FileNotFoundError(
            f"Graph file not found (config: output_graph_path): {graph_path}"
        )

    if not os.path.exists(node_path):
        logger.error(f"节点集文件不存在(配置字段: output_node_dir): {node_path}")
        raise FileNotFoundError(
            f"Nodes file not found (config: output_node_dir): {node_path}"
        )

    # ===================== 1. 读取节点集并构建ID映射 =====================
    try:
        nodes_set = pd.read_csv(node_path, names=["node"])
    except Exception as e:
        logger.error(
            f"读取节点集文件失败：{node_path}，分隔符：{csv_sep}，错误：{str(e)}"
        )
        raise RuntimeError(f"Failed to read nodes file: {node_path}") from e

    if nodes_set.empty:
        logger.error(f"节点集文件为空：{node_path}")
        raise ValueError("Nodes file is empty!")

    # 节点ID映射（从1开始编号，0保留给padding）
    node2id = {node: i + 1 for i, node in enumerate(nodes_set["node"])}
    logger.info(f"✅ 加载节点集完成：{node_path}，节点数量：{len(node2id)}")

    # ===================== 2. 读取图数据并校验 =====================
    try:
        df = pd.read_csv(graph_path, sep=csv_sep)
    except Exception as e:
        logger.error(
            f"读取图数据文件失败：{graph_path}，分隔符：{csv_sep}，错误：{str(e)}"
        )
        raise RuntimeError(f"Failed to read graph file: {graph_path}") from e

    if df.empty:
        logger.error(f"图数据文件为空：{graph_path}")
        raise ValueError("Graph file is empty!")

    # 校验必要列（源/目标节点 + 时间列）
    required_cols = [src_col, tgt_col]
    if time_col not in df.columns:
        logger.warning(f"图数据缺少时间列 {time_col}，将不添加时间属性到边")
        edge_attr = None
    else:
        required_cols.append(time_col)
        edge_attr = time_col

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error(
            f"图数据缺少必要列：{graph_path}，缺失列：{missing_cols}，配置节点列：{config['node_cols']}"
        )
        raise KeyError(
            f"Graph file missing columns: {missing_cols} (config node cols: {config['node_cols']})"
        )

    # 节点ID映射转换
    def _map_node_id(nid):
        if nid not in node2id:
            logger.error(f"节点 {nid} 不在节点集中！节点集路径：{node_path}")
            raise KeyError(
                f"Node {nid} not found in nodes set (nodes file: {node_path})!"
            )
        return node2id[nid]

    # 应用ID转换到源/目标节点列
    try:
        df[[src_col, tgt_col]] = df[[src_col, tgt_col]].apply(
            lambda col: col.map(_map_node_id)
        )
    except KeyError:
        raise  # 抛出原有异常，日志已在_map_node_id中记录
    except Exception as e:
        logger.error(f"节点ID映射转换失败: {graph_path}，错误: {str(e)}")
        raise RuntimeError(
            f"Failed to map node IDs for graph file: {graph_path}"
        ) from e

    # ===================== 3. 构建NetworkX图并移除自环 =====================
    graph = nx.from_pandas_edgelist(
        df,
        source=src_col,
        target=tgt_col,
        edge_attr=edge_attr,
        create_using=nx.MultiGraph,
    )

    # 移除自环边
    self_loop_edges = list(nx.selfloop_edges(graph))
    if self_loop_edges:
        logger.warning(f"检测到 {len(self_loop_edges)} 个自环边，已自动移除")
        graph.remove_edges_from(self_loop_edges)

    # 日志输出最终结果
    logger.info(
        f"✅ 成功构建NX图: \n"
        f"  - 图文件：{graph_path}\n"
        f"  - 节点数量：{graph.number_of_nodes()}\n"
        f"  - 边数量：{graph.number_of_edges()}\n"
        f"  - 时间属性：{'已包含' if edge_attr else '未包含'}"
    )

    return graph


def add_attr(graph, min_t=0, max_t=float("inf")):
    # 节点-影响因子-计算
    for node in graph.nodes():
        neighbors = list(graph.neighbors(node))
        if len(neighbors) == 0:
            act = 0
        else:
            deg = len(neighbors)
            avg_nei_deg = sum(
                len(list(graph.neighbors(neighbor))) for neighbor in neighbors
            ) / len(neighbors)
            act = deg + avg_nei_deg
        # 为节点添加属性
        graph.nodes[node]["act"] = act

    # 边-结构链接强度-计算
    for u, v, key in graph.edges(keys=True):
        L_uv = get_links(graph, u, v, min_t, max_t)
        L_sum_u = get_all_links(graph, u, min_t, max_t)
        L_sum_v = get_all_links(graph, v, min_t, max_t)

        CN_uv = len(list(nx.common_neighbors(graph, u, v)))
        D_u = len(list(graph.neighbors(u)))
        D_v = len(list(graph.neighbors(v)))

        sci_uv = (L_uv / L_sum_u) * (L_uv / L_sum_v) + (CN_uv / D_u) * (CN_uv / D_v)
        graph[u][v][key]["sci"] = sci_uv
    return graph


def get_links(graph, u, v, min_t=0, max_t=float("inf")):
    count = 0
    if graph.has_edge(u, v):
        for key in graph[u][v]:  # 遍历节点 u 和 v 之间的所有多重边
            edge_data = graph[u][v][key]
            if "time" in edge_data:
                timestamp = edge_data["time"]
                if min_t <= timestamp <= max_t:
                    count += 1
    return count


def get_all_links(graph, node, min_t=0, max_t=float("inf")):
    count = 0
    for neighbor in graph.neighbors(node):
        for key in graph[node][neighbor]:  # 遍历节点 node 和其邻居之间的所有多重边
            edge_data = graph[node][neighbor][key]
            if "time" in edge_data:
                timestamp = edge_data["time"]
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
        if (
            v2 != u
            and v2 not in adj[u]
            and (u, v2) not in neg_edges
            and (v2, u) not in neg_edges
        ):
            neg_edges.add((u, v2))

    # 2. 补齐随机负样本
    while len(neg_edges) < num_neg:
        u = random.choice(node_list)
        v = random.choice(node_list)
        if (
            u != v
            and v not in adj[u]
            and (u, v) not in neg_edges
            and (v, u) not in neg_edges
        ):
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
