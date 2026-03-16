import os
import random
import networkx as nx
import numpy as np
import warnings
import logging
from joblib import Parallel, delayed
from collections import defaultdict
from tqdm import tqdm
from data.data_utils import add_attr


logger = logging.getLogger(__name__)


class DataLoader:
    """
    图数据加载器：负责图数据的分割、过滤、随机游走提取、节点交互序列提取等操作
    """

    def __init__(
        self,
        graph: nx.MultiGraph,
        train_ratio: float = 0.5,
        val_ratio: float = 0.3,
        time_attr: str = "time",
    ):
        """
        初始化数据加载器
        :param graph: 原始多图数据(nx.MultiGraph)
        :param train_ratio: 训练集边占比 (0,1)
        :param val_ratio: 验证集边占比 (0,1)
        :param time_attr: 边的时间戳属性名(默认"time")
        """
        if not isinstance(graph, nx.MultiGraph):
            raise TypeError(f"graph必须是nx.MultiGraph类型, 当前为{type(graph)}")
        if graph.number_of_edges() == 0:
            raise ValueError("输入的图为空（无任何边）")

        self.graph = graph
        self.time_attr = time_attr
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio

        # 基础属性
        self.full_nodes = list(graph.nodes())
        self.num_nodes = len(self.full_nodes)

        # 分割数据集
        self._split_train_val_test()

        # 图的核心统数据
        stats = self._calculate_graph_stats(self.graph)

        self.num_nodes = stats["num_nodes"]
        self.total_edges = stats["total_edges"]
        self.num_unique_edges = stats["num_unique_edges"]
        self.avg_degree = stats["avg_degree"]
        self.avg_interact_freq = stats["avg_interact_freq"]
        self.density = stats["density"]
        self.avg_path_len = stats["avg_path_len"]

    def _split_train_val_test(self) -> None:
        """
        按时间顺序划分训练/验证/测试集，保证验证/测试集节点仅包含训练集节点
        """
        if not (0 < self.train_ratio < 1 and 0 < self.val_ratio < 1):
            raise ValueError(
                f"训练/验证比例必须在(0,1)范围内! train={self.train_ratio}, val={self.val_ratio}"
            )
        if self.train_ratio + self.val_ratio >= 1:
            raise ValueError(
                f"训练+验证比例需小于1! 当前总和={self.train_ratio + self.val_ratio}"
            )

        edges_with_attr = list(self.graph.edges(data=True))  # 保留所有边属性
        self._validate_time_attr(edges_with_attr)

        try:
            sorted_edges = sorted(
                edges_with_attr,
                key=lambda x: x[2][self.time_attr],  # x=(u, v, attr_dict)
            )
        except Exception as e:
            raise RuntimeError(f"按时间戳排序失败：{str(e)}")

        # Split
        total_edges = len(sorted_edges)
        train_end = int(total_edges * self.train_ratio)
        val_end = int(total_edges * (self.train_ratio + self.val_ratio))

        train_edges_raw = sorted_edges[:train_end]
        val_edges_raw = sorted_edges[train_end:val_end]
        test_edges_raw = sorted_edges[val_end:]

        # train graph
        self.train_graph = nx.MultiGraph()
        self.train_graph.add_edges_from(train_edges_raw)
        train_nodes = set(self.train_graph.nodes())

        # val graph
        val_edges_filtered = [
            (u, v, attr)
            for u, v, attr in val_edges_raw
            if u in train_nodes and v in train_nodes
        ]
        self.val_graph = nx.MultiGraph()
        self.val_graph.add_edges_from(val_edges_filtered)

        # test graph
        test_edges_filtered = [
            (u, v, attr)
            for u, v, attr in test_edges_raw
            if u in train_nodes and v in train_nodes
        ]
        self.test_graph = nx.MultiGraph()
        self.test_graph.add_edges_from(test_edges_filtered)

        # 统一节点集合（保证子图节点数与训练集一致）
        self.val_graph.add_nodes_from(train_nodes)
        self.test_graph.add_nodes_from(train_nodes)

        # 输出统计信息
        self._print_split_stats(
            total_edges,
            len(train_edges_raw),
            len(val_edges_raw),
            len(test_edges_raw),
            len(self.train_graph.edges()),
            len(self.val_graph.edges()),
            len(self.test_graph.edges()),
        )

    def _validate_time_attr(self, edges_with_attr: list) -> None:
        """校验时间戳属性的有效性"""
        for u, v, d in edges_with_attr:
            if self.time_attr not in d:
                raise KeyError(f"边({u},{v})缺少时间戳属性 '{self.time_attr}'")

    def _print_split_stats(
        self,
        total_edges: int,
        train_raw: int,
        val_raw: int,
        test_raw: int,
        train_filtered: int,
        val_filtered: int,
        test_filtered: int,
    ) -> None:
        """打印数据集分割统计信息（格式化输出）"""
        total_filtered = train_filtered + val_filtered + test_filtered
        filtered_out = total_edges - total_filtered

        # 计算比例
        train_ratio_raw = train_raw / total_edges if total_edges > 0 else 0
        val_ratio_raw = val_raw / total_edges if total_edges > 0 else 0
        test_ratio_raw = test_raw / total_edges if total_edges > 0 else 0

        train_ratio_filtered = (
            train_filtered / total_filtered if total_filtered > 0 else 0
        )
        val_ratio_filtered = val_filtered / total_filtered if total_filtered > 0 else 0
        test_ratio_filtered = (
            test_filtered / total_filtered if total_filtered > 0 else 0
        )

        # 格式化输出
        logger.info("📊 数据集划分完成")
        logger.info(
            f"   ├─ 原始 | 总边数: {total_edges} | 训练集: {train_raw} ({train_ratio_raw:.2%}) | 验证集: {val_raw} ({val_ratio_raw:.2%}) | 测试集: {test_raw} ({test_ratio_raw:.2%})"
        )
        logger.info(
            f"   ├─ 过滤后 | 总边数: {total_filtered} | 训练集: {train_filtered} ({train_ratio_filtered:.2%}) | 验证集: {val_filtered} ({val_ratio_filtered:.2%}) | 测试集: {test_filtered} ({test_ratio_filtered:.2%})"
        )
        logger.info(
            f"   └─ 过滤信息 | 共过滤掉 {filtered_out} 条边(验证/测试集含训练集外节点)"
        )

    def preprocess(
        self, task_type: str, mask_ratio: float = 0.1, seed: int = 42
    ) -> tuple[dict, dict, dict]:
        """
        训练前统一预处理: 边去重、邻接表生成、节点Mask(仅归纳式)
        :param task_type: 任务类型 "T"(Transductive)/"I"(Inductive)
        :param mask_ratio: 归纳式任务的节点Mask比例 (0,1)
        :param random_seed: 随机种子
        :return: (train_data, val_data, test_data) 预处理后的数据集字典
        """
        if task_type not in ["T", "I"]:
            raise ValueError(f"task_type必须为'T'或'I'，当前值: {task_type}")
        if not (0 < mask_ratio < 1) and task_type == "I":
            raise ValueError(
                f"归纳式任务mask_ratio必须在(0,1)之间! 当前值: {mask_ratio}"
            )

        if task_type == "I":
            # 节点Mask + 过滤测试/验证集
            (
                self.train_graph_masked,
                self.val_graph_filtered,
                self.test_graph_filtered,
            ) = self.mask_nodes_and_filter(
                train_graph=self.train_graph.copy(),  # 深拷贝保护原始数据
                val_graph=self.val_graph.copy(),
                test_graph=self.test_graph.copy(),
                mask_ratio=mask_ratio,
                random_seed=seed,
            )
        else:
            self.train_graph_masked = self.train_graph
            self.val_graph_filtered = self.val_graph
            self.test_graph_filtered = self.test_graph

        train_data = self._preprocess_single_graph(self.train_graph_masked, name="训练")
        val_data = self._preprocess_single_graph(self.val_graph_filtered, name="验证")
        test_data = self._preprocess_single_graph(self.test_graph_filtered, name="测试")
        return train_data, val_data, test_data

    def _preprocess_single_graph(
        self, graph: nx.MultiGraph, name: str = "train"
    ) -> dict:
        """
        单图预处理：边去重（保留时间属性）、邻接表生成、样本统计
        :param graph: 待处理的nx.MultiGraph
        :param name: 数据集名称（用于日志打印）
        :return: 预处理后的数据集字典
        """
        if not isinstance(graph, nx.MultiGraph):
            raise TypeError(f"graph必须是nx.MultiGraph类型，当前为{type(graph)}")
        if graph.number_of_nodes() == 0:
            raise ValueError(f"{name}集图为空（无节点）")

        # 1. 边去重 + 打乱
        edges_list = list(set(tuple(edge) for edge in graph.edges()))
        edges = np.array(edges_list)
        np.random.shuffle(edges)

        # 2. 生成邻接表（用于负采样）
        nodes = list(graph.nodes())
        adj = {n: set(graph.neighbors(n)) for n in nodes}

        # 3. 样本数量统计并打印
        pos_num = len(edges)  # 正样本数
        neg_num = pos_num  # 负样本数（平衡采样）
        total_num = pos_num + neg_num
        raw_pos_num = len(list(graph.edges()))  # 去重前正样本数
        dup_rate = 1 - pos_num / raw_pos_num if raw_pos_num > 0 else 0

        logger.info(f"📊 {name}集样本统计（边去重后）:")
        logger.info(
            f"   ├─ 正样本数: {pos_num} (去重前: {raw_pos_num}, 重复率: {dup_rate:.2%})"
        )
        logger.info(f"   ├─ 负样本数: {neg_num} (平衡采样)")
        logger.info(f"   ├─ 总样本数: {total_num} (正负样本比: 1:1)")
        logger.info(f"   └─ 节点数: {len(nodes)} | 邻接表大小: {len(adj)}")

        return {
            "edges": edges,  # 去重+打乱后的正样本边
            "pos_num": pos_num,  # 正样本数
            "neg_num": neg_num,  # 负样本数
            "total_num": total_num,  # 总样本数
            "nodes": nodes,  # 节点列表
            "adj": adj,  # 邻接表（用于负采样）
            "raw_pos_num": raw_pos_num,  # 去重前正样本数
            "dup_rate": dup_rate,  # 重复率
        }

    def mask_nodes_and_filter(
        self,
        train_graph: nx.MultiGraph,
        val_graph: nx.MultiGraph,
        test_graph: nx.MultiGraph,
        mask_ratio: float = 0.1,
        random_seed: int = 42,
    ) -> tuple[nx.MultiGraph, nx.MultiGraph, nx.MultiGraph]:
        """
        归纳式任务专用: 随机Mask训练集节点, 同步过滤验证/测试集(仅保留Mask节点相关边)
        :param train_graph: 训练图
        :param val_graph: 验证图
        :param test_graph: 测试图
        :param mask_ratio: Mask节点比例 (0,1)
        :param random_seed: 随机种子
        :return: (masked_train_graph, filtered_val_graph, filtered_test_graph)
        """
        if not (0 < mask_ratio < 1):
            raise ValueError(
                f"归纳式任务mask_ratio必须在(0,1)之间! 当前值: {mask_ratio}"
            )

        random.seed(random_seed)
        np.random.seed(random_seed)

        # --- 修改前 ---
        orig_train_nodes = train_graph.number_of_nodes()
        orig_train_edges = train_graph.number_of_edges()
        orig_val_edges = val_graph.number_of_edges()
        orig_test_edges = test_graph.number_of_edges()

        # sample
        train_nodes = list(train_graph.nodes())
        sample_num = int(mask_ratio * len(train_nodes))
        sampled_nodes = random.sample(train_nodes, max(1, sample_num))
        sampled_nodes_set = set(sampled_nodes)

        # 1. Mask训练集：移除采样节点及其关联边
        train_graph.remove_nodes_from(sampled_nodes)

        # 2. 过滤验证集：仅保留至少一端在Mask节点中的边
        val_edges_to_keep = [
            (u, v, k)
            for u, v, k in val_graph.edges(keys=True)
            if u in sampled_nodes_set or v in sampled_nodes_set
        ]
        val_graph_filtered = nx.MultiGraph()
        val_graph_filtered.add_edges_from(val_edges_to_keep)
        val_graph_filtered.add_nodes_from(sampled_nodes_set)

        # 3. 过滤测试集：逻辑同验证集
        test_edges_to_keep = [
            (u, v, k)
            for u, v, k in test_graph.edges(keys=True)
            if u in sampled_nodes_set or v in sampled_nodes_set
        ]
        test_graph_filtered = nx.MultiGraph()
        test_graph_filtered.add_edges_from(test_edges_to_keep)
        test_graph_filtered.add_nodes_from(sampled_nodes_set)

        # --- 修改后 ---
        new_train_nodes = train_graph.number_of_nodes()
        new_train_edges = train_graph.number_of_edges()
        new_val_edges = val_graph_filtered.number_of_edges()
        new_test_edges = test_graph_filtered.number_of_edges()

        mask_ratio_actual = sample_num / len(train_nodes) * 100
        logger.info(
            f"✅ [Inductive Masking] 从训练集中Mask掉{len(sampled_nodes)}个节点 (占比: {mask_ratio_actual:.1f}%)"
        )
        logger.info(
            f"  ├─ 训练图 (Train): 节点 {orig_train_nodes} → {new_train_nodes} | 边 {orig_train_edges} → {new_train_edges}"
        )
        logger.info(
            f"  ├─ 验证图 (Val)  : 边 {orig_val_edges} → {new_val_edges} (仅保留Mask节点相关边)"
        )
        logger.info(
            f"  └─ 测试图 (Test) : 边 {orig_test_edges} → {new_test_edges} (仅保留Mask节点相关边)"
        )

        if new_val_edges == 0:
            logger.info(
                "  ⚠️ 警告: 过滤后验证集边数为 0! 建议降低mask_ratio或检查数据集连通性。"
            )
        if new_test_edges == 0:
            logger.info(
                "  ⚠️ 警告: 过滤后测试集边数为 0! 建议降低mask_ratio或检查数据集连通性。"
            )
        return train_graph, val_graph_filtered, test_graph_filtered

    def _combine_train_val_graph(self) -> nx.MultiGraph:
        """
        合并训练集和验证集图（仅直推式任务使用），满足:
        1. 保留所有边属性（时间戳、权重等）
        2. 按时间戳排序合并（不打乱时序）
        3. 节点集合与训练集保持一致（避免数据泄露）
        :return: 合并后的训练+验证集图
        """
        # 初始化合并图，继承训练集的节点属性
        combined_graph = nx.MultiGraph()
        combined_graph.add_nodes_from(self.train_graph.nodes(data=True))

        # 提取训练集和验证集的所有边（保留属性）
        train_edges = list(self.train_graph.edges(data=True))
        val_edges = list(self.val_graph.edges(data=True))

        # 按时间戳排序（保证时序性）
        all_edges = train_edges + val_edges
        all_edges_sorted = sorted(all_edges, key=lambda x: x[2].get("time", 0))

        # 添加合并后的边
        combined_graph.add_edges_from(all_edges_sorted)

        return combined_graph

    def extract_interaction_seqs(
        self,
        task_type: str,
        graph: nx.MultiGraph | None = None,  # 兼容外部传入graph的场景
    ) -> dict[int, list[tuple[int, float]]]:
        """
        提取每个节点的交互序列（按时间倒序，去重邻居）
        :param task_type: 任务类型 "T" (Transductive/直推式) / "I" (Inductive/归纳式)
        :param graph: 可选：外部传入的图（优先级高于内置图）
        :return: 节点->[(邻居, 时间戳), ...] 的字典
        """
        if task_type not in ["T", "I"]:
            raise ValueError(f"task_type必须是'T'或'I'，当前为{task_type}")
        if graph is not None:
            # 外部传入的graph优先
            if not isinstance(graph, nx.MultiGraph):
                raise TypeError(
                    f"传入的graph必须是nx.MultiGraph类型, 当前为{type(graph)}"
                )
            target_graph = graph
            desc = "自定义图数据"
        else:
            if task_type == "T":
                target_graph = self._combine_train_val_graph()
                desc = "训练集 + 验证集合并图（直推式）"
            else:
                target_graph = self.train_graph
                desc = "训练集原始图（归纳式）"

        if target_graph is None:
            raise ValueError("target_graph初始化失败! 请检查task_type或传入的graph参数")
        if target_graph.number_of_nodes() == 0:
            raise ValueError("target_graph为空(无节点)，无法提取交互序列")
        if target_graph.number_of_edges() == 0:
            raise ValueError("target_graph为空(无边)，无法提取交互序列")

        # extract
        node_interaction_seq = {}
        for node in target_graph.nodes():
            neighbors_with_time = []

            for neighbor, edge_dict in target_graph[node].items():
                for edge_attr in edge_dict.values():
                    if self.time_attr not in edge_attr:
                        raise KeyError(
                            f"边({node},{neighbor})缺少'{self.time_attr}'属性"
                        )
                    neighbors_with_time.append((neighbor, edge_attr[self.time_attr]))

            # 按照时间倒序排列
            neighbors_with_time.sort(key=lambda x: x[1], reverse=True)

            # 去重方案1️⃣: 交互序列可以有重复的邻居，只要不相邻即可
            # unique_neighbors_with_time = []
            # for i in range(len(neighbors_with_time)):
            #     if i == 0 or neighbors_with_time[i][0] != neighbors_with_time[i-1][0]:
            #         unique_neighbors_with_time.append((neighbors_with_time[i]))

            # 去重方案2️⃣: 交互序列中无重复邻居，只保留最后一次交互
            unique_neighbors = {}
            for neighbor, timestamp in neighbors_with_time:
                if neighbor not in unique_neighbors:
                    unique_neighbors[neighbor] = timestamp
            unique_neighbors_with_time = [(n, t) for n, t in unique_neighbors.items()]
            unique_neighbors_with_time.sort(
                key=lambda x: x[1], reverse=True
            )  # 保证时间倒序

            node_interaction_seq[node] = unique_neighbors_with_time

        # 统计信息
        total_nodes = len(node_interaction_seq)
        avg_seq_len = (
            sum(len(seq) for seq in node_interaction_seq.values()) / total_nodes
            if total_nodes > 0
            else 0
        )
        logger.info(f"✅ 交互序列提取完成")
        logger.info(f"   ├─ 数据源: {desc}")
        logger.info(f"   ├─ 处理节点数: {total_nodes}")
        logger.info(f"   └─ 平均序列长度: {avg_seq_len:.2f}")
        return node_interaction_seq, round(avg_seq_len)

    def extract_ctx_window(
        self,
        task_type: str,
        walk_num: int,
        walk_len: int,
        p: float = 1.0,
        q: float = 1.0,
        workers: int = 1,
        quiet: bool = False,
        graph: nx.MultiGraph | None = None,  # 兼容外部传入graph的场景
    ) -> dict:
        """
        并行提取上下文序列（双向随机游走）
        :param task_type: 任务类型 "T" (Transductive/直推式) / "I" (Inductive/归纳式)
        :param walk_length: 游走长度
        :param num_walks: 每个节点的游走次数
        :param p: node2vec返回参数 (default: 1.0)
        :param q: node2vec进出参数 (default: 1.0)
        :param workers: 并行进程数 (default: 1)
        :param quiet: 是否静默模式（关闭进度条） (default: False)
        :param graph: 可选：外部传入的图（优先级高于内置图） (default: None)
        :return: (节点->上下文序列字典, 仅节点的上下文列表)
        """
        CONST_KEYS = {
            "first_travel": "first_travel_key",
            "probabilities": "probabilities_key",
            "neighbors": "neighbors_key",
            "neighbors_time": "neighbors_time_key",
            "weight": "sci",
        }
        # check
        if task_type not in ["T", "I"]:
            raise ValueError(f"task_type必须是'T'或'I'，当前为{task_type}")
        if walk_len <= 0 or walk_num <= 0:
            raise ValueError(
                f"游走长度/次数必须大于0! walk_length={walk_len}, num_walks={walk_num}"
            )

        if graph is not None:
            # 外部传入的graph优先
            if not isinstance(graph, nx.MultiGraph):
                raise TypeError(
                    f"传入的graph必须是nx.MultiGraph类型, 当前为{type(graph)}"
                )
            target_graph = graph
        else:
            if task_type == "T":
                # 直推式：使用训练+验证集合并图
                target_graph = self._combine_train_val_graph()
            else:
                # 归纳式：仅使用训练集图
                target_graph = self.train_graph

        if target_graph is None:
            raise ValueError("target_graph初始化失败! 请检查task_type或传入的graph参数")
        if target_graph.number_of_nodes() == 0:
            raise ValueError("target_graph为空(无节点)，无法提取上下文窗口")
        if target_graph.number_of_edges() == 0:
            raise ValueError("target_graph为空(无边)，无法提取上下文窗口")

        add_attr(target_graph)
        # Precomputes transition probabilities
        d_graph, max_time = self._precompute_transition_probs(
            graph=target_graph, p=p, q=0.5, const_keys=CONST_KEYS, quiet=quiet
        )

        # Split num_walks for each worker
        workers = min(workers, os.cpu_count() or 1)
        num_walks_chunks = np.array_split(range(walk_num), workers)

        # 1. 预先定义好任务列表
        tasks = (
            delayed(self._parallel_generate_context)(
                d_graph=d_graph,
                walk_length=walk_len,
                num_walks=len(chunk),
                const_keys=CONST_KEYS,
                use_linear=True,
                half_life=1.0,
            )
            for chunk in num_walks_chunks
        )
        results_gen = Parallel(n_jobs=workers, return_as="generator")(tasks)
        context_chunks = []

        with tqdm(
            total=walk_num,
            desc="🔄 提取节点上下文",
            unit="walk",
            disable=quiet,
        ) as pbar:
            for i, res in enumerate(results_gen):
                context_chunks.append(res)
                pbar.update(len(num_walks_chunks[i]))

        # Merge contexts
        context_dict = defaultdict(list)
        for chunk in context_chunks:
            for node, seqs in chunk.items():
                context_dict[node].extend(seqs)

        logger.info("✅ 上下文窗口提取完成")
        logger.info(
            f"   ├─ 游走配置: 次数: {walk_num} | 长度: {walk_len} | CPU进程数: {workers}"
        )
        total_contexts = sum(len(v) for v in context_dict.values())
        avg_seq_per_node = (
            total_contexts / len(context_dict) if len(context_dict) > 0 else 0
        )
        logger.info(
            f"   ├─ 提取结果: 覆盖节点数: {len(context_dict)} | 总序列数: {total_contexts}"
        )
        logger.info(f"   └─ 统计信息: 单节点平均序列数: {avg_seq_per_node:.2f}")
        return context_dict

    def _precompute_transition_probs(
        self,
        graph: nx.MultiGraph,
        p: float = 1.0,
        q: float = 1.0,
        const_keys: dict[str, str] = None,
        quiet: bool = False,
    ) -> tuple[dict, float]:
        """
        预计算每个节点的转移概率(node2vec策略)
        :param graph: 输入图
        :param p: 返回参数
        :param q: 进出参数
        :param const_keys: 常量键名字典
        :param quiet: 是否静默模式
        :return: (转移概率字典, 图的最大时间戳)
        """
        if const_keys is None:
            const_keys = {
                "first_travel": "first_travel_key",
                "probabilities": "probabilities_key",
                "neighbors": "neighbors_key",
                "neighbors_time": "neighbors_time_key",
                "weight": "sci",
            }

        d_graph = defaultdict(dict)
        # 获取网络最大时间戳
        try:
            max_time = max([attr["time"] for _, _, attr in graph.edges(data=True)])
        except ValueError:
            raise ValueError("图中无有效时间戳边！")

        first_travel_done = set()

        nodes_generator = (
            graph.nodes()
            if quiet
            else tqdm(graph.nodes(), desc="🔄 计算游走转移概率", leave=True)
        )

        node_act = nx.get_node_attributes(graph, "act")
        for source in nodes_generator:
            # Init probabilities dict for first travel
            self._init_node_dict(d_graph, source, const_keys)
            for current_node in graph.neighbors(source):
                # Init probabilities dict
                self._init_node_dict(d_graph, current_node, const_keys)

                unnormalized_weights = []
                first_travel_weights = []
                neighbors = []

                # Calculate unnormalized weights
                for dest in graph.neighbors(current_node):
                    edge_weight = graph[current_node][dest][0].get(
                        const_keys["weight"], 1.0
                    )
                    if dest == source:  # Backwards probability
                        ss_weight = edge_weight * node_act.get(dest, 1.0) / p
                    elif (
                        dest in graph[source]
                    ):  # If the neighbor is connected to the source
                        ss_weight = edge_weight * node_act.get(dest, 1.0)
                    else:
                        ss_weight = edge_weight * node_act.get(dest, 1.0) / q

                    # Assign the unnormalized sampling strategy weight, normalize during random walk
                    unnormalized_weights.append(ss_weight)
                    if current_node not in first_travel_done:
                        first_travel_weights.append(edge_weight)
                    neighbors.append(dest)

                # Normalize
                unnormalized_weights = np.array(unnormalized_weights)
                if unnormalized_weights.sum() == 0:
                    raise ValueError(f"节点{current_node}的转移权重和为0!")

                d_graph[current_node][const_keys["probabilities"]][source] = (
                    unnormalized_weights / unnormalized_weights.sum()
                )

                if current_node not in first_travel_done:
                    first_travel_weights = np.array(first_travel_weights)
                    d_graph[current_node][const_keys["first_travel"]] = (
                        first_travel_weights / first_travel_weights.sum()
                    )
                    first_travel_done.add(current_node)

                # Save neighbors and time_edges
                d_graph[current_node][const_keys["neighbors"]] = neighbors
                d_graph[current_node][const_keys["neighbors_time"]] = (
                    self._get_neighbor_time_map(graph, current_node, neighbors)
                )

        return d_graph, max_time

    def _parallel_generate_walks(
        self,
        d_graph: defaultdict[any, dict[str, any]],
        walk_length: int,
        num_walks: int,
        max_time: float,
        cpu_num: int,
        const_keys: dict[str, str] = None,
        quiet: bool = False,
        use_linear: bool = True,
        half_life: float = 1.0,
    ) -> defaultdict[any, list[list[tuple[any, float]]]]:
        """
        生成随机游走序列
        """
        if const_keys is None:
            const_keys = self._get_default_const_keys()

        walks = defaultdict(list)

        pbar = (
            None
            if quiet
            else tqdm(
                total=num_walks,
                desc=f"🔄 生成游走序列 (CPU {cpu_num})",
                unit="轮",
                leave=True,
            )
        )

        for _ in range(num_walks):
            if not quiet:
                pbar.update(1)

            # Shuffle
            nodes = list(d_graph.keys())
            random.shuffle(nodes)

            # Random walk
            for source in nodes:
                walk = [(source, 0.0)]
                last_time = 0.0
                while len(walk) < walk_length:
                    # For the first step
                    try:
                        if len(walk) == 1:
                            probs = d_graph[walk[-1][0]][const_keys["first_travel"]]
                        else:
                            probs = d_graph[walk[-1][0]][const_keys["probabilities"]][
                                walk[-2][0]
                            ]
                    except KeyError as e:
                        raise KeyError(f"节点{walk[-1][0]}的转移概率不存在！{e}")

                    walk_options = self._get_valid_walk_options(
                        d_graph,
                        walk[-1][0],
                        probs,
                        const_keys,
                        last_time,
                        filter_func=lambda t: t > last_time,
                    )
                    # Skip dead end nodes
                    if not walk_options:
                        break

                    if len(walk) == 1:
                        last_time = max(opt[2] for opt in walk_options)

                    final_probs = self._compute_final_probabilities(
                        walk_options, use_linear, half_life, last_time
                    )
                    next_idx = np.random.choice(len(walk_options), p=final_probs)
                    next_node, _, next_time = walk_options[next_idx]
                    last_time = next_time
                    walk.append((next_node, last_time))

                # walk = list(map(str, walk))  # Convert all to strings
                walks[source].append(walk)

        if not quiet:
            pbar.close()

        return walks

    def _parallel_generate_context(
        self,
        d_graph: defaultdict[any, dict[str, any]],
        walk_length: int,
        num_walks: int,
        const_keys: dict[str, str] = None,
        use_linear: bool = True,
        half_life: float = 1.0,
        global_progress: any = None,
        progress_lock: any = None,
    ) -> defaultdict[any, list[list[tuple[any, float]]]]:
        """
        生成上下文序列（双向游走）
        """
        if const_keys is None:
            const_keys = self._get_default_const_keys()

        contexts = defaultdict(list)
        half_len = (walk_length - 1) // 2 + 1

        for _ in range(num_walks):
            # Shuffle the nodes
            nodes = list(d_graph.keys())
            random.shuffle(nodes)

            # Start a random walk from every node
            for source in nodes:
                # Walk forward (half_len)
                walk = [(source, 0.0)]
                last_time = 0.0

                while len(walk) < half_len:
                    try:
                        probs = (
                            d_graph[walk[-1][0]][const_keys["first_travel"]]
                            if len(walk) == 1
                            else d_graph[walk[-1][0]][const_keys["probabilities"]][
                                walk[-2][0]
                            ]
                        )
                    except KeyError as e:
                        raise KeyError(f"节点{walk[-1][0]}的转移概率不存在！{e}")

                    walk_options = self._get_valid_walk_options(
                        d_graph,
                        walk[-1][0],
                        probs,
                        const_keys,
                        last_time,
                        filter_func=lambda t: t > last_time,
                    )
                    if not walk_options:
                        break

                    if len(walk) == 1:
                        last_time = max(opt[2] for opt in walk_options)

                    final_probs = self._compute_final_probabilities(
                        walk_options, use_linear, half_life, last_time
                    )
                    next_idx = np.random.choice(len(walk_options), p=final_probs)
                    next_node, _, next_time = walk_options[next_idx]

                    last_time = next_time
                    walk.append((next_node, last_time))

                # Walk backward (half_len)
                if len(walk) < 2:
                    continue

                last_time = walk[1][1]
                walk.pop(0)
                walk.insert(0, (source, last_time))

                while len(walk) < walk_length:
                    try:
                        probs = (
                            d_graph[walk[0][0]][const_keys["first_travel"]]
                            if len(walk) == 1
                            else d_graph[walk[0][0]][const_keys["probabilities"]][
                                walk[1][0]
                            ]
                        )
                    except KeyError as e:
                        raise KeyError(f"节点{walk[0][0]}的转移概率不存在！{e}")

                    walk_options = self._get_valid_walk_options(
                        d_graph,
                        walk[0][0],
                        probs,
                        const_keys,
                        last_time,
                        filter_func=lambda t: t < last_time,
                    )

                    if not walk_options:
                        break

                    final_probs = self._compute_final_probabilities(
                        walk_options, use_linear, half_life, last_time
                    )
                    next_idx = np.random.choice(len(walk_options), p=final_probs)
                    next_node, _, next_time = walk_options[next_idx]

                    last_time = next_time
                    walk.insert(0, (next_node, last_time))

                # padding
                # while len(walk) < walk_length:
                #     walk.insert(0, (0, 0))
                contexts[source].append(walk)

                if global_progress is not None and progress_lock is not None:
                    with progress_lock:
                        global_progress.value += 1

        return contexts

    def _init_node_dict(
        self,
        d_graph: defaultdict[any, dict[str, any]],
        node: any,
        const_keys: dict[str, str],
    ) -> None:
        """初始化节点的字典结构"""
        if const_keys["probabilities"] not in d_graph[node]:
            d_graph[node][const_keys["probabilities"]] = dict()
        if const_keys["first_travel"] not in d_graph[node]:
            d_graph[node][const_keys["first_travel"]] = dict()
        if const_keys["neighbors"] not in d_graph[node]:
            d_graph[node][const_keys["neighbors"]] = dict()
        if const_keys["neighbors_time"] not in d_graph[node]:
            d_graph[node][const_keys["neighbors_time"]] = dict()

    def _get_neighbor_time_map(
        self, graph: nx.MultiGraph, node: any, neighbors: list[any]
    ) -> dict[any, list[float]]:
        """获取节点邻居的时间戳映射"""
        neighbor_time_map = defaultdict(list)
        for neighbor in neighbors:
            edge_attrs = graph[node][neighbor]
            for attr in edge_attrs.values():
                if "time" not in attr:
                    raise ValueError(f"edge ({node}-{neighbor}) no time attribute!")
                neighbor_time_map[neighbor].append(attr["time"])
        return neighbor_time_map

    def _get_valid_walk_options(
        self,
        d_graph: defaultdict[any, dict[str, any]],
        current_node: any,
        probs: np.ndarray,
        const_keys: dict[str, str],
        last_time: float,
        filter_func: callable,
    ) -> list[tuple[any, float, float]]:
        """筛选有效的游走选项"""
        walk_options = []
        neighbors = d_graph[current_node][const_keys["neighbors"]]
        neighbor_times = d_graph[current_node][const_keys["neighbors_time"]]

        for neighbor, p in zip(neighbors, probs):
            times = neighbor_times[neighbor]
            walk_options.extend([(neighbor, p, t) for t in times if filter_func(t)])

        return walk_options

    def _compute_final_probabilities(
        self,
        walk_options: list[tuple[any, float, float]],
        use_linear: bool = True,
        half_life: float = 1.0,
        last_time: float = 0.0,
    ) -> np.ndarray:
        """计算最终的游走选择概率"""
        if use_linear:
            # 线性时间权重
            times = np.array([opt[2] for opt in walk_options])
            time_ranks = np.argsort(np.argsort(times)[::-1]) + 1  # 倒序排名
            probs = np.array([opt[1] for opt in walk_options])
            final_probs = time_ranks * probs
        else:
            # 指数时间权重
            final_probs = np.array(
                [
                    np.exp(opt[1] * (last_time - opt[2]) / half_life)
                    for opt in walk_options
                ]
            )

        # 归一化概率
        final_probs = final_probs / final_probs.sum()
        return final_probs

    def _get_default_const_keys(self):
        """获取默认的常量键名"""
        return {
            "first_travel": "first_travel_key",
            "probabilities": "probabilities_key",
            "neighbors": "neighbors_key",
            "neighbors_time": "neighbors_time_key",
            "weight": "sci",
        }

    def estimate_walk_params(
        self,
        graph: nx.MultiGraph | None = None,
        task_type: str = "T",
        sample_ratio: float = 0.1,  # 替换原sample_size
        sample_min: int = 20,
        sample_max: int = 200,
        quiet: bool = False,
    ) -> tuple[int, int]:
        """
        基于网络特征自动估算随机游走的最优参数(WALK_NUM, WALK_LEN)
        :param graph: 可选：外部传入的图（优先级高于内置图）
        :param task_type: 任务类型 "T"(Transductive/直推式)/"I"(Inductive/归纳式)
        :param sample_size: 计算平均路径长度的采样节点数(默认100, 节点少则取全部)
        :param quiet: 是否静默模式(关闭日志输出), 默认False
        :return: 估算的游走次数(WALK_NUM)和长度(WALK_LEN)
        """
        if task_type not in ["T", "I"]:
            raise ValueError(f"task_type必须为'T'或'I'，当前值: {task_type}")

        if graph is not None:
            if not isinstance(graph, nx.MultiGraph):
                raise TypeError(
                    f"传入的graph必须是nx.MultiGraph类型, 当前为{type(graph)}"
                )
            target_graph = graph
        else:
            if task_type == "T":
                target_graph = self._combine_train_val_graph()
            else:
                target_graph = self.train_graph

        # 核心统计特征
        stats = self._calculate_graph_stats(
            target_graph,
            sample_ratio=sample_ratio,
            sample_min=sample_min,
            sample_max=sample_max,
        )
        num_nodes = stats["num_nodes"]
        total_edges = stats["total_edges"]
        avg_degree = stats["avg_degree"]
        density = stats["density"]
        avg_interact_freq = stats["avg_interact_freq"]
        avg_path_len = round(stats["avg_path_len"])
        # diameter = stats["diameter"]

        # ========== 游走次数估算 ==========
        # 密度阈值（经验值）：极稀疏<0.01，中等稀疏0.01~0.1，稠密>0.1
        DENSITY_THRESHOLD_1 = 0.01
        DENSITY_THRESHOLD_2 = 0.1
        # 基础游走次数（按网络规模+密度划分）
        if density < DENSITY_THRESHOLD_1:  # 极稀疏网络（如引文/社交网络）
            base_walk_num = 50 if num_nodes < 1000 else 30 if num_nodes < 10000 else 15
        elif density < DENSITY_THRESHOLD_2:  # 中等稀疏网络（如知识图谱）
            base_walk_num = 30 if num_nodes < 1000 else 20 if num_nodes < 10000 else 10
        else:  # 稠密网络（如通信/交易网络）
            base_walk_num = 20 if num_nodes < 1000 else 10 if num_nodes < 10000 else 5

        # 交互频率修正：频率越高，游走次数可越少（最多减少50%）
        freq_correction = min(
            max(avg_interact_freq / 2, 0.5), 2
        )  # 修正系数（最多减少一半或者翻一倍）
        walk_num = max(round(base_walk_num / freq_correction), 5)  # 兜底：至少5次

        # ========== 游走长度估算 ==========
        min_walk_len = 11  # 最小11步（太短学不到拓扑）
        max_walk_len = 20  # 最大20步（超长游走引入噪声，且慢）
        walk_len = max(min_walk_len, min(avg_path_len * 2, max_walk_len))

        if not quiet:
            density_desc = (
                "极稀疏"
                if density < DENSITY_THRESHOLD_1
                else "中等稀疏"
                if density < DENSITY_THRESHOLD_2
                else "稠密"
            )
            logger.info(f"📊 游走参数自动估算完成（基于网络特征）")
            logger.info(f"   ├─ 网络基础特征:")
            logger.info(
                f"   │  - 节点数: {num_nodes:,} | 总边数(含重复): {total_edges:,} | 唯一边数: {stats['num_unique_edges']:,}"
            )
            logger.info(
                f"   │  - 平均度: {avg_degree:.1f} | 平均交互频率: {avg_interact_freq:.1f} | 平均路径长度: {avg_path_len:.1f}"
            )
            logger.info(f"   │  - 网络密度: {density:.4f} ({density_desc})")
            logger.info(f"   └─ 推荐游走参数:")
            logger.info(f"      - WALK_NUM (每节点游走次数): {walk_num}")
            logger.info(f"      - WALK_LEN (单次游走长度): {walk_len}")

        return walk_num, walk_len

    def _calculate_graph_stats(
        self,
        graph: nx.MultiGraph,
        sample_ratio: float = 0.2,
        sample_min: int = 20,
        sample_max: int = 500,
    ) -> dict:
        """
        计算图的核心统计特征（供参数估算使用）
        :param graph: 输入图
        :param sample_size: 采样节点数（计算平均路径长度）
        :return: 统计特征字典
        """
        if graph.number_of_nodes() == 0:
            raise ValueError("图为空（无节点），无法计算统计特征")

        num_nodes = graph.number_of_nodes()
        total_edges = graph.number_of_edges()  # 原始总边数（含重复边）

        # 仅保留(u, v)对，忽略边属性和key
        unique_edges = set((u, v) for u, v, _ in graph.edges(keys=True))
        num_unique_edges = len(unique_edges)

        # 平均度（基于唯一边，反映真实连接密度）
        avg_degree = (2 * num_unique_edges) / num_nodes if num_nodes > 0 else 1.0
        # 平均交互频率（总边数/唯一边数，反映节点对的交互频繁度）
        avg_interact_freq = (
            total_edges / num_unique_edges if num_unique_edges > 0 else 1.0
        )

        # 网络密度（基于唯一边）
        if num_nodes < 2:
            density = 0.0
        else:
            density = (2 * num_unique_edges) / (num_nodes * (num_nodes - 1))

        # ========== 平均最短路径长度 ==========
        avg_path_len = 0.0
        valid_samples = 0
        nodes = list(graph.nodes())

        # 按比例采样 + 上下限兜底
        sample_count = int(num_nodes * sample_ratio)
        sample_count = max(sample_min, min(sample_count, sample_max))
        sample_count = min(sample_count, num_nodes)
        sample_nodes = random.sample(nodes, sample_count)
        for node in sample_nodes:
            try:
                # 计算单个节点到所有可达节点的最短路径长度
                path_lengths = nx.single_source_shortest_path_length(graph, node)
                if len(path_lengths) > 1:  # 排除仅自身的情况
                    avg_path_len += sum(path_lengths.values()) / (len(path_lengths) - 1)
                    valid_samples += 1
            except Exception as e:
                warnings.warn(f"计算节点{node}的最短路径失败: {str(e)}", UserWarning)
                continue

        # 兜底：无有效采样时默认5
        avg_path_len = avg_path_len / valid_samples if valid_samples > 0 else 5.0

        # 直径计算(可能非常慢，尤其是大图)，这里我们选择不计算或使用平均路径长度的倍数作为近似
        # diameter = 0
        # components = list(nx.connected_components(graph))
        # for comp in components:
        #     subG = graph.subgraph(comp)
        #     try:
        #         comp_diameter = nx.diameter(subG)
        #         diameter = max(diameter, comp_diameter)
        #     except:
        #         continue
        # # 兜底
        # diameter = diameter if diameter > 0 else int(avg_path_len * 2)

        return {
            "num_nodes": num_nodes,
            "total_edges": total_edges,  # 原始总边数（含重复）
            "num_unique_edges": num_unique_edges,  # 唯一边数（去重）
            "avg_degree": avg_degree,  # 真实平均度（基于唯一边）
            "avg_interact_freq": avg_interact_freq,  # 平均交互频率
            "density": density,  # 真实网络密度（基于唯一边）
            "avg_path_len": avg_path_len,
            # "diameter": diameter              # 图直径
        }
