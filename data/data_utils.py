import hashlib
import logging
import os
import re
import pandas as pd
import random
import networkx as nx
from datetime import datetime


logger = logging.getLogger(__name__)


def trans_id(nid: any) -> str:
    """
    节点ID匿名化
    Args:
        nid: 原始节点ID(任意类型: 数字、字符串、用户名、邮箱等)
    Returns:
        str: MD5哈希后的前16位
    """
    original_id_str = str(nid).strip()

    # MD5哈希（128位，32个十六进制字符）
    md5_hash = hashlib.md5(original_id_str.encode("utf-8"))
    hash_hex = md5_hash.hexdigest()

    # 截取前16位
    anon_id = f"U{hash_hex[:16]}"

    return anon_id


def split_snap_by_month(
    df: pd.DataFrame,
    output_snap_dir: str,
    time_col: str = "time",
    date_format: str = "%Y-%m",
    sep: str = "\t",
) -> None:
    """
    按月份切割数据生成快照文件
    Args:
        df: 原始数据框
        output_snap_dir: 快照输出目录
        time_col: 时间戳列名（秒级时间戳）
        date_format: 日期格式化字符串
        sep: 输出文件分隔符
    """
    # 参数校验
    if time_col not in df.columns:
        raise ValueError(f"时间列 '{time_col}' 不存在！当前列: {df.columns.tolist()}")
    if len(df) == 0:
        raise ValueError("输入数据框为空，无法切分快照！")

    # 时间转换
    try:
        df["date"] = df[time_col].apply(
            lambda x: datetime.fromtimestamp(float(x)).strftime(date_format)
        )
    except Exception as e:
        raise RuntimeError(f"时间戳解析失败: {str(e)}")

    os.makedirs(output_snap_dir, exist_ok=True)

    # 按月份切分快照
    month_groups = df.groupby("date")
    if len(month_groups) == 0:
        raise ValueError("未找到有效月份分组！")

    for idx, (month, df_month) in enumerate(month_groups, 1):
        df_output = df_month.drop("date", axis=1)
        output_file = os.path.join(output_snap_dir, f"{month}.csv")
        df_output.to_csv(output_file, sep=sep, index=False)
        logger.info(
            f"✅ 按月份切割快照 {idx}/{len(month_groups)}: {month}.csv (行数: {len(df_output)})"
        )


def split_snap_by_uniform(
    df, output_snap_dir, train_ratio=0.5, snapshots_num=5, sep="\t"
):
    """
    截取训练集，并均匀分割为指定数量的快照文件

    参数说明:
    df: pandas.DataFrame - 原始数据框
    output_snap_dir: str - 快照文件输出目录
    train_ratio: float - 训练数据占比(0 < train_ratio ≤ 1), 默认0.5(50%)
    snapshots_num: int - 快照文件数量(≥1), 默认5
    sep: str - 输出文件分隔符(默认'\t')

    返回值:
    None - 直接将快照文件写入指定目录
    """
    # 参数校验
    if not isinstance(train_ratio, float) or train_ratio <= 0 or train_ratio > 1:
        raise ValueError(f"train_ratio 必须是0到1之间的浮点数, 当前值: {train_ratio}")

    if not isinstance(snapshots_num, int) or snapshots_num < 1:
        raise ValueError(f"snapshots_num 必须是≥1的整数, 当前值: {snapshots_num}")

    # 截取训练集
    train_end_idx = int(len(df) * train_ratio)
    df_train = df.iloc[0:train_end_idx, :]

    if len(df_train) == 0:
        raise ValueError(
            f"训练集数据为空！总行数: {len(df)}, 训练集比例: {train_ratio}"
        )
    os.makedirs(output_snap_dir, exist_ok=True)

    # 均匀切分快照
    chunk_size = len(df_train) // snapshots_num
    if chunk_size == 0:
        chunk_size = 1
        snapshots_num = len(df_train)
        logger.info(
            f"⚠️ 警告: 快照数量({snapshots_num}) > 训练集行数({len(df_train)})，已调整为每行一个快照"
        )
    for i in range(snapshots_num):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < snapshots_num - 1 else len(df_train)
        sub_df = df_train.iloc[start:end]

        output_file_path = os.path.join(output_snap_dir, f"train_snap_{i + 1}.csv")
        sub_df.to_csv(output_file_path, sep=sep, index=False)
        logger.info(
            f"✅ 训练集快照 {i + 1}/{snapshots_num} 生成完成 (行数: {len(sub_df)})"
        )


def normalize_whitespace(
    input_data: str | list[str] | pd.Series, keep_empty_lines: bool = False
) -> str | list[str] | pd.Series:
    """
    统一归一化空白字符: 将任意空白字符(空格、\t、\n、多个连续空格)替换为单个空格
    Args:
        input_data: 输入数据(字符串/字符串列表/pd.Series)
        keep_empty_lines: 是否保留空行(仅对列表/Series生效), 默认False(过滤空行)
    Returns:
        归一化后的结果（与输入类型一致）
    """
    # 正则表达式: 匹配任意空白字符（空格、\t、\n、\r等）
    whitespace_pattern = re.compile(r"\s+")

    def _normalize_single_line(line: str) -> str:
        """处理单行字符串"""
        # 1. 替换所有空白字符为单个空格
        normalized = whitespace_pattern.sub(" ", line.strip())
        return normalized.strip()

    if isinstance(input_data, str):
        # 场景1: 字符串
        return _normalize_single_line(input_data)

    elif isinstance(input_data, list):
        # 场景2: 字符串列表（比如文件行列表）
        normalized_lines = []
        for line in input_data:
            normalized_line = _normalize_single_line(str(line))
            # 过滤空行（如果需要）
            if keep_empty_lines or normalized_line:
                normalized_lines.append(normalized_line)
        return normalized_lines

    elif isinstance(input_data, pd.Series):
        # 场景3: pandas Series（比如DataFrame的某一列）
        normalized_series = input_data.astype(str).apply(_normalize_single_line)
        if not keep_empty_lines:
            normalized_series = normalized_series[normalized_series != ""]
        return normalized_series

    else:
        raise TypeError(
            f"不支持的输入类型: {type(input_data)}, 仅支持str/list/pd.Series"
        )


def read_file_norm_ws(
    file_path: str,
    skip_rows: int = 0,
    col_names: list[str] | None = None,
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """
    读取文件并标准化空白字符, 返回DataFrame(适配任意空白分隔符[空格/\t/多个空格混合])
    Args:
        file_path: 文件路径
        skip_rows: 跳过前N行
        col_names: 列名列表
        encoding: 文件编码
    Returns:
        解析后的DataFrame
    """
    with open(file_path, "r", encoding=encoding) as f:
        lines = f.readlines()

    if skip_rows > 0:
        lines = lines[skip_rows:]

    normalized_lines = normalize_whitespace(lines, keep_empty_lines=False)

    parsed_data = [line.split(" ") for line in normalized_lines]

    # 校验字段数
    if col_names:
        expected_cols = len(col_names)
        for idx, row in enumerate(parsed_data):
            if len(row) != expected_cols:
                raise ValueError(
                    f"第{skip_rows + idx + 1}行字段数不匹配: "
                    f"实际{len(row)}个，期望{expected_cols}个"
                )
    return pd.DataFrame(parsed_data, columns=col_names)


def save_nodes_mapping(
    nodes_iterable, output_path: str, sep: str = ",", hint: str = "全部节点"
):
    """
    保存节点映射
    :param nodes_iterable: 任何包含节点ID的可迭代对象[list, set, pd.Series等]
    :param output_path: 映射文件保存路径
    :param sep: 分隔符
    """
    try:
        # 1. 去重并排序
        unique_nodes = sorted(list(set(nodes_iterable)))

        # 2. 构建映射
        mapping_data = [
            {
                "original_id": r_id,
                "anon_id": trans_id(r_id),
                "numeric_id": i + 1,  # 节点ID映射（从1开始编号，0保留给padding）
            }
            for i, r_id in enumerate(unique_nodes)
        ]

        mapping_df = pd.DataFrame(mapping_data)

        # 3. 保存
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        mapping_df.to_csv(output_path, sep=sep, index=False)

        logger.info(
            f"✅ 节点映射表({hint})已保存: {output_path} (节点总数: {len(mapping_df)})"
        )
        return mapping_df

    except Exception as e:
        logger.error(f"❌ 建立节点映射失败: {str(e)}")
        raise RuntimeError(f"Node mapping failed: {e}")


def build_nx_graph(data_config: dict):
    """
    根据配置构建 NetworkX MultiGraph 对象
    核心逻辑: 读取配置指定的图文件/节点集文件 → 节点ID映射 → 构建带时间属性的无向多重图

    Args:
        data_config: 数据集预处理配置字典(需包含以下字段)
            - csv_sep: 文件分隔符
            - output_graph_path: 图数据文件路径(预处理后生成)
            - output_nodes_mapping_path: 节点映射文件路径(预处理后生成)
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
        "csv_sep",
        "output_graph_path",  # 图数据
        "output_nodes_mapping_path",  # 节点映射文件
        "node_cols",
        "time_col",
    ]
    missing_keys = [key for key in required_config_keys if key not in data_config]
    if missing_keys:
        raise KeyError(f"构建NX图缺少必要配置字段: {missing_keys}")

    csv_sep = data_config["csv_sep"]
    graph_path = data_config["output_graph_path"]
    mapping_path = data_config["output_nodes_mapping_path"]
    src_col, tgt_col = data_config["node_cols"]
    time_col = data_config["time_col"]

    # fmt: off
    if not os.path.exists(graph_path):
        logger.error(f"图数据文件不存在(配置字段: output_graph_path): {graph_path}, 请检查是否进行过预处理!")
        raise FileNotFoundError(f"Graph file not found (config: output_graph_path): {graph_path}")

    if not os.path.exists(mapping_path):
        logger.error(f"节点集文件不存在(配置字段: output_nodes_mapping_path): {mapping_path}, 请检查是否进行过预处理!")
        raise FileNotFoundError(f"Nodes file not found (config: output_nodes_mapping_path): {mapping_path}")

    # ===================== 1. 读取节点映射文件 =====================
    mapping_df = pd.read_csv(mapping_path, sep=data_config["csv_sep"])
    anon2id = dict(zip(mapping_df["anon_id"], mapping_df["numeric_id"]))

    # ===================== 2. 读取图数据并转换 =====================
    df = pd.read_csv(graph_path, sep=csv_sep)
    df[src_col] = df[src_col].map(anon2id)
    df[tgt_col] = df[tgt_col].map(anon2id)

    if df[src_col].isnull().any() or df[tgt_col].isnull().any():
        logger.error("检测到部分匿名 ID 在映射表中不存在，请检查预处理的一致性！")
        raise ValueError("ID mapping inconsistency detected.")

    # 转回整数类型 (防止 map 产生 float)
    df[src_col] = df[src_col].astype(int)
    df[tgt_col] = df[tgt_col].astype(int)

    # ===================== 3. 构建NetworkX图并移除自环 =====================
    graph = nx.from_pandas_edgelist(
        df,
        source=src_col,
        target=tgt_col,
        edge_attr=time_col,
        create_using=nx.MultiGraph,
    )

    # 移除自环边
    self_loop_edges = list(nx.selfloop_edges(graph))
    if self_loop_edges:
        logger.warning(f"检测到 {len(self_loop_edges)} 个自环边，已自动移除")
        graph.remove_edges_from(self_loop_edges)

    # 日志输出最终结果
    logger.info("✅ NX图构建完成:")
    logger.info(f"   ├─ 图加载路径: {graph_path}")
    logger.info(f"   ├─ 映射表路径: {mapping_path}(节点总数: {len(mapping_df)})")
    logger.info(f"   └─ 节点数: {graph.number_of_nodes()} | 边数: {graph.number_of_edges()}")

    # fmt: on
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

    # 尝试: 根据节点的度进行加权采样（💥 别用，效果很差）
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
