# 数据集预处理配置
DATA_CONFIGS = {
    "uci": {
        "input_file_path": "data/UCI/0.origin/graph.txt",
        "skip_rows": 2,  # 跳过行数
        "col_names": ["src_node", "tgt_node", "weight", "time"],  # 列名映射
        "node_cols": ["src_node", "tgt_node"],  # 节点（源/目标）列名
        "time_col": "time",  # 时间戳列名
        "csv_sep": "\t",  # 文件分隔符
        "output_graph_path": "data/UCI/0.origin/graph.csv",  # csv格式图数据保存地址
        "output_node_path": "data/UCI/1.nodes_set/nodes.csv",  # csv格式节点集保存地址
        "need_cut_snap": False,  # 是否划分快照
        "output_snap_dir": "data/UCI/1.snapshots",  # 快照保存目录
        "date_format": "%Y-%m",  # 日期格式化
        "train_ratio": 0.5,  # 训练集比例
        "snapshots_num": 5,  # 训练集快照数量
    },
    "ia": {
        "input_file_path": "data/IA/0.origin/graph.txt",
        "skip_rows": 0,  # 跳过行数
        "col_names": ["src_node", "tgt_node", "weight", "time"],  # 列名映射
        "node_cols": ["src_node", "tgt_node"],  # 节点（源/目标）列名
        "time_col": "time",  # 时间戳列名
        "csv_sep": "\t",  # 文件分隔符
        "output_graph_path": "data/IA/0.origin/graph.csv",  # csv格式图数据保存地址
        "output_node_path": "data/IA/1.nodes_set/nodes.csv",  # csv格式节点集保存地址
        "need_cut_snap": False,  # 是否划分快照
        "output_snap_dir": "data/IA/1.snapshots",  # 快照保存目录
        "date_format": "%Y-%m",  # 日期格式化
        "train_ratio": 0.5,  # 训练集比例
        "snapshots_num": 5,  # 训练集快照数量
    },
    "zhihu": {
        "input_file_path": "data/ZhiHu/0.origin/graph.txt",
        "skip_rows": 1,  # 跳过行数
        "col_names": ["src_node", "tgt_node", "time"],  # 列名映射
        "node_cols": ["src_node", "tgt_node"],  # 节点（源/目标）列名
        "time_col": "time",  # 时间戳列名
        "csv_sep": "\t",  # 文件分隔符
        "output_graph_path": "data/ZhiHu/0.origin/graph.csv",  # csv格式图数据保存地址
        "output_node_path": "data/ZhiHu/1.nodes_set/nodes.csv",  # csv格式节点集保存地址
        "need_cut_snap": False,  # 是否划分快照
        "output_snap_dir": "data/ZhiHu/1.snapshots",  # 快照保存目录
        "date_format": "%Y-%m",  # 日期格式化
        "train_ratio": 0.5,  # 训练集比例
        "snapshots_num": 5,  # 训练集快照数量
    },
}


def get_config(dataset: str):
    """
    获取指定数据集的预处理配置(忽略大小写)
    :param dataset_name: 数据集名称
    :return: 该数据集的配置字典
    """
    ignore_case = dataset.strip().lower()

    if ignore_case not in DATA_CONFIGS:
        raise KeyError(
            f"数据集 {dataset} 的预处理配置不存在! 支持的数据集(不区分大小写):{list(DATA_CONFIGS.keys())}"
        )
    return DATA_CONFIGS[ignore_case]
