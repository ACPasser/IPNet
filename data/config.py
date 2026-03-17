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
        # 快照相关
        "need_cut_snap": False,  # 是否划分
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
        # 快照相关
        "need_cut_snap": False,  # 是否划分
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
        # 快照相关
        "need_cut_snap": False,  # 是否划分
        "output_snap_dir": "data/ZhiHu/1.snapshots",  # 快照保存目录
        "date_format": "%Y-%m",  # 日期格式化
        "train_ratio": 0.5,  # 训练集比例
        "snapshots_num": 5,  # 训练集快照数量
    },
}

# 模型的默认配置
MODEL_DEFAULT_CONFIG = {
    # 基础配置
    "SEED": 19,
    # 是否预处理
    "PRE_PROCESS": False,
    # 数据集与任务配置
    "DATASET": "UCI",
    "TASK_TYPE": "T",  # T: Transductive, I: Inductive
    "MASK_RATIO": 0.1,
    # 普通参数
    "VERSION": "mean",  # mean/att/w2v
    "FEAT_DIM": 128,
    "RNN_TYPE": "GRU",  # LSTM/GRU
    "PADDING_NODE": 0,
    # 核心超参
    "IS_LEN": -1,  # -1: auto
    "WALK_NUM": -1,  # -1: auto
    "WALK_LEN": -1,  # -1: auto
    # 训练配置
    "EPOCH": 50,
    "BATCH_SIZE": 64,
    "LR": 1e-4,
    "THREAD_NUM": 5,
    "DEVICE": -1,  # -2:CPU, -1:MPS, ≥0:GPU
    # 路径配置
    "CHECKPOINT_DIR": "training_output/{dataset}/saved_checkpoints/{timestamp}",
    "BEST_MODEL_PATH": "training_output/{dataset}/best_models/{timestamp}/best-model.pth",
    "RESULT_PATH": "training_output/{dataset}/results/{core_params}/IPNet-{version}.csv",  # core_params是动态生成的参数组合
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
