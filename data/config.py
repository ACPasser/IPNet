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

# 训练默认配置
DEFAULT_TRAIN_CONFIG = {
    # 基础配置
    "SEED": 19,
    # 是否预处理
    "PRE_PROCESS": False,
    # 数据集与任务配置
    "DATASET": "UCI",
    "TASK_TYPE": "T",  # T: Transductive, I: Inductive
    "MASK_RATIO": 0.1,
    # 训练配置
    "EPOCH": 50,
    "BATCH_SIZE": 64,
    "LR": 1e-4,
    "THREAD_NUM": 5,
    "DEVICE": -1,  # -2:CPU, -1:MPS, ≥0:GPU
    # 核心超参
    "IS_LEN": -1,  # -1: auto
    "WALK_NUM": -1,  # -1: auto
    "WALK_LEN": -1,  # -1: auto
    # 路径配置
    "CHECKPOINT_PATH": "outputs/{dataset}/saved_checkpoints/{timestamp}/checkpoint-epoch-{epoch}.pth",
    "BEST_MODEL_PATH": "outputs/{dataset}/best_models/{timestamp}/best-model.pth",
    "RESULT_PATH": "outputs/{dataset}/results/{core_params}/IPNet-{version}.csv",  # core_params是动态生成的参数组合
}

# 模型默认参数(注意不能和DEFAULT_TRAIN_CONFIG中的配置重名)
DEFAULT_MODEL_CONFIG = {
    # 命令行参数
    "VERSION": "mean",  # mean/att/w2v
    "FEAT_DIM": 128,
    "RNN_TYPE": "GRU",  # LSTM/GRU
    "PADDING_NODE": 0,
    "N_HEAD": 8,
    "DROPOUT": 0.3,
    # 其他参数
    "FINAL_SEQ_LEN": None,
    "FINAL_WALK_LEN": None,
    # 参数保存地址, 用于后续加载
    "PARAM_SAVE_CONFIG": {
        "dir": "outputs/{dataset}/model_param/{timestamp}",
        "node_feature": "node_feature.npy",
        "interactions": "interactions.json",
        "contexts": "contexts.json",
        "other_params": "other_params.json",
    },
}


def get_data_config(dataset: str):
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
