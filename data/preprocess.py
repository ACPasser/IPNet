import logging
import os
import pandas as pd

from data.config import get_config
from data.data_utils import read_file_norm_ws, trans_id

logger = logging.getLogger(__name__)


def preprocess(config=None, dataset_name=None):
    """
    通用数据集预处理函数
    :param config: 自定义配置字典(优先级高于dataset_name)
    :param dataset_name: 数据集名称(默认从config.py中读取配置)
    """
    # 优先级：自定义config > 从配置文件读取
    if config is None:
        if dataset_name is None:
            raise ValueError("至少传入 config 或 dataset_name 一种")
        config = get_config(dataset_name)

    # 1. 读取原始数据
    try:
        df = read_file_norm_ws(
            file_path=config["input_file_path"],
            skip_rows=config["skip_rows"],
            col_names=config["col_names"],
        )
        logger.info(f"✅ 数据集加载成功: {config['input_file_path']} (行数: {len(df)})")

        # 匿名化处理
        source_col, target_col = config["node_cols"]
        df[[source_col, target_col]] = df[[source_col, target_col]].apply(
            lambda x: x.map(trans_id)
        )

        # 保存csv格式图数据
        output_graph_dir = os.path.dirname(config["output_graph_path"])
        if output_graph_dir:
            os.makedirs(output_graph_dir, exist_ok=True)
        df.to_csv(config["output_graph_path"], sep=config["csv_sep"], index=False)
        logger.info(f"✅ 预处理完成, 数据集文件保存至: {config['output_graph_path']}")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"读取原始数据失败：文件不存在 - {str(e)}") from e
    except Exception as e:
        raise RuntimeError(f"处理原始数据失败：{str(e)}") from e

    # 2. 切割快照（可选）
    if config["need_cut_snap"]:
        try:
            from model.model_utils import split_snap_by_month, split_snap_by_uniform

            # 方式一：按月切割快照（全量数据）
            split_snap_by_month(
                df=df,
                output_snap_dir=config["output_snap_dir"],
                time_col=config["time_col"],
                date_format=config["date_format"],
                sep=config["csv_sep"],
            )

            # 方式二：均匀切割快照（训练集）
            train_uniform_dir = os.path.join(config["output_snap_dir"], "train_uniform")
            split_snap_by_uniform(
                df=df,
                output_snap_dir=train_uniform_dir,
                train_ratio=config["train_ratio"],
                snapshots_num=config["snapshots_num"],
                sep=config["csv_sep"],
            )

        except Exception as e:
            raise RuntimeError(f"生成快照文件失败：{str(e)}") from e

    # 3. 保存节点集
    try:
        node_dict = {}
        node_dict.update({node: 1 for node in df[source_col].unique()})
        node_dict.update({node: 1 for node in df[target_col].unique()})
        output_graph_dir = os.path.dirname(config["output_node_path"])
        if output_graph_dir:  # 避免空目录
            os.makedirs(output_graph_dir, exist_ok=True)

        # 1. 去重并排序
        node_list = sorted(list(node_dict.keys()))
        if not node_list:
            raise ValueError("节点列表为空！")

        # 2. 保存节点文件
        df_node = pd.DataFrame(node_list, columns=["node"])
        df_node.to_csv(
            config["output_node_path"], sep=config["csv_sep"], index=False, header=False
        )
        logger.info(
            f"✅ 节点集文件保存至: {config['output_node_path']} (节点数: {len(node_list)})"
        )
    except Exception as e:
        raise RuntimeError(f"节点集保存失败：{str(e)}") from e

    logger.info(f"🎉 数据集 {config['DATASET']} 预处理完成！")
