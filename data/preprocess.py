import logging
import os
import pandas as pd

from data.config import get_data_config
from data.data_utils import read_file_norm_ws, trans_id, save_nodes_mapping

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
        config = get_data_config(dataset_name)

    try:
        # 1. 读取原始数据
        df = read_file_norm_ws(
            file_path=config["input_file_path"],
            skip_rows=config["skip_rows"],
            col_names=config["col_names"],
        )
        logger.info(f"✅ 数据集加载完成: {config['input_file_path']} (行数: {len(df)})")

        # 2. 按时间列升序排列
        time_col = config["time_col"]  # 时间列需是数值型或日期型
        if time_col in df.columns:
            df[time_col] = pd.to_numeric(df[time_col], errors="raise")
            df = df.sort_values(by=time_col, ascending=True).reset_index(drop=True)
        else:
            logger.error(f"⚠️ 未在数据集中找到时间列 [{time_col}]")

        source_col, target_col = config["node_cols"]
        # 3. 保存节点(全部节点)映射: 原始ID -> 匿名化ID -> 数字ID
        save_nodes_mapping(
            nodes_iterable=pd.concat([df[source_col], df[target_col]]).unique(),
            output_path=config["output_nodes_mapping_path"],
            sep=config["csv_sep"],
        )

        # 4. 匿名化处理
        df[[source_col, target_col]] = df[[source_col, target_col]].apply(
            lambda x: x.map(trans_id)
        )

        # 5. 保存图数据
        output_graph_dir = os.path.dirname(config["output_graph_path"])
        if output_graph_dir:
            os.makedirs(output_graph_dir, exist_ok=True)
        df.to_csv(config["output_graph_path"], sep=config["csv_sep"], index=False)

        # 切割快照（可选）
        if config["need_cut_snap"]:
            from data.data_utils import split_snap_by_month, split_snap_by_uniform

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

        logger.info("✅ 数据集预处理完成")

    except FileNotFoundError as e:
        raise FileNotFoundError(f"数据集预处理执行失败: 文件不存在 - {str(e)}") from e
    except Exception as e:
        raise RuntimeError(f"数据集预处理执行失败: {str(e)}") from e
