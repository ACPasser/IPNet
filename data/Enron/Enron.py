from datetime import datetime
import pandas as pd
import os
import sys
# ========== 添加项目根目录到Python路径 ==========
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
sys.path.append(project_root) # 添加根目录到sys.path，让Python能找到data/util
from data.util import trans_id, cut_snapshots_by_month, save_node_set


if __name__ == '__main__':
    # 配置项
    CONFIG = {
        "input_file_path": "data/Enron/0.origin/graph.txt",
        "output_graph_path": "data/Enron/0.origin/graph.csv",
        "output_snap_dir": "data/Enron/1.snapshots",
        "output_node_dir": "data/Enron/1.nodes_set",
        "csv_sep": "\t",          # 文件分隔符
        "time_col": "time",       # 时间戳列名
        "date_format": "%Y-%m",   # 日期格式化
        "skip_rows": 0,           # 跳过行数
        "col_names": ['from_id', 'to_id', 'type', 'time']  # 列名映射
    }

    # 1. 读取原始数据
    try:
        df = pd.read_csv(
            CONFIG["input_file_path"],
            sep=' ',
            header=None,
            skiprows=CONFIG["skip_rows"],
            names=CONFIG["col_names"],
            dtype=str  # 统一按字符串读取，避免ID类型错误
        )
        print(f"✅ 成功读取原始数据: {CONFIG['input_file_path']} (行数: {len(df)})")
        # 转换节点ID
        df[['from_id', 'to_id']] = df[['from_id', 'to_id']].apply(
            lambda x: x.map(trans_id)
        )
        # save graph
        df.to_csv(CONFIG["output_graph_path"], sep=CONFIG["csv_sep"], index=False)
        print(f"✅ 保存处理后图数据: {CONFIG['output_graph_path']}")
    except Exception as e:
        raise FileNotFoundError(f"读取原始数据失败：{str(e)}")
    
    # 2. 按月切分快照（用于baseline）
    cut_snapshots_by_month(
        df=df,
        output_snap_dir=CONFIG["output_snap_dir"],
        time_col=CONFIG["time_col"],
        date_format=CONFIG["date_format"],
        sep=CONFIG["csv_sep"]
    )

    # 3. save node set
    node_dict = {}
    node_dict.update({node: 1 for node in df['from_id'].unique()})
    node_dict.update({node: 1 for node in df['to_id'].unique()})
    save_node_set(
        node_dict=node_dict,
        output_node_dir=CONFIG["output_node_dir"],
        sep=CONFIG["csv_sep"]
    )

    print("\n🎉 所有处理完成！")