from datetime import datetime
import os
import pandas as pd


# ===================== 通用工具函数 =====================
def trans_id(nid) -> str:
    """
    节点ID转换函数: 不使用数据集的原始ID, 防止引入无关信息, 统一添加前缀
    Args:
        nid: 原始节点ID
    Returns:
        str: 带前缀的标准化节点ID
    """
    return f"U{str(nid)}"

def cut_snapshots_by_month(
    df: pd.DataFrame,
    output_snap_dir: str,
    time_col: str = 'time',
    date_format: str = '%Y-%m',
    sep: str = '\t'
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
    # 1. 输入校验
    if time_col not in df.columns:
        raise ValueError(f"时间列 '{time_col}' 不存在！当前列：{df.columns.tolist()}")
    if len(df) == 0:
        raise ValueError("输入数据框为空，无法切分快照！")
    
    # 2. 时间转换（增加异常处理）
    try:
        df['date'] = df[time_col].apply(
            lambda x: datetime.fromtimestamp(float(x)).strftime(date_format)
        )
    except Exception as e:
        raise RuntimeError(f"时间戳解析失败：{str(e)}")
    
    # 3. 按月份分组生成快照
    month_groups = df.groupby('date')
    if len(month_groups) == 0:
        raise ValueError("未找到有效月份分组！")
    
    # 确保输出目录存在
    os.makedirs(output_snap_dir, exist_ok=True)
    
    # 生成快照
    for idx, (month, df_month) in enumerate(month_groups, 1):
        df_output = df_month.drop('date', axis=1)
        output_file = os.path.join(output_snap_dir, f"{month}.csv")
        df_output.to_csv(output_file, sep=sep, index=False)
        print(f"✅ 按月份切割快照 {idx}/{len(month_groups)}: {month}.csv (行数: {len(df_output)})")

# 生成均匀切割的快照列表（只针对训练集）
def gen_uniform_snapshots(df, output_snap_dir, train_ratio=0.5, snapshots_num=5, sep='\t'):
    """
    将数据按指定比例截取训练集，并均匀分割为指定数量的快照文件
    
    参数说明：
    df: pandas.DataFrame - 原始数据框
    output_snap_dir: str - 快照文件输出目录
    train_ratio: float - 训练数据占比(0 < train_ratio ≤ 1), 默认0.5(50%)
    snapshots_num: int - 快照文件数量(≥1), 默认5
    sep: str - 输出文件分隔符(默认'\t')
    
    返回值：
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
        raise ValueError(f"训练集数据为空！总行数: {len(df)}, 训练集比例: {train_ratio}")
    
    # 计算分片大小
    chunk_size = len(df_train) // snapshots_num
    if chunk_size == 0:
        chunk_size = 1
        snapshots_num = len(df_train)
        print(f"⚠️ 警告：快照数量({snapshots_num}) > 训练集行数({len(df_train)})，已调整为每行一个快照")
    
    # 确保输出目录存在
    os.makedirs(output_snap_dir, exist_ok=True)
    
    # 生成均匀分片快照
    for i in range(snapshots_num):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < snapshots_num - 1 else len(df_train)
        sub_df = df_train.iloc[start:end]
        
        output_file_path = os.path.join(output_snap_dir, f'train_snap_{i + 1}.csv')
        sub_df.to_csv(output_file_path, sep=sep, index=False)
        print(f"✅ 训练集快照 {i+1}/{snapshots_num} 生成完成 (行数: {len(sub_df)})")

def save_node_set(
    node_dict: dict,
    output_node_dir: str,
    filename: str = 'nodes.csv',
    sep: str = '\t'
) -> None:
    """
    保存节点集合到文件
    Args:
        node_dict: 节点字典(key为节点ID)
        output_node_dir: 节点文件输出目录
        filename: 输出文件名
        sep: 分隔符
    """
    # 1. 去重并排序
    node_list = sorted(list(node_dict.keys()))
    if not node_list:
        raise ValueError("节点列表为空！")
    
    # 2. 保存节点文件
    os.makedirs(output_node_dir, exist_ok=True)
    node_file_path = os.path.join(output_node_dir, filename)
    df_node = pd.DataFrame(node_list, columns=['node'])
    df_node.to_csv(node_file_path, sep=sep, index=False, header=False)
    print(f"✅ 生成节点集文件: {filename} (节点数: {len(node_list)})")

import re
from typing import Union, List, Optional
import pandas as pd


def normalize_whitespace(input_data: Union[str, List[str], pd.Series], 
                         keep_empty_lines: bool = False) -> Union[str, List[str], pd.Series]:
    """
    统一归一化空白字符：将任意空白字符（空格、\t、\n、多个连续空格）替换为单个空格
    支持字符串、字符串列表、pandas Series 三种输入类型
    
    Args:
        input_data: 输入数据（字符串/字符串列表/pd.Series）
        keep_empty_lines: 是否保留空行（仅对列表/Series生效），默认False（过滤空行）
    
    Returns:
        归一化后的结果（与输入类型一致）
    """
    # 正则表达式：匹配任意空白字符（空格、\t、\n、\r等）
    whitespace_pattern = re.compile(r'\s+')
    
    def _normalize_single_line(line: str) -> str:
        """处理单行字符串"""
        # 1. 替换所有空白字符为单个空格
        normalized = whitespace_pattern.sub(' ', line.strip())
        # 2. 去除首尾空格（最终结果不会有首尾空格）
        return normalized.strip()
    
    # 分类型处理输入
    if isinstance(input_data, str):
        # 场景1：处理单个字符串
        return _normalize_single_line(input_data)
    
    elif isinstance(input_data, list):
        # 场景2：处理字符串列表（比如文件行列表）
        normalized_lines = []
        for line in input_data:
            normalized_line = _normalize_single_line(str(line))
            # 过滤空行（如果需要）
            if keep_empty_lines or normalized_line:
                normalized_lines.append(normalized_line)
        return normalized_lines
    
    elif isinstance(input_data, pd.Series):
        # 场景3：处理pandas Series（比如DataFrame的某一列）
        normalized_series = input_data.astype(str).apply(_normalize_single_line)
        if not keep_empty_lines:
            normalized_series = normalized_series[normalized_series != '']
        return normalized_series
    
    else:
        raise TypeError(f"不支持的输入类型: {type(input_data)}，仅支持str/list/pd.Series")


# ------------------------------ 扩展工具：读取文件并归一化空白符 ------------------------------
def read_file_with_normalized_whitespace(file_path: str, 
                                         skip_rows: int = 0,
                                         col_names: Optional[List[str]] = None,
                                         encoding: str = 'utf-8') -> pd.DataFrame:
    """
    读取文本文件，自动归一化空白字符，返回DataFrame
    适配任意空白分隔符（空格/\t/多个空格混合）
    
    Args:
        file_path: 文件路径
        skip_rows: 跳过前N行
        col_names: 列名列表
        encoding: 文件编码
    
    Returns:
        解析后的DataFrame
    """
    # 1. 读取文件所有行
    with open(file_path, 'r', encoding=encoding) as f:
        lines = f.readlines()
    
    # 2. 跳过指定行数
    if skip_rows > 0:
        lines = lines[skip_rows:]
    
    # 3. 归一化空白字符
    normalized_lines = normalize_whitespace(lines, keep_empty_lines=False)
    
    # 4. 分割每行成字段，转换为DataFrame
    parsed_data = [line.split(' ') for line in normalized_lines]
    
    # 校验字段数（如果指定了列名）
    if col_names:
        expected_cols = len(col_names)
        for idx, row in enumerate(parsed_data):
            if len(row) != expected_cols:
                raise ValueError(f"第{skip_rows + idx + 1}行字段数不匹配："
                                 f"实际{len(row)}个，期望{expected_cols}个")
    
    # 5. 转换为DataFrame
    df = pd.DataFrame(parsed_data, columns=col_names)
    return df