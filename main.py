import argparse
import logging
from data.config import MODEL_DEFAULT_CONFIG, get_config
from model.train import run_training


def main(args: argparse.Namespace) -> dict:
    """
    命令行入口函数: 将命令行参数转换为配置字典, 调用run_training
    Args:
        args: 命令行参数对象
    Returns:
        dict: 训练结果
    """
    # 转字典
    args_dict = vars(args)
    # 只保留显式传入的参数
    config_updates = {k: v for k, v in args_dict.items() if v is not None}
    return run_training(config_updates, get_config(config_updates["DATASET"]))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="IPNet for Dynamic Network Link Prediction (Apple Silicon Adapted)")
    # 基础配置
    parser.add_argument("--seed", dest="SEED", type=int, help=f"Random seed (default: {MODEL_DEFAULT_CONFIG['SEED']})")
    # 预处理
    parser.add_argument("--pp", dest="PRE_PROCESS", type=bool, default=True, help="Whether to run dataset preprocessing before training (default: True)")
    # 数据集与任务配置
    parser.add_argument("--dataset", dest="DATASET", default="UCI", help=f"Dataset name (default: {MODEL_DEFAULT_CONFIG['DATASET']})")
    parser.add_argument("--ty", dest="TASK_TYPE", help=f"Task type: T/I (default: {MODEL_DEFAULT_CONFIG['TASK_TYPE']})")
    parser.add_argument("--mask", dest="MASK_RATIO", type=float, help=f"Mask ratio for inductive task (default: {MODEL_DEFAULT_CONFIG['MASK_RATIO']})")
    # 模型参数
    parser.add_argument("--v", dest="VERSION", help=f"IPNet version: mean/att/w2v (default: {MODEL_DEFAULT_CONFIG['VERSION']})")
    parser.add_argument("--fd", dest="FEAT_DIM", type=int, help=f"Feature dimension (default: {MODEL_DEFAULT_CONFIG['FEAT_DIM']})")
    parser.add_argument("--rnn", dest="RNN_TYPE", help=f"RNN type: LSTM/GRU (default: {MODEL_DEFAULT_CONFIG['RNN_TYPE']})")
    parser.add_argument("--pd", dest="PADDING_NODE", type=int, help=f"Padding node ID (default: {MODEL_DEFAULT_CONFIG['PADDING_NODE']})")
    # 核心超参
    parser.add_argument("--il", dest="IS_LEN", type=int, help=f"Interaction sequence length (default: {MODEL_DEFAULT_CONFIG['IS_LEN']})")
    parser.add_argument("--wn", dest="WALK_NUM", type=int, help=f"Random walks per node (default: {MODEL_DEFAULT_CONFIG['WALK_NUM']})")
    parser.add_argument("--wl", dest="WALK_LEN", type=int, help=f"Single walk length (default: {MODEL_DEFAULT_CONFIG['WALK_LEN']})")
    # 训练配置
    parser.add_argument("--epoch", dest="EPOCH", type=int, help=f"Training epochs (default: {MODEL_DEFAULT_CONFIG['EPOCH']})")
    parser.add_argument("--bs", dest="BATCH_SIZE", type=int, help=f"Batch size (default: {MODEL_DEFAULT_CONFIG['BATCH_SIZE']})")
    parser.add_argument("--lr", dest="LR", type=float, help=f"Learning rate (default: {MODEL_DEFAULT_CONFIG['LR']})")
    parser.add_argument("--thread", dest="THREAD_NUM", type=int, help=f"Number of workers (default: {MODEL_DEFAULT_CONFIG['THREAD_NUM']})")
    parser.add_argument("--device", dest="DEVICE", type=int, help=f"Device: -2=CPU, -1=MPS, ≥0=GPU (default: {MODEL_DEFAULT_CONFIG['DEVICE']})")

    args = parser.parse_args()
    # 执行训练并返回结果
    result = main(args)
