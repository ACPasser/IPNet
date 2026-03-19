import logging
import numpy as np
import torch
import random
import os
import re
import json
from datetime import datetime
from model.IPNet import IPNet
from pathlib import Path

logger = logging.getLogger(__name__)


class EarlyStopMonitor(object):
    def __init__(self, max_round=3, higher_better=True, tolerance=1e-3):
        self.max_round = max_round
        self.num_round = 0

        self.epoch_count = 0
        self.best_epoch = 0

        self.last_best = None
        self.higher_better = higher_better
        self.tolerance = tolerance

    def early_stop_check(self, curr_val):
        if not self.higher_better:
            curr_val *= -1
        if self.last_best is None:
            self.last_best = curr_val
        elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
            self.last_best = curr_val
            self.num_round = 0
            self.best_epoch = self.epoch_count
        else:
            self.num_round += 1
        self.epoch_count += 1
        return self.num_round >= self.max_round


def set_random_seed(seed: int) -> None:
    """
    统一设置随机种子(兼容CPU/GPU/MPS, 适配不同PyTorch版本)
    Args:
        seed: 随机种子值
    """
    # 1. 基础随机种子（Python/numpy/PyTorch CPU）
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)  # 固定Python哈希种子

    # 2. CUDA（NVIDIA GPU）种子 + 确定性设置
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # 关闭自动调优（避免非确定性）

    # 3. MPS（Apple Silicon）种子 + 确定性设置（版本兼容）
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # 仅在PyTorch 2.1+版本设置MPS确定性（低版本无该属性）
        if hasattr(torch.backends.mps, "deterministic"):
            torch.backends.mps.deterministic = True

    # 4. 全局确定性算法（带异常处理，避免算子不支持）
    try:
        # warn_only=True：不支持的算子仅警告，不崩溃
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception as e:
        logger.info(f"⚠️ 部分算子不支持确定性模式：{e}，已降级为警告模式")

    # logger.info(f"✅ Random seeds set to: {seed} (deterministic mode enabled)")


def get_device(gpu_flag: int) -> torch.device:
    """
    Args:
        gpu_flag:
            -2 = 强制使用CPU
            -1 = 尝试使用Apple Silicon MPS(M1/M2/M3系列)
            ≥0 = 尝试使用NVIDIA GPU(支持多卡时可指定卡号, 如0/1/2)
    Returns: torch.device
    """
    if gpu_flag == -2:
        device = torch.device("cpu")
        logger.info("✅ 设备选用: CPU")

    elif gpu_flag == -1:
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps")
            logger.info("📌 设备选用: Apple MPS (Apple Silicon acceleration)")
        else:
            device = torch.device("cpu")
            logger.info(
                "⚠️ MPS not available (need PyTorch ≥2.0 + Apple Silicon), fallback to CPU"
            )

    else:
        if torch.cuda.is_available():
            # 如果传入具体卡号（如0/1），则使用对应GPU；否则默认用第0卡
            device = torch.device(f"cuda:{gpu_flag}" if gpu_flag >= 0 else "cuda:0")
            logger.info(f"📌 设备选用: NVIDIA GPU (cuda:{gpu_flag})")
        else:
            device = torch.device("cpu")
            logger.info("⚠️ NVIDIA GPU not available, fallback to CPU")

    return device


def get_result_path(cfg, final_seq_len, final_walk_num, final_walk_len):
    # 取核心参数
    core_param_desc_list = [
        f"SEED-{cfg['SEED']}",
        f"TASK-{cfg['TASK_TYPE']}",
        f"IL-{final_seq_len}",
        f"WN-{final_walk_num}",
        f"WL-{final_walk_len}",
    ]

    if cfg["TASK_TYPE"] == "I":
        mask_percent = cfg["MASK_RATIO"] * 100
        if mask_percent.is_integer():
            mask_ratio_str = f"{int(mask_percent)}%"
        else:
            mask_ratio_str = f"{mask_percent:.1f}%"
        core_param_desc_list.append(f"MR-{mask_ratio_str}")

    core_params_str = "_".join(core_param_desc_list)

    # 测试结果路径
    result_path = os.path.join(
        cfg["RESULT_PATH"].format(
            dataset=cfg["DATASET"], core_params=core_params_str, version=cfg["VERSION"]
        ),
    )

    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    return result_path


def load_best_model(config: dict, device: torch.device | None = None) -> IPNet:
    """
    加载最近训练的最佳模型
    Args:
        config: 合并了"训练配置"和"模型配置"
        device: 指定 or 根据config["DEVICE"]选择
    Returns:
        加载好的IPNet模型(已设置为eval模式)
    """
    target_device = device or get_device(config["DEVICE"])

    # 1. 参数字典
    state_dict_path, timestamp = get_state_dict_path(config)
    state_dict = torch.load(state_dict_path, map_location=target_device)

    # 1. 模型初始化参数
    param_save_config = {
        **config["PARAM_SAVE_CONFIG"],
        "dir": config["PARAM_SAVE_CONFIG"]["dir"].format(
            dataset=config["DATASET"], timestamp=timestamp
        ),
    }
    ipnet = _rebuild_model(param_save_config, target_device)
    ipnet.load_state_dict(state_dict)
    ipnet.eval()

    return ipnet


def get_state_dict_path(config: dict) -> tuple[str, str]:
    """
    获取最新的模型路径
    Args:
        config: 训练配置, 需包含:
            - "BEST_MODEL_PATH": 路径模板，如 "outputs/{dataset}/best_models/{timestamp}/best-model.pth"
            - "DATASET": 数据集名称
    Returns:
        Tuple[str, str]: (模型文件路径, 时间戳)
    Raises:
        FileNotFoundError: 如果找不到任何存在的模型文件
    """
    base_dir = Path(
        config["BEST_MODEL_PATH"].format(dataset=config["DATASET"], timestamp="")
    ).parent

    if not base_dir.exists():
        raise FileNotFoundError(f"模型目录不存在: {base_dir}")

    # 时间戳目录(格式：8位日期+4位时间，如202603162257)
    timestamp_pattern = re.compile(r"^\d{12}$")
    timestamp_dirs = [
        d for d in base_dir.iterdir() if d.is_dir() and timestamp_pattern.match(d.name)
    ]

    if not timestamp_dirs:
        raise FileNotFoundError(f"未找到时间戳格式的目录: {base_dir}")

    timestamp_dirs.sort(key=lambda x: x.name, reverse=True)

    checked_dirs = []
    for ts_dir in timestamp_dirs:
        checked_dirs.append(ts_dir.name)

        state_dict_path = config["BEST_MODEL_PATH"].format(
            dataset=config["DATASET"], timestamp=ts_dir.name
        )

        if os.path.exists(state_dict_path):
            # fmt: off
            fmt_timestamp = datetime.strptime(ts_dir.name, "%Y%m%d%H%M").strftime("%Y-%m-%d %H:%M")
            logger.info("✅ 参数字典加载完成")
            logger.info(f"   ├─ 路径: {state_dict_path}")
            logger.info(f"   └─ 训练时间: {fmt_timestamp}")
            # fmt: on
            return state_dict_path, ts_dir.name
        else:
            logger.debug(f"目录 {ts_dir.name} 下没有模型文件")

    error_msg = (
        f"在以下所有 {len(checked_dirs)} 个时间戳目录中均未找到模型文件:\n"
        f"  基础目录: {base_dir}\n"
        f"  检查的目录 (从新到旧): {', '.join(checked_dirs[:10])}"
    )
    if len(checked_dirs) > 10:
        error_msg += f"\n  ... 还有 {len(checked_dirs) - 10} 个更早的目录"

    raise FileNotFoundError(error_msg)


def _rebuild_model(param_save_config: dict, device: torch.device) -> IPNet:
    """
    根据模型配置中的 PARAM_SAVE_CONFIG 重建模型
    """
    psc = param_save_config  # 简化
    base_dir = psc["dir"]

    # 1. 节点特征
    node_feature = np.load(os.path.join(base_dir, psc["node_feature"]))

    # 2. 交互序列
    with open(os.path.join(base_dir, psc["interactions"]), "r") as f:
        interactions = json.load(f)
        interactions = {int(k): v for k, v in interactions.items()}

    # 3. 上下文
    with open(os.path.join(base_dir, psc["contexts"]), "r") as f:
        contexts = json.load(f)
        contexts = {int(k): v for k, v in contexts.items()}

    # 4. 其他参数
    with open(os.path.join(base_dir, psc["other_params"]), "r", encoding="utf-8") as f:
        model_params = json.load(f)
    model = IPNet(
        node_feature=node_feature,
        interactions=interactions,
        contexts=contexts,
        final_seq_len=model_params["FINAL_SEQ_LEN"],
        final_walk_len=model_params["FINAL_WALK_LEN"],
        version=model_params["VERSION"],
        rnn_type=model_params["RNN_TYPE"],
        n_head=model_params["N_HEAD"],
        dropout_p=model_params["DROPOUT"],
        padding_node=model_params["PADDING_NODE"],
        device=device,
    )

    return model
