import logging
import numpy as np
import torch
import random
import os

logger = logging.getLogger(__name__)


def set_random_seeds(seed: int) -> None:
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
        logger.info("✅ 训练设备选用: CPU")

    elif gpu_flag == -1:
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps")
            logger.info("📌 训练设备选用: Apple MPS (Apple Silicon acceleration)")
        else:
            device = torch.device("cpu")
            logger.info(
                "⚠️ MPS not available (need PyTorch ≥2.0 + Apple Silicon), fallback to CPU"
            )

    else:
        if torch.cuda.is_available():
            # 如果传入具体卡号（如0/1），则使用对应GPU；否则默认用第0卡
            device = torch.device(f"cuda:{gpu_flag}" if gpu_flag >= 0 else "cuda:0")
            logger.info(f"📌 训练设备选用: NVIDIA GPU (cuda:{gpu_flag})")
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
        os.path.dirname(os.path.abspath(__file__)),
        cfg["RESULT_PATH"].format(
            dataset=cfg["DATASET"], core_params=core_params_str, version=cfg["VERSION"]
        ),
    )

    result_dir = os.path.dirname(result_path)
    if result_dir:
        os.makedirs(result_dir, exist_ok=True)

    return result_path


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
