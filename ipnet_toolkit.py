import logging
import numpy as np
import torch
from data.config import DEFAULT_TRAIN_CONFIG, DEFAULT_MODEL_CONFIG, get_data_config
from data.preprocess import preprocess
from model.IPNet import IPNet
from model.model_utils import get_device, load_best_model
from model.train import run_experiment

logger = logging.getLogger(__name__)


class IPNetToolkit:
    """IPNet模型工具类: 功能封装, 方便外部调用"""

    def __init__(self, input_configs: dict | None = None):
        self.config = {**DEFAULT_TRAIN_CONFIG, **DEFAULT_MODEL_CONFIG, **input_configs}
        self.device = get_device(self.config["DEVICE"])
        self.IPNet = None

    def train(self, data_config: dict | None = None) -> dict:
        """训练"""
        data_config = data_config or get_data_config(self.config["DATASET"])
        if self.config["PRE_PROCESS"]:
            try:
                preprocess(data_config)
            except Exception as e:
                logger.error(f"❌ 数据集预处理失败: {str(e)}", exc_info=True)
                raise RuntimeError(f"数据集预处理失败: {str(e)}") from e
        return run_experiment(config=self.config, data_config=data_config)

    def load_best_model(self) -> IPNet:
        """加载训练最佳模型"""
        self.IPNet = load_best_model(config=self.config, device=self.device)

    def predict(
        self,
        src_nodes: list[int] | np.ndarray | torch.Tensor,
        tgt_nodes: list[int] | np.ndarray | torch.Tensor,
    ) -> np.ndarray:
        """预测节点对是否存在链接"""
        # 实现预测逻辑
        pass

    def predict_batch(
        self,
        edges: list[tuple[int, int]] | np.ndarray | torch.Tensor,
        batch_size: int = 1024,
    ) -> np.ndarray:
        """批量预测"""
        pass

    def get_node_embedding(self, nodes: int | list[int]) -> np.ndarray:
        """获取节点嵌入向量"""
        pass

    def save_model(self, save_path: str) -> str:
        """保存当前模型"""
        pass

    def info(self) -> dict:
        """获取模型信息"""
        pass
