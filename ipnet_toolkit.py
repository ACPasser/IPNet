import logging
import numpy as np
import torch
import pandas as pd
from data.preprocess import preprocess
from data.data_utils import trans_id
from model.ipnet import IPNet
from model.model_utils import get_device, load_best_model
from model.pipeline import train_and_eval, test_model

logger = logging.getLogger(__name__)


class IPNetToolkit:
    """IPNet模型工具类: 功能封装, 方便外部调用"""

    def __init__(self, config: dict):
        self.config = config  # 包含: 数据集配置、训练配置、模型配置
        self.device = get_device(self.config["DEVICE"])
        self.raw2id: dict[str, int] = {}
        self.IPNet = None

    def _load_node_mapping(self):
        """构建 node2id 映射"""
        try:
            mapping_df = pd.read_csv(
                self.config["output_nodes_mapping_path"],
                sep=self.config["csv_sep"],
            )
            self.raw2id = dict(zip(mapping_df["original_id"], mapping_df["numeric_id"]))
            logger.info(f"✅ 训练节点映射加载完成(节点数: {len(self.raw2id)})")
        except Exception as e:
            logger.error(f"节点映射记载失败: {e}, 请检查是否预处理过数据集...")
            raise

    def run_pipeline(self, do_preprocess: bool | None = None) -> dict:
        """训练"""
        # fmt: off
        should_run = (do_preprocess if do_preprocess is not None else self.config["PRE_PROCESS"])
        if should_run:
            preprocess(self.config)
        self.IPNet = train_and_eval(config=self.config, device=self.device)
        self._load_node_mapping()
        # fmt: on

    def run_preprocess(self):
        """手动触发预处理"""
        preprocess(self.config)

    def load_best_model(self) -> IPNet:
        """加载训练最佳模型"""
        self.IPNet = load_best_model(config=self.config, device=self.device)
        self._load_node_mapping()

    def _check_model_loaded(self):
        """检查模型是否已加载，如果未加载则抛出异常"""
        # fmt: off
        if self.IPNet is None:
            raise RuntimeError("模型尚未加载, 无法进行预测。请先调用 run_training() 训练模型或 load_best_model() 加载模型...")
        # fmt: on

    def test_model(self) -> dict:
        """测试当前模型"""
        self._check_model_loaded()
        return test_model(self.IPNet, config=self.config)

    def predict(
        self,
        edges: list[tuple] | np.ndarray,
        batch_size: int = 1024,
    ) -> np.ndarray:
        """批量预测节点对（如: [('user_1', 'user_2'), ... ]）形成链接的概率
        Args:
            edges: 待预测的节点对列表(训练集中的原始ID)
            batch_size (可选): 批处理大小
        返回:
            np.ndarray: 一维概率NP数组
        """
        # check
        if not self.raw2id:
            self._load_node_mapping()
        self._check_model_loaded()
        self.IPNet.eval()

        # [N, 2]
        edges_arr = np.array(edges, dtype=object)
        if len(edges_arr) == 0:
            return np.array([], dtype=np.float32)

        all_scores = []
        with torch.no_grad():
            for i in range(0, len(edges_arr), batch_size):
                batch = edges_arr[i : i + batch_size]

                # 1. 原始 ID -> 匿名 ID -> 数字 ID
                src_numeric = self._to_numeric_ids(batch[:, 0])
                tgt_numeric = self._to_numeric_ids(batch[:, 1])

                # 2. 转 Tensor
                # fmt: off
                src_tensor = torch.as_tensor(src_numeric, dtype=torch.long, device=self.device)
                tgt_tensor = torch.as_tensor(tgt_numeric, dtype=torch.long, device=self.device)
                edges_tensor = torch.stack([src_tensor, tgt_tensor], dim=1)
                # fmt: on
                # 3. 预测
                scores = self.IPNet(edges_tensor)
                batch_scores = scores.cpu().numpy()

                all_scores.append(batch_scores)

        return np.concatenate(all_scores)

    def _to_numeric_ids(self, raw_nodes: list | np.ndarray) -> np.ndarray:
        """原始ID -> 数字ID"""
        # fmt: off
        numeric_ids = []
        for node in raw_nodes:
            nid = self.raw2id.get(node, self.config["PADDING_NODE"])  # 不存在，则返回pending node
            if nid == 0:
                logger.warning(f"节点 {node} (匿名: {trans_id(node)}) 不在训练节点集中！")
            numeric_ids.append(nid)
        # fmt: on
        return np.array(numeric_ids)

    def get_node_embedding(self, nodes: int | list[int]) -> np.ndarray:
        """获取节点嵌入向量"""
        pass

    def save_model(self, save_path: str) -> str:
        """保存当前模型"""
        pass

    def info(self) -> dict:
        """获取模型信息"""
        pass
