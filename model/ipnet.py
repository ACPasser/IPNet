import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from data.config import DEFAULT_MODEL_CONFIG

logger = logging.getLogger(__name__)


def init_weights(m: nn.Module):
    """参数统一初始化"""
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, (nn.LSTM, nn.GRU)):
        for name, param in m.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0.0)


class InteractionSequenceEncoder(nn.Module):
    """对「节点交互序列」进行时间 / 位置编码，生成特征"""

    def __init__(
        self,
        node_num: int,
        time_dim: int,
        pos_dim: int,
        node_interactions: dict[int, list[tuple[int, float]]],
        specified_seq_len: int | None = None,
        padding_node: int = 0,
        dropout_p: float = 0.1,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.node_num = node_num
        self.time_dim = time_dim
        self.pos_dim = pos_dim
        self.dropout_p = dropout_p
        self.device = device
        self.padding_node = padding_node

        if specified_seq_len is not None:
            self.seq_len = specified_seq_len
        else:
            self.seq_len = max(
                len(seq) for seq in node_interactions.values()
            )  # 默认取最长，不足的padding
        self.node_interactions = self._pad_interactions(node_interactions)

        # --------------------------- 时间编码 ---------------------------
        self.time_tensor = torch.zeros(
            (self.node_num, self.seq_len), dtype=torch.float32, device=self.device
        )
        self._init_time_tensor()

        # 双频率基（增强时间特征区分度）
        self.basis_freq1 = nn.Parameter(
            torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.time_dim // 2))
            .float()
            .to(device)
        )
        self.basis_freq2 = nn.Parameter(
            torch.from_numpy(1 / np.exp(np.linspace(0, 5, self.time_dim // 2)))
            .float()
            .to(device)
        )
        self.phase = nn.Parameter(torch.zeros(self.time_dim).float().to(device))

        # 时间特征门控融合（减少噪声）
        self.time_gate = nn.Sequential(
            nn.Linear(self.time_dim, self.time_dim), nn.Sigmoid(), nn.Dropout(dropout_p)
        ).to(device)
        self.time_gate.apply(init_weights)

        # --------------------------- 位置编码 ---------------------------
        self.pos_tensor = torch.zeros(
            (self.node_num, self.seq_len, self.seq_len),
            dtype=torch.float32,
            device=self.device,
        )
        self._init_pos_tensor()

        self.trainable_embedding = nn.Sequential(
            nn.Linear(self.seq_len, self.pos_dim),
            nn.GELU(),
            nn.LayerNorm(self.pos_dim),
            nn.Linear(self.pos_dim, self.pos_dim),
        ).to(device)
        self.trainable_embedding.apply(init_weights)

    def _pad_interactions(
        self, node_interactions: dict[int, list[tuple[int, float]]]
    ) -> dict[int, list[tuple[int, float]]]:
        """
        Args:
            node_interactions: 原始交互序列字典 {节点ID: [(邻居ID, 时间戳), ...]}
        Returns:
            经过padding的交互序列字典
        """
        padded_interactions = {}
        for node_id, interactions in node_interactions.items():
            current_len = len(interactions)

            if current_len == self.seq_len:
                # 直接保留
                padded_interaction = interactions
            elif current_len < self.seq_len:
                # 已按时间倒序排列，在结尾padding
                pad_num = self.seq_len - current_len
                base_ts = interactions[-1][1] if current_len > 0 else 0.0
                pad = [(self.padding_node, base_ts) for _ in range(pad_num)]
                padded_interaction = interactions + pad
            else:
                # 截断策略：保留最近的交互
                padded_interaction = interactions[: self.seq_len]

            padded_interactions[node_id] = padded_interaction

        return padded_interactions

    def _init_time_tensor(self):
        for node_id, seq in self.node_interactions.items():
            ts_list = [interaction[1] for interaction in seq]
            self.time_tensor[node_id] = torch.tensor(
                ts_list, dtype=torch.float32, device=self.device
            )

        # 填充归一化∆t [node_num, seq_len-1]
        time_diff = self.time_tensor[:, :-1] - self.time_tensor[:, 1:]
        # [node_num, 1]
        time_span = (self.time_tensor[:, 0:1] - self.time_tensor[:, -1:]) + 1e-8

        normed_diff = time_diff / time_span
        normed_diff = torch.clamp(normed_diff, min=-1.0, max=1.0)

        # [node_num, seq_len]
        self.time_tensor[:, :-1] = normed_diff
        self.time_tensor[-1:, -1] = 0.0

    def _init_pos_tensor(self):
        """填充归一化位置计数"""
        # node_pos_counter = {}
        # for node in self.node_interactions:
        #     node_pos_counter[node] = np.zeros(self.seq_len, dtype=np.float32)
        # node_pos_counter[self.padding_node] = np.zeros(self.seq_len, dtype=np.float32)
        node_pos_counter = np.zeros((self.node_num, self.seq_len), dtype=np.float32)

        # 统计节点在交互序列中各位置的出现次数
        for _, seq in self.node_interactions.items():
            for seq_idx in range(self.seq_len):
                node, _ = seq[seq_idx]
                node_pos_counter[node][seq_idx] += 1

        for node, seq in self.node_interactions.items():
            for seq_idx in range(self.seq_len):
                self.pos_tensor[node, seq_idx, :] = torch.tensor(
                    node_pos_counter[node], dtype=torch.float32, device=self.device
                )

    def _encode_time(self, nodes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            nodes: [batch] 节点ID张量
        Returns:
            [B, L, TD] 交互时间编码特征
        """
        batch_size = nodes.shape[0]
        times_tensor = self.time_tensor[nodes]  # [B, L]
        times_tensor = times_tensor.view(batch_size, self.seq_len, 1)

        # 双频率编码
        freq1 = times_tensor * self.basis_freq1.view(1, 1, -1)
        freq2 = times_tensor * self.basis_freq2.view(1, 1, -1)
        map_ts = torch.cat(
            [
                torch.cos(freq1 + self.phase[: self.time_dim // 2]),
                torch.sin(freq2 + self.phase[self.time_dim // 2 :]),
            ],
            dim=-1,
        )

        # 门控融合
        gate = self.time_gate(map_ts)
        map_ts = map_ts * gate + map_ts.detach() * (1 - gate)  # 残差门控

        map_ts = F.layer_norm(map_ts, map_ts.shape[1:])  # 增加层归一化
        map_ts = F.dropout(map_ts, p=self.dropout_p, training=self.training)
        return map_ts

    def _encode_pos(self, nodes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            nodes: [batch] 节点ID张量
        Returns:
            [B, L, pos_dim] 位置编码特征
        """
        # [B, L, L]
        pos_tensor_batch = self.pos_tensor[nodes]

        # 重塑维度以适配嵌入层：[B×L, L]
        batch_shape = pos_tensor_batch.shape
        pos_reshaped = pos_tensor_batch.reshape(-1, self.seq_len)

        # 过嵌入层：[B×L, pos_dim]
        pos_encoded = self.trainable_embedding(pos_reshaped)

        # 恢复原始维度：[B, L, pos_dim]
        pos_encoded = pos_encoded.reshape(batch_shape[0], batch_shape[1], self.pos_dim)

        pos_encoded = F.layer_norm(pos_encoded, pos_encoded.shape[1:])
        pos_encoded = F.dropout(pos_encoded, p=self.dropout_p, training=self.training)
        return pos_encoded

    def forward(
        self, nodes: torch.Tensor, return_separate: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播：融合时间+位置编码
        Args:
            nodes: [batch] 节点ID张量
            return_separate: 是否返回分离的时间/位置特征（默认返回拼接后的融合特征）
        Returns:
            如果return_separate=True: (time_feat, pos_feat)
            否则: 拼接后的融合特征 [B, L, time_dim+pos_dim]
        """
        time_feat = self._encode_time(nodes)  # [B, L, time_dim]
        pos_feat = self._encode_pos(nodes)  # [B, L, pos_dim]

        if return_separate:
            return time_feat, pos_feat
        else:
            # 拼接融合时间+位置特征
            fusion_feat = torch.cat([time_feat, pos_feat], dim=-1)
            return fusion_feat


class SequenceFeatureAggregator(nn.Module):
    """对序列编码特征进行聚合，生成节点级全局特征"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        rnn_type: str = "GRU",
        dropout_p: float = 0.1,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.hidden_dim_one_direction = self.hidden_dim // 2
        self.rnn_type = rnn_type.upper()
        assert self.rnn_type in ["LSTM", "GRU"], (
            f"RNN类型仅支持LSTM/GRU, 当前为{self.rnn_type}"
        )
        self.device = device

        # 构建RNN（统一设备）
        if self.rnn_type == "LSTM":
            self.rnn = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=self.hidden_dim_one_direction,
                batch_first=True,
                bidirectional=True,
            ).to(self.device)
        else:
            self.rnn = nn.GRU(
                input_size=self.input_dim,
                hidden_size=self.hidden_dim_one_direction,
                batch_first=True,
                bidirectional=True,
            ).to(self.device)

        self.dropout = nn.Dropout(dropout_p).to(self.device)
        # 统一初始化
        self.apply(init_weights)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X: [B, L, D] 输入特征
        Returns:
            [B, D] 编码后的特征
        """
        X = F.layer_norm(X, normalized_shape=(self.input_dim,)).to(self.device)
        encoded_features, _ = self.rnn(X)

        # 方式1. 取第一个时间步
        # encoded_features = encoded_features[:, 0, :]
        # 方式2. 取所有时间步的均值，保留完整序列信息
        encoded_features = encoded_features.mean(dim=1)
        encoded_features = self.dropout(encoded_features)
        return encoded_features


class ContextEncoder(nn.Module):
    """上下文编码器（时间+位置编码）"""

    def __init__(
        self,
        nodes_num: int,
        time_dim: int,
        pos_dim: int,
        contexts: dict[int, list[list[tuple[int, float]]]],
        specified_walk_len: int | None = None,
        padding_node: int = 0,
        dropout_p: float = 0.1,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.node_num = nodes_num
        self.time_dim = time_dim
        self.pos_dim = pos_dim
        self.padding_node = padding_node
        self.dropout_p = dropout_p
        self.device = device

        # 确定游走参数（所有节点轮数/长度一致）
        self.walk_num = len(next(iter(contexts.values())))  # 所有节点轮数一致
        if specified_walk_len is not None:
            self.walk_len = specified_walk_len
        else:
            self.walk_len = max(
                len(walk) for walks in contexts.values() for walk in walks
            )  # 取最长，不足的padding
        self.contexts = self._pad_contexts(contexts)

        # --------------------------- 时间编码相关 ---------------------------
        self.time_tensor = torch.zeros(
            (self.node_num, self.walk_num, self.walk_len),
            dtype=torch.float32,
            device=self.device,
        )

        # 双频率基（增强时间特征区分度）
        self.basis_freq1 = nn.Parameter(
            torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.time_dim // 2))
            .float()
            .to(device)
        )
        self.basis_freq2 = nn.Parameter(
            torch.from_numpy(1 / np.exp(np.linspace(0, 5, self.time_dim // 2)))
            .float()
            .to(device)
        )
        self.phase = nn.Parameter(torch.zeros(self.time_dim).float().to(device))

        # 时间特征门控融合（减少噪声）
        self.time_gate = nn.Sequential(
            nn.Linear(self.time_dim, self.time_dim), nn.Sigmoid(), nn.Dropout(dropout_p)
        ).to(device)
        self.time_gate.apply(init_weights)

        # --------------------------- 位置编码相关 ---------------------------
        self.pos_tensor = torch.zeros(
            (self.node_num, self.walk_num, self.walk_len, self.walk_len),
            dtype=torch.float32,
            device=self.device,
        )

        self.trainable_embedding = nn.Sequential(
            nn.Linear(self.walk_len, self.pos_dim),
            nn.GELU(),
            nn.LayerNorm(self.pos_dim),
            nn.Linear(self.pos_dim, self.pos_dim),
        ).to(device)
        self.trainable_embedding.apply(init_weights)

        # 初始化时间/位置编码
        self._init_time_tensor()
        self._init_pos_tensor()

    def _pad_contexts(
        self, contexts: dict[int, list[list[tuple[int, float]]]]
    ) -> dict[int, list[list[tuple[int, float]]]]:
        """
        对contexts中的游走序列进行padding, 确保所有walk长度等于target_walk_len
        截断时适配双向时序游走逻辑: 从中心节点向两侧裁剪，保证中心节点位置不变
        Args:
            contexts: 原始上下文字典 {节点ID: [多轮游走列表]}
            target_walk_len: 目标游走长度
        Returns:
            经过padding的上下文字典
        """
        padded_contexts = {}
        for node_id, walk_list in contexts.items():
            padded_walk_list = []

            for walk in walk_list:
                current_len = len(walk)
                if current_len == self.walk_len:
                    # 直接保留
                    padded_walk = walk
                elif current_len < self.walk_len:
                    # 补齐，contexts已默认按时间升序排列，在开头进行padding
                    pad_num = self.walk_len - current_len
                    pad = [
                        (self.padding_node, walk[0][1]) for _ in range(pad_num)
                    ]  # 以第一步时间戳为基准进行padding
                    padded_walk = pad + walk
                else:
                    # 从中心向两侧裁剪
                    center_idx = len(walk) // 2
                    half_len = self.walk_len // 2
                    start_idx = max(0, center_idx - half_len)
                    end_idx = min(len(walk), center_idx + half_len + 1)
                    padded_walk = walk[start_idx:end_idx]
                    # 若截取后不足，再补
                    if len(padded_walk) < self.walk_len:
                        pad_num = self.walk_len - len(padded_walk)
                        base_ts = padded_walk[0][1]
                        pad = [(self.padding_node, base_ts) for _ in range(pad_num)]
                        padded_walk = pad + padded_walk

                padded_walk_list.append(padded_walk)

            padded_contexts[node_id] = padded_walk_list

        return padded_contexts

    def _init_time_tensor(self):
        # 填充归一化∆t
        for node_id, window in self.contexts.items():
            ts_batch = []
            for i in range(self.walk_num):
                walk = window[i]
                ts = torch.tensor(
                    [w[1] for w in walk], dtype=torch.float32, device=self.device
                )
                ts_batch.append(ts)

            # [walk_num, walk_len]
            ts_batch = torch.stack(ts_batch)

            # [walk_num, walk_len-1]
            time_diff = ts_batch[:, 1:] - ts_batch[:, :-1]

            # [walk_num, 1]
            time_span = (ts_batch[:, -1] - ts_batch[:, 0]).unsqueeze(-1) + 1e-8
            normed_diff = time_diff / time_span
            normed_diff = torch.clamp(normed_diff, min=-1.0, max=1.0)

            self.time_tensor[node_id, : self.walk_num, 0] = 0.0
            self.time_tensor[node_id, : self.walk_num, 1:] = normed_diff

    def _init_pos_tensor(self):
        """填充归一化位置计数"""
        node_pos_counter = {}
        for node in self.contexts:
            node_pos_counter[node] = np.zeros(self.walk_len, dtype=np.float32)
        node_pos_counter[self.padding_node] = np.zeros(self.walk_len, dtype=np.float32)

        # 统计节点在窗口中各位置的出现次数
        for _, windows in self.contexts.items():
            for walk in windows:
                for pos_idx in range(self.walk_len):
                    node, _ = walk[pos_idx]
                    node_pos_counter[node][pos_idx] += 1

        for node, windows in self.contexts.items():
            pos_batch = []
            for walk in windows:
                pos_row = np.zeros((self.walk_len, self.walk_len), dtype=np.float32)

                # 填充当前游走的位置计数（归一化）
                for pos_idx in range(self.walk_len):
                    nd, _ = walk[pos_idx]
                    pos_row[pos_idx] = node_pos_counter[nd] / self.walk_num

                pos_tensor = torch.tensor(
                    pos_row, dtype=torch.float32, device=self.device
                )
                pos_batch.append(pos_tensor)

            # [walk_num, walk_len, walk_len]
            pos_batch = torch.stack(pos_batch)
            self.pos_tensor[node, :, :, :] = pos_batch

    def _encode_time(self, nodes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            nodes: [batch] 节点ID张量
        Returns:
            [B, WN, WL, TD] 上下文时间编码特征
        """
        batch_size = nodes.shape[0]
        times_tensor = self.time_tensor[nodes]  # [B, WN, WL]

        times_tensor = times_tensor.view(batch_size, self.walk_num, self.walk_len, 1)

        # 双频率编码
        freq1 = times_tensor * self.basis_freq1.view(1, 1, 1, -1)
        freq2 = times_tensor * self.basis_freq2.view(1, 1, 1, -1)
        map_ts = torch.cat(
            [
                torch.cos(freq1 + self.phase[: self.time_dim // 2]),
                torch.sin(freq2 + self.phase[self.time_dim // 2 :]),
            ],
            dim=-1,
        )

        # 门控融合
        gate = self.time_gate(map_ts)
        map_ts = map_ts * gate + map_ts.detach() * (1 - gate)  # 残差门控

        map_ts = F.layer_norm(map_ts, map_ts.shape[1:])  # 增加层归一化
        map_ts = F.dropout(map_ts, p=self.dropout_p, training=self.training)
        return map_ts

    def _encode_pos(self, nodes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            nodes: [batch] 节点ID张量
        Returns:
            [B, WN, WL, pos_dim] 上下文位置编码特征
        """
        # [B, WN, WL, WL]
        pos_tensor_batch = self.pos_tensor[nodes]

        # 重塑维度以适配嵌入层：[B×WN×WL, WL]
        batch_shape = pos_tensor_batch.shape  # [B, WN, WL, WL]
        pos_reshaped = pos_tensor_batch.reshape(-1, self.walk_len)

        # 过嵌入层：[B×WN×WL, pos_dim]
        pos_encoded = self.trainable_embedding(pos_reshaped)

        # 恢复原始维度：[B, WN, WL, pos_dim]
        pos_encoded = pos_encoded.reshape(
            batch_shape[0], batch_shape[1], batch_shape[2], self.pos_dim
        )

        pos_encoded = F.layer_norm(pos_encoded, pos_encoded.shape[1:])
        pos_encoded = F.dropout(pos_encoded, p=self.dropout_p, training=self.training)
        return pos_encoded

    def forward(
        self, nodes: torch.Tensor, return_separate: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播：融合时间+位置编码
        Args:
            nodes: [batch] 节点ID张量
            return_separate: 是否返回分离的时间/位置特征（默认返回拼接后的融合特征）
        Returns:
            如果return_separate=True: (time_feat, pos_feat)
            否则: 拼接后的融合特征 [B, WN, WL, time_dim+pos_dim]
        """
        time_feat = self._encode_time(nodes)  # [B, WN, WL, time_dim]
        pos_feat = self._encode_pos(nodes)  # [B, WN, WL, pos_dim]

        if return_separate:
            return time_feat, pos_feat
        else:
            # 拼接融合时间+位置特征
            fusion_feat = torch.cat([time_feat, pos_feat], dim=-1)
            return fusion_feat


class ContextualFeatureAggregator(nn.Module):
    """上下文特征编码器（支持注意力聚合）"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        version: str,
        n_head: int = 8,
        rnn_type: str = "GRU",
        dropout_p: float = 0.1,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.hidden_dim_one_direction = self.hidden_dim // 2
        self.rnn_type = rnn_type.upper()
        assert self.rnn_type in ["LSTM", "GRU"], (
            f"RNN类型仅支持LSTM/GRU, 当前为{self.rnn_type}"
        )
        self.version = version
        assert self.version in ["mean", "att"], (
            f"聚合方式仅支持mean/att, 当前为{self.version}"
        )
        self.device = device
        self.rnn_input_norm = nn.LayerNorm(input_dim).to(self.device)
        self.dropout = nn.Dropout(dropout_p).to(self.device)
        self.device = device

        # 构建RNN
        if self.rnn_type == "LSTM":
            self.rnn = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=self.hidden_dim_one_direction,
                batch_first=True,
                bidirectional=True,
            ).to(self.device)
        else:
            self.rnn = nn.GRU(
                input_size=self.input_dim,
                hidden_size=self.hidden_dim_one_direction,
                batch_first=True,
                bidirectional=True,
            ).to(self.device)

        # Transformer注意力聚合（仅当version=att时启用）
        self.model_dim = self.hidden_dim_one_direction * 2
        if self.version == "att":
            self.n_head = n_head
            if self.model_dim % self.n_head != 0:
                # 自动调整model_dim为n_head的整数倍（保证注意力维度合法）
                self.model_dim = ((self.model_dim // self.n_head) + 1) * self.n_head
                # 投影层适配维度
                self.dim_adapter = nn.Linear(
                    self.hidden_dim_one_direction * 2, self.model_dim
                ).to(self.device)
                self.dim_adapter.apply(init_weights)
            else:
                self.dim_adapter = None

            # Transformer层
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.model_dim,
                nhead=self.n_head,
                dim_feedforward=4 * self.model_dim,
                dropout=dropout_p,
                activation="relu",
                batch_first=True,
                norm_first=True,
                layer_norm_eps=1e-6,
            )
            self.att_layers = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=1,
                norm=nn.LayerNorm(self.model_dim),
            ).to(self.device)

            # 3. 增强注意力分数学习（多头注意力+门控机制）
            # 替换原线性层为多头注意力分数计算
            self.att_score_layer = nn.Sequential(
                nn.Linear(self.model_dim, self.model_dim),
                nn.Tanh(),  # 非线性激活增强分数区分度
                nn.Linear(self.model_dim, 1, bias=False),
            ).to(self.device)
            self.att_score_layer.apply(init_weights)

            # 4. 注意力权重平滑系数（防止权重过于集中）
            self.att_smoothing = 0.1

            # 5. 增强投影层（残差连接+更优的激活）
            self.att_proj = nn.Sequential(
                nn.Linear(self.model_dim, self.model_dim),
                nn.LayerNorm(self.model_dim),
                nn.GELU(),
                nn.Dropout(dropout_p * 0.5),
                nn.Linear(self.model_dim, self.model_dim),
            ).to(self.device)
            self.att_proj.apply(init_weights)

            # 6. 残差连接（防止梯度消失）
            self.residual_proj = (
                nn.Linear(self.model_dim, self.model_dim)
                if self.model_dim != self.hidden_dim
                else nn.Identity()
            )
        else:
            self.att_layers = None
            self.att_score_layer = None
            self.att_proj = None
            self.dim_adapter = None
            self.residual_proj = None

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X: [batch, walk_num, walk_len, dim] 输入特征
        Returns:
            [batch, model_dim] 聚合后的特征
        """
        batch, walk_num, walk_len, dim = X.shape
        # 重塑维度以适配RNN
        X_reshaped = X.reshape(batch * walk_num, walk_len, dim)
        X_reshaped = self.rnn_input_norm(X_reshaped)  # RNN输入归一化
        encoded_all, _ = self.rnn(X_reshaped)  # [batch×seq_num, cs_len, model_dim]

        # 均值聚合
        encoded_mean = encoded_all.mean(dim=1)  # [batch×seq_num, model_dim]
        ft = encoded_mean.reshape(batch, walk_num, -1)  # [batch, walk_num, model_dim]

        # 最终聚合
        if self.version == "mean":
            output = torch.mean(ft, dim=1)
        else:
            # 维度适配（如果需要）
            if self.dim_adapter is not None:
                ft = self.dim_adapter(ft)

            att_output = self.att_layers(ft)  # [batch, walk_num, model_dim]
            att_scores = self.att_score_layer(att_output)  # [batch, walk_num, 1]
            # 权重平滑：防止单一样本权重占比过高
            att_weights = F.softmax(att_scores / self.att_smoothing, dim=1)
            # 权重归一化（可选，提升稳定性）
            att_weights = att_weights / (att_weights.sum(dim=1, keepdim=True) + 1e-8)

            # 调试信息
            # if self.training and torch.rand(1).item() < 0.05:
            #     sample_weights = att_weights[0, :, 0].detach().cpu().numpy()
            #     logger.info(f"注意力权重（第一个样本）: {sample_weights.round(3)}")
            #     logger.info(f"权重最大值位置: {sample_weights.argmax()}, 最大值: {sample_weights.max():.3f}")
            #     logger.info(f"权重标准差: {sample_weights.std():.3f}（>0.1表示权重有区分度）")

            # 注意力聚合
            output = (att_output * att_weights).sum(dim=1)  # [batch, model_dim]

            # 残差连接
            residual = self.residual_proj(ft.mean(dim=1))
            output = output + residual
            output = self.att_proj(output)

        output = self.dropout(output)
        return output


class MergeLayer(nn.Module):
    """
    特征融合层（链接预测场景：源/目标节点特征融合→分数输出）
    """

    def __init__(
        self,
        src_dim: int,
        tgt_dim: int,
        hidden_dim: int,
        out_dim: int = 1,
        dropout_p: float = 0.1,
        activation: nn.Module = nn.ReLU(),
        use_layer_norm: bool = True,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.use_layer_norm = use_layer_norm
        if self.use_layer_norm:
            self.ln1 = nn.LayerNorm(hidden_dim).to(device)
        self.fc1 = nn.Linear(src_dim + tgt_dim, hidden_dim).to(device)
        self.fc2 = nn.Linear(hidden_dim, out_dim).to(device)
        self.act = activation.to(device)
        self.dropout = nn.Dropout(dropout_p).to(device)
        self.apply(init_weights)

    def forward(
        self,
        src_embed: torch.Tensor,
        tgt_embed: torch.Tensor,
        return_hidden: bool = False,  # 返回隐藏层特征（用于可视化/下游任务）
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播：融合源/目标节点特征，输出链接预测分数
        Args:
            src_embed: [B, src_dim] 源节点特征
            tgt_embed: [B, tgt_dim] 目标节点特征
            return_hidden: 是否返回隐藏层特征（默认仅返回输出）
        Returns:
            若return_hidden=False: [B, out_dim] 链接预测分数
            若return_hidden=True: ([B, out_dim], [B, hidden_dim]) 分数 + 隐藏层特征
        """
        x = torch.cat([src_embed, tgt_embed], dim=-1)  # [B, src_dim + tgt_dim]

        h = self.fc1(x)  # [B, hidden_dim]
        if self.use_layer_norm:
            h = self.ln1(h)
        h = self.act(h)
        h = self.dropout(h)

        z = self.fc2(h)  # [B, out_dim]

        if return_hidden:
            return z, h
        return z


class IPNet(nn.Module):
    def __init__(
        self,
        node_feature: np.ndarray,
        interactions: dict[int, list[tuple[int, float]]],
        contexts: dict[int, list[list[tuple[int, float]]]],
        final_seq_len: int | None = None,  # 指定的序列长度
        final_walk_len: int | None = None,  # 指定的游走/窗口长度
        version: str = "mean",
        rnn_type: str = "GRU",
        n_head: int = 8,
        dropout_p: float = 0.3,
        padding_node: int = 0,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        # 基础配置
        self.rnn_type = rnn_type.upper()
        self.version = version
        self.dropout_p = dropout_p
        self.padding_node = padding_node
        self.n_head = n_head
        self.device = device
        self.node_feature = node_feature
        self.interactions = interactions
        self.contexts = contexts
        self.seq_len = final_seq_len
        self.walk_len = final_walk_len
        # 节点特征初始化
        self.node_num = self.node_feature.shape[0]
        self.feat_dim = self.node_feature.shape[1]
        if self.version == "w2v":
            node_feat_tensor = torch.from_numpy(node_feature.astype(np.float32)).to(
                device
            )
            node_feat_tensor[self.padding_node] = 0.0
            self.node_embed = nn.Embedding.from_pretrained(
                embeddings=node_feat_tensor,
                padding_idx=self.padding_node,  # 屏蔽padding节点的梯度更新
                freeze=False,
            ).to(device)

        # 维度计算
        assert self.feat_dim % 2 == 0, f"节点特征维度需为偶数，当前为{self.feat_dim}"
        self.time_dim = self.feat_dim // 2
        self.pos_dim = self.feat_dim // 2
        self.model_dim = self.feat_dim
        self.out_dim = self.feat_dim

        # --------------------------- 1. 交互模式学习 ---------------------------
        self.interaction_encoder = InteractionSequenceEncoder(
            node_num=self.node_num,
            time_dim=self.time_dim,
            pos_dim=self.pos_dim,
            node_interactions=self.interactions,
            specified_seq_len=self.seq_len,
            padding_node=self.padding_node,
            dropout_p=dropout_p,
            device=device,
        )

        self.sequence_aggregator = SequenceFeatureAggregator(
            input_dim=self.model_dim,
            hidden_dim=self.model_dim,
            rnn_type=self.rnn_type,
            dropout_p=self.dropout_p,
            device=device,
        )

        # --------------------------- 2. 时序上下文建模 ---------------------------
        # w2v版本不需要时序上下文建模，forward函数中直接拼接word2vec特征
        if self.version in ["mean", "att"]:
            self.ctx_encoder = ContextEncoder(
                nodes_num=self.node_num,
                time_dim=self.time_dim,
                pos_dim=self.pos_dim,
                contexts=self.contexts,
                specified_walk_len=self.walk_len,
                padding_node=self.padding_node,
                dropout_p=self.dropout_p,
                device=device,
            )

            self.ctx_aggregator = ContextualFeatureAggregator(
                input_dim=self.model_dim,
                hidden_dim=self.model_dim,
                version=self.version,
                rnn_type=self.rnn_type,
                dropout_p=self.dropout_p,
                n_head=n_head,
                device=device,
            )

        # --------------------------- 3. 特征融合层 ---------------------------
        self.merge_layer = MergeLayer(
            src_dim=self.out_dim * 2,
            tgt_dim=self.out_dim * 2,
            hidden_dim=self.feat_dim,
            out_dim=1,
            dropout_p=self.dropout_p,
            device=device,
        )

        # 模型信息打印
        self._print_model_info()

    def _print_model_info(self):
        # fmt: off
        logger.info("✅ IPNet模型初始化完成")
        logger.info(f"   ├─ 节点数: {self.node_num}(内置虚拟节点{self.padding_node}) | 节点维度: {self.feat_dim}")
        logger.info(f"   ├─ 交互序列长度: {self.seq_len}")
        if self.version in ["mean", "att"]:
            logger.info(f"   ├─ 上下文编码: 窗口数量: {self.ctx_encoder.walk_num} | 窗口长度: {self.walk_len} | 聚合方式: {self.version}")
        else:
            logger.info(f"   ├─ 上下文编码 - 聚合方式: {self.version}")
        if self.version == "att":
            logger.info(f"   ├─ 注意力头数: {self.n_head}")
        logger.info(f"   ├─ 时间编码维度: {self.time_dim} | 位置编码维度: {self.pos_dim}")
        logger.info(f"   ├─ RNN: {self.rnn_type}")
        logger.info(f"   ├─ Dropout: {self.dropout_p}")
        logger.info(f"   └─ Device: {self.device}")
        # fmt: on

    def forward(self, edges: torch.Tensor, return_logits: bool = False) -> torch.Tensor:
        """
        前向传播
        Args:
            edges: [batch, 2] 边张量
            return_logits: 是否返回原始logits(计算loss), False则返回sigmoid概率
        Returns:
            [batch] logits或概率值
        """
        if isinstance(edges, np.ndarray):
            edges = torch.from_numpy(edges).to(self.device)
        elif not isinstance(edges, torch.Tensor):
            raise TypeError(f"edges必须是numpy数组或torch张量, 当前类型: {type(edges)}")

        src_nodes = edges[:, 0]
        tgt_nodes = edges[:, 1]

        src_embed = self.forward_msg(src_nodes)
        tgt_embed = self.forward_msg(tgt_nodes)

        score_logits = self.merge_layer(src_embed, tgt_embed).squeeze(dim=-1)

        if return_logits:
            return score_logits  # 返回logits计算loss
        else:
            return score_logits.sigmoid()  # 返回概率

    def forward_msg(self, nodes: torch.Tensor) -> torch.Tensor:
        """
        节点特征编码
        Args:
            nodes: [batch] 节点ID张量
        Returns:
            [batch, out_dim*2] 编码后的节点特征
        """
        # --------------------------- 1. 交互模式特征 ---------------------------
        interaction_feat = self.interaction_encoder(nodes)  # [B, L, time_dim+pos_dim]
        interaction_feat = self.sequence_aggregator(interaction_feat)  # [B, model_dim]

        # --------------------------- 2. 上下文高阶特征 ---------------------------
        if self.version == "w2v":
            ctx_feat = self.node_embed(nodes)
        else:
            ctx_feat = self.ctx_encoder(nodes)  # [B, WN, WL, time_dim+pos_dim]
            ctx_feat = self.ctx_aggregator(ctx_feat)  # [B, model_dim]

        # Final Embeds
        return torch.cat([interaction_feat, ctx_feat], dim=-1)

    def save_init_param(self, param_save_config: dict) -> str:
        """
        保存模型初始化参数(供后续加载模型使用)
        """
        base_dir = param_save_config["dir"]
        os.makedirs(base_dir, exist_ok=True)
        try:
            # 1. 保存节点特征（npy格式）
            node_feature_path = os.path.join(
                base_dir, param_save_config["node_feature"]
            )
            np.save(node_feature_path, self.node_feature)

            # tuple => list 的递归函数（解决JSON无法序列化tuple的问题）
            def convert_tuple_to_list(obj):
                if isinstance(obj, tuple):
                    return list(obj)
                elif isinstance(obj, dict):
                    return {str(k): convert_tuple_to_list(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_tuple_to_list(i) for i in obj]
                return obj

            # 2. 保存交互序列和上下文窗口(JSON格式)
            interactions_path = os.path.join(
                base_dir, param_save_config["interactions"]
            )
            with open(interactions_path, "w", encoding="utf-8") as f:
                json.dump(
                    convert_tuple_to_list(self.interactions),
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

            contexts_path = os.path.join(base_dir, param_save_config["contexts"])
            with open(contexts_path, "w", encoding="utf-8") as f:
                json.dump(
                    convert_tuple_to_list(self.contexts),
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

            # 3. 保存其他参数
            model_config = DEFAULT_MODEL_CONFIG.copy()
            model_config["VERSION"] = self.version
            model_config["RNN_TYPE"] = self.rnn_type
            model_config["PADDING_NODE"] = self.padding_node
            model_config["N_HEAD"] = self.n_head
            model_config["DROPOUT"] = self.dropout_p
            model_config["FINAL_SEQ_LEN"] = self.seq_len
            model_config["FINAL_WALK_LEN"] = self.walk_len

            other_params_path = os.path.join(
                base_dir, param_save_config["other_params"]
            )

            with open(other_params_path, "w", encoding="utf-8") as f:
                json.dump(model_config, f, indent=2, ensure_ascii=False)

            logger.info("✅ IPNet模型初始化参数保存完成")
            logger.info(f"   ├─ 节点特征已保存至: {node_feature_path}")
            logger.info(f"   ├─ 交互序列已保存至: {interactions_path}")
            logger.info(f"   ├─ 上下文窗口已保存至: {contexts_path}")
            logger.info(f"   └─ 其他初始化参数已保存至: {other_params_path}")

        except OSError as e:
            raise OSError(f"保存初始化配置失败: {str(e)}") from e
