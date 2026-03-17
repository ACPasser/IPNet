import os
import time
import csv
import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from tqdm import tqdm
from model.IPNet import IPNet
from datetime import datetime
from gensim.models import Word2Vec
from data.config import MODEL_DEFAULT_CONFIG
from data.data_utils import build_nx_graph_from_config, negative_sampling
from data.data_loader import DataLoader
from data.preprocess import preprocess
from model.model_utils import (
    set_random_seeds,
    get_device,
    get_result_path,
    EarlyStopMonitor,
)

logger = logging.getLogger(__name__)


def run_training(model_config: dict, data_config: dict) -> dict:
    """
    核心训练函数：接收配置字典，完成数据加载、模型训练、测试、结果保存
    Args:
        config: 训练配置字典(DEFAULT_MODEL_CONFIG)
    Returns:
        dict: 包含测试指标、模型路径、耗时等结果
    """
    # 以默认配置为基础，用自定义配置覆盖需要修改的参数
    final_config = MODEL_DEFAULT_CONFIG.copy()
    final_config.update(model_config)
    cfg = final_config  # 简化变量名

    # 配置校验
    if cfg["PADDING_NODE"] < 0:
        raise ValueError(
            f"Padding node ID must be non-negative! Current: {cfg['PADDING_NODE']}"
        )
    if cfg["TASK_TYPE"] == "I" and not (0 < cfg["MASK_RATIO"] < 1):
        raise ValueError(
            f"Inductive task requires mask_ratio in (0,1)! Current: {cfg['MASK_RATIO']}"
        )
    if cfg["TASK_TYPE"] == "I":
        mask_str = str(cfg["MASK_RATIO"])
        if "." in mask_str and len(mask_str.split(".")[1]) > 2:
            raise ValueError(
                f"mask_ratio must have at most 2 decimal places! Current: {cfg['MASK_RATIO']}"
            )

    if cfg["PRE_PROCESS"]:
        try:
            preprocess(dataset_name=cfg["DATASET"])
        except Exception as e:
            logger.error(f"❌ 预处理数据集失败: {str(e)}", exc_info=True)
            raise  # 预处理失败则终止训练

    # 设置随机种子
    set_random_seeds(cfg["SEED"])
    # Device
    device = get_device(cfg["DEVICE"])
    # 保存路径(模型检查点、最佳模型)
    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    best_model_path = os.path.join(
        current_dir,
        cfg["BEST_MODEL_PATH"].format(dataset=cfg["DATASET"], timestamp=timestamp),
    )

    ckpt_dir = os.path.join(
        current_dir,
        cfg["CHECKPOINT_DIR"].format(dataset=cfg["DATASET"], timestamp=timestamp),
    )

    # 1. 数据加载与预处理
    start_time = time.time()
    graph = build_nx_graph_from_config(data_config)
    num_nodes = len(graph.nodes())
    data_loader = DataLoader(graph)

    # 2. 交互序列提取
    worker_num = min(cfg["THREAD_NUM"], os.cpu_count() or 1)
    interactions, avg_seq_len = data_loader.extract_interaction_seqs(cfg["TASK_TYPE"])
    if cfg["IS_LEN"] == -1:
        final_seq_len = avg_seq_len
        logger.info(f"📏 模型采纳推荐的平均序列长度: {final_seq_len}")
    else:
        final_seq_len = cfg["IS_LEN"]
        logger.info(f"📏 模型使用指定的交互序列长度: {final_seq_len}")

    # 3. 上下文窗口提取
    final_walk_num = cfg["WALK_NUM"]
    final_walk_len = cfg["WALK_LEN"]
    need_estimate = (cfg["WALK_NUM"] == -1) or (cfg["WALK_LEN"] == -1)
    if need_estimate:
        recommend_walk_num, recommend_walk_len = data_loader.estimate_walk_params(
            task_type=cfg["TASK_TYPE"]
        )
        param_updates = []
        if cfg["WALK_NUM"] == -1:
            final_walk_num = recommend_walk_num
            param_updates.append(f"游走次数: {final_walk_num}")
        if cfg["WALK_LEN"] == -1:
            final_walk_len = (
                recommend_walk_len * 2
                if cfg["VERSION"] == "w2v"
                else recommend_walk_len
            )
            param_updates.append(f"游走长度: {final_walk_len}")
        if param_updates:
            logger.info(f"📏 模型采纳推荐的游走参数：{', '.join(param_updates)}")
    else:
        logger.info(
            f"📏 模型使用指定的游走参数：游走次数: {final_walk_num}, 游走长度: {final_walk_len}"
        )

    contexts = data_loader.extract_ctx_window(
        cfg["TASK_TYPE"],
        walk_num=final_walk_num,
        walk_len=final_walk_len,
        workers=worker_num,
    )

    # 4. 节点特征初始化
    node_feature = np.random.randn(num_nodes + 1, cfg["FEAT_DIM"]).astype(np.float32)
    if cfg["VERSION"] == "w2v":
        ctx = []
        for windows in contexts.values():
            ctx.extend([[w[0] for w in walk] for walk in windows])
        word2vec = Word2Vec(
            ctx,
            vector_size=cfg["FEAT_DIM"],
            workers=worker_num,
            window=final_walk_len,
            min_count=final_seq_len,
            batch_words=cfg["BATCH_SIZE"],
            negative=final_seq_len,
            seed=cfg["SEED"],
        )
        logger.info("✅ Word2Vec训练完成")
        logger.info(
            f"   ├─ 节点数: {len(word2vec.wv.index_to_key)} | 特征维度: {word2vec.wv.vector_size}"
        )
        logger.info(f"   └─ 游走次数: {final_walk_num} | 游走长度: {final_walk_len}")

        for node in word2vec.wv.index_to_key:
            node_feature[node] = word2vec.wv.get_vector(node)

    # 5. 模型初始化
    model = IPNet(
        node_feature=node_feature,
        interactions=interactions,
        contexts=contexts,
        ckpt_dir=ckpt_dir,
        specified_seq_len=final_seq_len,
        specified_walk_len=final_walk_len,
        version=cfg["VERSION"],
        rnn_type=cfg["RNN_TYPE"],
        padding_node=cfg["PADDING_NODE"],
        device=device,
    )
    model.to(device)

    # 6. 训练
    train_data, val_data, test_data = data_loader.preprocess(
        cfg["TASK_TYPE"], cfg["MASK_RATIO"], cfg["SEED"]
    )
    logger.info("🚀 Start training...")
    train(
        model=model,
        train_data=train_data,
        val_data=val_data,
        batch_size=cfg["BATCH_SIZE"],
        epochs=cfg["EPOCH"],
        lr=cfg["LR"],
        seed=cfg["SEED"],
        best_model_path=best_model_path,
    )

    # 7. 测试
    test_neg = np.array(
        negative_sampling(
            test_data["nodes"],
            test_data["adj"],
            test_data["neg_num"],
            hard_ratio=0.0,
            seed=cfg["SEED"],
        )
    )
    np.random.shuffle(test_neg)
    test_acc, test_ap, test_f1, test_auc = eval_one_epoch(
        model, test_data["edges"], test_neg, cfg["BATCH_SIZE"]
    )

    # 测试结果打印优化版
    logger.info("=" * 60)  # 醒目分隔线，突出测试结果环节
    logger.info("📊 Final Test Results (最终测试指标)")
    logger.info("┌─────────────┬──────────┬───────────┐")
    logger.info("│   Metric    │  Value   │  Percent  │")
    logger.info("├─────────────┼──────────┼───────────┤")
    # 格式化输出每个指标，左对齐名称，右对齐数值，补充百分比（分类任务更直观）
    logger.info(f"│  Test Acc   │ {test_acc:>8.4f} │ {test_acc * 100:>8.2f}% │")
    logger.info(f"│  Test AUC   │ {test_auc:>8.4f} │ {test_auc * 100:>8.2f}% │")
    logger.info(f"│  Test AP    │ {test_ap:>8.4f} │ {test_ap * 100:>8.2f}% │")
    logger.info(f"│  Test F1    │ {test_f1:>8.4f} │ {test_f1 * 100:>8.2f}% │")
    logger.info("└─────────────┴──────────┴───────────┘")

    # 8. 结果保存
    execution_time = time.time() - start_time
    task_type = cfg["TASK_TYPE"]
    task_desc = "Transductive (直推式)" if task_type == "T" else "Inductive (归纳式)"

    # 核心完成提示
    logger.info(
        f"✅ 训练完成! | {task_desc} | 数据集: {cfg['DATASET']} | 模型版本: {cfg['VERSION']} | 耗时: {execution_time:.2f}s ({execution_time / 60:.2f}min)"
    )
    result_path = get_result_path(cfg, final_seq_len, final_walk_num, final_walk_len)
    logger.info(f"📁 结果文件保存至：{result_path}")
    logger.info("=" * 60)

    need_header = not os.path.exists(result_path)
    with open(result_path, "a+", newline="") as f:
        writer = csv.writer(f)
        if need_header:
            writer.writerow(
                [
                    "Training_Date",
                    "Task_Type",
                    "Acc",
                    "AUC",
                    "AP",
                    "F1",
                    "Time(s)",
                    "Seed",
                    "Best_Model_Path",
                ]
            )
        writer.writerow(
            [
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                cfg["TASK_TYPE"],
                test_acc * 100,
                test_auc * 100,
                test_ap * 100,
                test_f1 * 100,
                execution_time,
                cfg["SEED"],
                best_model_path,
            ]
        )

    # 返回训练结果
    return {
        "test_acc": test_acc,
        "test_auc": test_auc,
        "test_ap": test_ap,
        "test_f1": test_f1,
        "execution_time": execution_time,
        "best_model_path": best_model_path,
        "config": final_config,
    }


def train(
    model: IPNet,
    train_data: dict,
    val_data: dict,
    batch_size: int = 64,
    epochs: int = 50,
    lr: float = 1e-4,
    seed=None,
    best_model_path: str = None,
) -> None:
    """
    模型训练函数
    Args:
        model: 待训练的IPNet模型
        train_data: 训练图数据(字典格式, 包含edges、nodes、adj等键值对)
        val_data: 验证图数据(字典格式, 包含edges、nodes、adj等键值对)
        batch_size: 批次大小
        epochs: 训练轮数
        lr: 学习率
        seed: 随机种子
        best_model_path: 最佳模型保存路径
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.8, patience=1
    )

    early_stopper = EarlyStopMonitor(max_round=1)

    train_edges = train_data["edges"]
    train_nodes = train_data["nodes"]
    train_adj = train_data["adj"]
    num_instances = train_data["pos_num"]

    num_instances = len(train_edges)
    num_batches = math.ceil(num_instances / batch_size)
    idx_list = np.arange(num_instances)
    is_early_stopped = False

    for epoch in range(epochs):
        epoch_metrics = {"acc": [], "ap": [], "auc": [], "f1": [], "loss": []}
        np.random.shuffle(idx_list)  # shuffle for every epoch

        for batch_idx in tqdm(range(num_batches), desc=f"Epoch {epoch + 1}/{epochs}"):
            start_idx = batch_idx * batch_size
            end_idx = min(num_instances, start_idx + batch_size)
            batch_indices = idx_list[start_idx:end_idx]
            train_pos_edges = train_edges[batch_indices]
            curr_batch_size = len(train_pos_edges)
            train_neg_edges = np.array(
                negative_sampling(train_nodes, train_adj, curr_batch_size, seed=seed)
            )
            np.random.shuffle(train_neg_edges)

            model.train()

            # forward
            optimizer.zero_grad()

            all_edges = np.concatenate([train_pos_edges, train_neg_edges], axis=0)
            all_logits = model(all_edges, return_logits=True)

            # Label Smoothing
            pos_label = torch.ones(curr_batch_size, device=all_logits.device) * 0.9
            neg_label = torch.zeros(curr_batch_size, device=all_logits.device) + 0.1
            all_labels = torch.cat([pos_label, neg_label], dim=0)

            # loss
            loss = F.binary_cross_entropy_with_logits(all_logits, all_labels)

            # backward
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # metrics
            with torch.no_grad():
                model.eval()
                # 对logits做sigmoid得到概率
                pred_score = torch.sigmoid(all_logits).cpu().numpy()
                true_label = np.concatenate(
                    [np.ones(curr_batch_size), np.zeros(curr_batch_size)]
                )
                pred_label = pred_score > 0.5

                # 更新指标
                epoch_metrics["acc"].append((pred_label == true_label).mean())
                epoch_metrics["ap"].append(
                    average_precision_score(true_label, pred_score)
                )
                epoch_metrics["auc"].append(roc_auc_score(true_label, pred_score))
                epoch_metrics["f1"].append(f1_score(true_label, pred_label))
                epoch_metrics["loss"].append(loss.item())

                # 训练阶段，打印前3个batch的pred_score分布（仅调试用）
                # if batch_idx < 3:  # 只打印前3个batch
                #     logger.info(f"=== Epoch {epoch+1} Batch {batch_idx+1} Pred Score ===")
                #     # 正样本预测概率（前10个）
                #     pos_pred = pred_score[:10]
                #     logger.info(f"正样本预测概率（前10个）: {np.round(pos_pred, 4)} | 均值: {np.mean(pos_pred):.4f}")
                #     # 负样本预测概率（前10个）
                #     neg_pred = pred_score[-10:]
                #     logger.info(f"负样本预测概率（前10个）: {np.round(neg_pred, 4)} | 均值: {np.mean(neg_pred):.4f}")
                #     # 正负样本概率重叠度（越小越好）
                #     overlap = np.min(pos_pred) - np.max(neg_pred)
                #     logger.info(f"正负样本概率重叠度（正最小-负最大）: {overlap:.4f}")

        # validation phase
        val_edges_pos = val_data["edges"]
        val_nodes = val_data["nodes"]
        val_adj = val_data["adj"]
        val_edges_neg = np.array(
            negative_sampling(val_nodes, val_adj, val_data["neg_num"], seed=seed)
        )
        np.random.shuffle(val_edges_neg)
        val_acc, val_ap, val_f1, val_auc = eval_one_epoch(
            model, val_edges_pos, val_edges_neg, batch_size
        )
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_auc)
        # if epoch < 5 or (epoch + 1) % 5 == 0:  # 前5轮必打，之后每5轮打一次
        #     with torch.no_grad():
        #         model.eval()
        #         # 取验证集前10个正样本+前10个负样本
        #         val_sample_pos = val_edges_pos[:10]
        #         val_sample_neg = val_edges_neg[:10]
        #         val_sample_all = np.concatenate(
        #             [val_sample_pos, val_sample_neg], axis=0
        #         )

        #         # 预测分数
        #         val_sample_logits = model(val_sample_all, return_logits=True)
        #         val_sample_scores = torch.sigmoid(val_sample_logits).cpu().numpy()

        #         # DEBUG: 打印验证阶段的预测分数分布，观察正负样本的区分度
        #         logger.info(f"=== Epoch {epoch + 1} Validation Phase Analysis ===")
        #         # 正样本预测
        #         logger.info(f"Val正样本节点对（前10个）: {val_sample_pos.tolist()}")
        #         val_sample_logits_np = val_sample_logits.cpu().numpy()
        #         pos_logits = val_sample_logits_np[:10]
        #         logger.info(
        #             f"Val正样本原始logits（前10个）: {np.round(pos_logits, 4)} | 均值: {np.mean(pos_logits):.4f}"
        #         )
        #         pos_pred = val_sample_scores[:10]
        #         logger.info(
        #             f"Val正样本预测概率（前10个）: {np.round(pos_pred, 4)} | 均值: {np.mean(pos_pred):.4f}"
        #         )

        #         # 负样本预测
        #         neg_pred = val_sample_scores[-10:]
        #         neg_pred = val_sample_scores[-10:]
        #         logger.info(f"Val负样本节点对（前10个）: {val_sample_neg.tolist()}")
        #         logger.info(
        #             f"Val负样本预测概率（前10个）: {np.round(neg_pred, 4)} | 均值: {np.mean(neg_pred):.4f}"
        #         )
        #         # 正负样本概率重叠度（>0表示区分度好，<0表示有混淆）
        #         overlap = np.min(pos_pred) - np.max(neg_pred)
        #         logger.info(
        #             f"Val正负样本概率重叠度（正最小-负最大）: {overlap:.4f} (＞0=区分度好)"
        #         )

        logger.info(f"=== Epoch {epoch + 1} Results (lr: {current_lr:.6f}) ===")
        logger.info(f"Train Loss: {np.mean(epoch_metrics['loss']):.4f}")
        logger.info(
            f"Train Acc: {np.mean(epoch_metrics['acc']):.4f} | Val Acc: {val_acc:.4f}"
        )
        logger.info(
            f"Train AUC: {np.mean(epoch_metrics['auc']):.4f} | Val AUC: {val_auc:.4f}"
        )
        logger.info(
            f"Train AP: {np.mean(epoch_metrics['ap']):.4f} | Val AP: {val_ap:.4f}"
        )
        logger.info(
            f"Train F1: {np.mean(epoch_metrics['f1']):.4f} | Val F1: {val_f1:.4f}"
        )

        # save checkpoint
        model.save_checkpoint(epoch, optimizer, np.mean(epoch_metrics["loss"]))

        # early stop check
        if early_stopper.early_stop_check(val_auc):
            is_early_stopped = True
            break

    if best_model_path is not None:
        best_model_dir = os.path.dirname(best_model_path)
        if best_model_dir:
            os.makedirs(best_model_dir, exist_ok=True)
        model.load_checkpoint(epoch=early_stopper.best_epoch)
        torch.save(model.state_dict(), best_model_path)

        if is_early_stopped:
            stop_type = f"早停（连续{early_stopper.max_round}轮无性能提升）"
        else:
            stop_type = f"训练完成（完成全部{epochs}轮训练）"

        logger.info(f"✅ 模型训练结束: {stop_type}")
        logger.info(f"   ├─ 最优轮次: {early_stopper.best_epoch + 1}")
        logger.info(f"   └─ 最优模型已保存至: {best_model_path}")


def eval_one_epoch(
    model: IPNet,
    pos_edges: np.ndarray,
    neg_edges: np.ndarray,
    batch_size: int = 64,
) -> tuple[float, float, float, float]:
    """
    单轮验证/测试函数（无梯度计算，纯推理）
    Args:
        model: 已训练的IPNet模型
        pos_edges: 正样本边数据
        neg_edges: 负采样边数据
        batch_size: 批次大小
    Returns:
        平均准确率、平均AP、平均F1、平均AUC
    """
    model.eval()
    all_pred_scores = []

    all_test_edges = np.concatenate([pos_edges, neg_edges], axis=0)
    all_test_labels = np.concatenate(
        [np.ones(len(pos_edges)), np.zeros(len(neg_edges))]
    )

    num_instances = len(all_test_edges)
    num_batches = math.ceil(num_instances / batch_size)
    with torch.no_grad():
        for i in range(num_batches):
            batch_edges = all_test_edges[i * batch_size : (i + 1) * batch_size]
            logits = model(batch_edges, return_logits=True)
            scores = torch.sigmoid(logits).cpu().numpy()
            all_pred_scores.append(scores)

    all_pred_scores = np.concatenate(all_pred_scores)
    all_pred_labels = all_pred_scores > 0.5

    # metrics
    acc = (all_pred_labels == all_test_labels).mean()
    ap = average_precision_score(all_test_labels, all_pred_scores)
    f1 = f1_score(all_test_labels, all_pred_labels)
    auc = roc_auc_score(all_test_labels, all_pred_scores)

    return acc, ap, f1, auc
