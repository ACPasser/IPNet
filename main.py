import argparse
import csv
import math
import os
import random
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gensim.models import Word2Vec
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sympy import true
from tqdm import tqdm

from data.config import get_config
from data_loader import DataLoader
from IPNet import IPNet
from utils import EarlyStopMonitor, build_nx_graph_from_config, negative_sampling


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
        print(f"⚠️ 部分算子不支持确定性模式：{e}，已降级为警告模式")

    # print(f"✅ Random seeds set to: {seed} (deterministic mode enabled)")


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

    early_stopper = EarlyStopMonitor(max_round=3)

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
                #     print(f"\n=== Epoch {epoch+1} Batch {batch_idx+1} Pred Score ===")
                #     # 正样本预测概率（前10个）
                #     pos_pred = pred_score[:10]
                #     print(f"正样本预测概率（前10个）: {np.round(pos_pred, 4)} | 均值: {np.mean(pos_pred):.4f}")
                #     # 负样本预测概率（前10个）
                #     neg_pred = pred_score[-10:]
                #     print(f"负样本预测概率（前10个）: {np.round(neg_pred, 4)} | 均值: {np.mean(neg_pred):.4f}")
                #     # 正负样本概率重叠度（越小越好）
                #     overlap = np.min(pos_pred) - np.max(neg_pred)
                #     print(f"正负样本概率重叠度（正最小-负最大）: {overlap:.4f}")

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
        #         print(f"\n=== Epoch {epoch + 1} Validation Phase Analysis ===")
        #         # 正样本预测
        #         print(f"Val正样本节点对（前10个）: {val_sample_pos.tolist()}")
        #         val_sample_logits_np = val_sample_logits.cpu().numpy()
        #         pos_logits = val_sample_logits_np[:10]
        #         print(
        #             f"Val正样本原始logits（前10个）: {np.round(pos_logits, 4)} | 均值: {np.mean(pos_logits):.4f}"
        #         )
        #         pos_pred = val_sample_scores[:10]
        #         print(
        #             f"Val正样本预测概率（前10个）: {np.round(pos_pred, 4)} | 均值: {np.mean(pos_pred):.4f}"
        #         )

        #         # 负样本预测
        #         neg_pred = val_sample_scores[-10:]
        #         neg_pred = val_sample_scores[-10:]
        #         print(f"Val负样本节点对（前10个）: {val_sample_neg.tolist()}")
        #         print(
        #             f"Val负样本预测概率（前10个）: {np.round(neg_pred, 4)} | 均值: {np.mean(neg_pred):.4f}"
        #         )
        #         # 正负样本概率重叠度（>0表示区分度好，<0表示有混淆）
        #         overlap = np.min(pos_pred) - np.max(neg_pred)
        #         print(
        #             f"Val正负样本概率重叠度（正最小-负最大）: {overlap:.4f} (＞0=区分度好)"
        #         )

        print(f"\n=== Epoch {epoch + 1} Results (lr: {current_lr:.6f}) ===")
        print(f"Train Loss: {np.mean(epoch_metrics['loss']):.4f}")
        print(
            f"Train Acc: {np.mean(epoch_metrics['acc']):.4f} | Val Acc: {val_acc:.4f}"
        )
        print(
            f"Train AUC: {np.mean(epoch_metrics['auc']):.4f} | Val AUC: {val_auc:.4f}"
        )
        print(f"Train AP: {np.mean(epoch_metrics['ap']):.4f} | Val AP: {val_ap:.4f}")
        print(f"Train F1: {np.mean(epoch_metrics['f1']):.4f} | Val F1: {val_f1:.4f}")

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

        print(f"\n✅ 模型训练结束: {stop_type}")
        print(f"   ├─ 最优轮次: {early_stopper.best_epoch + 1}")
        print(f"   └─ 最优模型已保存至: {best_model_path}")


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
        print("✅ Forced to use device: CPU")

    elif gpu_flag == -1:
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps")
            print("📌 Using device: Apple MPS (Apple Silicon acceleration)")
        else:
            device = torch.device("cpu")
            print(
                "⚠️ MPS not available (need PyTorch ≥2.0 + Apple Silicon), fallback to CPU"
            )

    else:
        if torch.cuda.is_available():
            # 如果传入具体卡号（如0/1），则使用对应GPU；否则默认用第0卡
            device = torch.device(f"cuda:{gpu_flag}" if gpu_flag >= 0 else "cuda:0")
            print(f"📌 Using device: NVIDIA GPU (cuda:{gpu_flag})")
        else:
            device = torch.device("cpu")
            print("⚠️ NVIDIA GPU not available, fallback to CPU")

    return device


def main(args: argparse.Namespace) -> None:
    """
    主函数：数据加载 => 提取交互序列和上下文窗口 => 训练、测试模型 => 保存结果
    """
    task_type_name = (
        "Transductive (直推式)" if args.TASK_TYPE == "T" else "Inductive (归纳式)"
    )
    print(
        f"\n📌 数据集: {args.DATASET} | 任务类型: {task_type_name} | 模型版本: {args.VERSION} | 种子: {args.SEED}"
    )

    # 设置随机种子
    set_random_seeds(args.SEED)

    # 路径配置
    paths = {
        "model_dir": f"model/{args.DATASET}",
        "result_dir": f"results/{args.DATASET}",
    }

    # 模型保存路径
    timestamp = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d_%H_%M_%S")
    # 每个检查点
    ckpt_dir = os.path.join(paths["model_dir"], "saved_checkpoints", timestamp)
    # 验证集上的最佳模型
    best_model_path = os.path.join(
        paths["model_dir"], "best_models", timestamp, "best-model.pth"
    )

    # Device
    device = get_device(args.device)

    # 1. 数据加载与预处理
    start_time = time.time()
    graph = build_nx_graph_from_config(get_config(args.DATASET))
    num_nodes = len(graph.nodes())
    data_loader = DataLoader(graph)

    # 2. 交互序列提取
    worker_num = min(args.THREAD_NUM, os.cpu_count() or 1)
    interactions, avg_seq_len = data_loader.extract_interaction_seqs(args.TASK_TYPE)
    if args.IS_LEN == -1:
        final_seq_len = avg_seq_len
        print(f"📏 模型采纳推荐的平均序列长度: {final_seq_len}")
    else:
        final_seq_len = args.IS_LEN
        print(f"📏 模型使用指定的交互序列长度: {final_seq_len}")

    # 3. 上下文窗口提取（基于随机游走）
    final_walk_num = args.WALK_NUM
    final_walk_len = args.WALK_LEN
    need_estimate = (args.WALK_NUM == -1) or (args.WALK_LEN == -1)
    if need_estimate:
        recommend_walk_num, recommend_walk_len = data_loader.estimate_walk_params(
            task_type=args.TASK_TYPE
        )
        param_updates = []
        if args.WALK_NUM == -1:
            final_walk_num = recommend_walk_num
            param_updates.append(f"游走次数: {final_walk_num}")
        if args.WALK_LEN == -1:
            # w2v版本的walk_len一般需要更长的游走
            final_walk_len = (
                recommend_walk_len * 2 if args.VERSION == "w2v" else recommend_walk_len
            )
            param_updates.append(f"游走长度: {final_walk_len}")
        if param_updates:
            print(f"📏 模型采纳推荐的游走参数：{', '.join(param_updates)}")
    else:
        print(
            f"📏 模型使用指定的游走参数：游走次数: {final_walk_num}, 游走长度: {final_walk_len}"
        )

    contexts = data_loader.extract_ctx_window(
        args.TASK_TYPE,
        walk_num=final_walk_num,
        walk_len=final_walk_len,
        workers=worker_num,
    )

    # 4. 节点特征随机初始化（正态分布），包含padding节点
    node_feature = np.random.randn(num_nodes + 1, args.FEAT_DIM).astype(np.float32)
    if args.VERSION == "w2v":
        ctx = []
        for windows in contexts.values():
            # 遍历每个节点的所有序列，提取每个元素的第0位（节点）
            ctx.extend([[w[0] for w in walk] for walk in windows])
        # print(ctx[:2])  # check前2个上下文窗口
        word2vec = Word2Vec(
            ctx,
            vector_size=args.FEAT_DIM,
            workers=worker_num,
            window=final_walk_len,
            min_count=final_seq_len,
            batch_words=args.BATCH_SIZE,
            negative=final_seq_len,
            seed=args.SEED,
        )
        print("\n✅ Word2Vec训练完成")
        print(
            f"   ├─ 节点数: {len(word2vec.wv.index_to_key)} | 特征维度: {word2vec.wv.vector_size}"
        )
        print(f"   └─ 游走次数: {final_walk_num} | 游走长度: {final_walk_len}")

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
        version=args.VERSION,
        rnn_type=args.RNN_TYPE,
        padding_node=args.PADDING_NODE,
        device=device,
    )
    model.to(device)

    # 6. 训练
    # 训练/验证/测试数据预处理
    train_data, val_data, test_data = data_loader.preprocess(
        args.TASK_TYPE, args.MASK_RATIO, args.SEED
    )
    print("\n🚀 Start training...")
    train(
        model=model,
        train_data=train_data,
        val_data=val_data,
        batch_size=args.BATCH_SIZE,
        epochs=args.EPOCH,
        lr=args.LR,
        seed=args.SEED,
        best_model_path=best_model_path,
    )

    # 7. 测试
    print("\n📝 Final testing...")
    test_neg = np.array(
        negative_sampling(
            test_data["nodes"],
            test_data["adj"],
            test_data["neg_num"],
            hard_ratio=0.0,
            seed=args.SEED,
        )
    )
    np.random.shuffle(test_neg)
    test_acc, test_ap, test_f1, test_auc = eval_one_epoch(
        model, test_data["edges"], test_neg, args.BATCH_SIZE
    )

    print("=== Test Results ===")
    print(f"Test Acc: {test_acc:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Test AP: {test_ap:.4f}")
    print(f"Test F1: {test_f1:.4f}")

    execution_time = time.time() - start_time
    task_desc = "Transductive" if args.TASK_TYPE == "T" else "Inductive"
    print(
        f"\n✅ Finish {task_desc} LP Task on {args.DATASET}! Cost time: {execution_time:.2f}s"
    )

    # 8. 结果保存
    core_param_desc_list = [
        f"SEED-{args.SEED}",
        f"TASK-{args.TASK_TYPE}",
        f"IL-{final_seq_len}",
        f"WN-{final_walk_num}",
        f"WL-{final_walk_len}",
    ]
    if args.TASK_TYPE == "I":
        mask_percent = (
            int(args.MASK_RATIO * 100)
            if args.MASK_RATIO * 100 == int(args.MASK_RATIO * 100)
            else args.MASK_RATIO * 100
        )
        mask_ratio_str = (
            f"{mask_percent:.0f}%"
            if args.MASK_RATIO * 100 % 1 == 0
            else f"{mask_percent:.1f}%"
        )
        core_param_desc_list.append(f"MR-{mask_ratio_str}")
    result_dir = os.path.join(paths["result_dir"], "_".join(core_param_desc_list))
    os.makedirs(result_dir, exist_ok=true)
    result_path = os.path.join(result_dir, f"IPNet-{args.VERSION}.csv")
    print(f"📁 结果文件将保存至：{result_path}")
    # 写入CSV（追加模式，自动加表头）
    need_header = not os.path.exists(result_path)
    with open(result_path, "a+", newline="") as f:
        writer = csv.writer(f)
        if need_header:
            writer.writerow(
                [
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
                args.TASK_TYPE,
                test_acc * 100,
                test_auc * 100,
                test_ap * 100,
                test_f1 * 100,
                execution_time,
                args.SEED,
                best_model_path,
            ]
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="IPNet for Dynamic Network Link Prediction (Apple Silicon Adapted)"
    )

    # 基础配置
    parser.add_argument(
        "--seed",
        dest="SEED",
        type=int,
        default=19,
        help="Random seed for reproducibility (default: 19)",
    )

    # 数据集与任务配置
    parser.add_argument(
        "--dataset", dest="DATASET", default="UCI", help="Dataset name (default: UCI)"
    )
    parser.add_argument(
        "--ty",
        dest="TASK_TYPE",
        default="T",
        help="Task type: T (Transductive)/I (Inductive)",
    )
    parser.add_argument(
        "--mask",
        dest="MASK_RATIO",
        type=float,
        default=0.1,
        help="Mask ratio for inductive task (0<mask<1, default: 0.1)",
    )

    # 模型参数
    parser.add_argument(
        "--v",
        dest="VERSION",
        default="mean",
        help="IPNet version: mean/att/w2v (default: w2v)",
    )
    parser.add_argument(
        "--fd",
        dest="FEAT_DIM",
        type=int,
        default=128,
        help="Feature dimension (default: 128)",
    )
    parser.add_argument(
        "--rnn",
        dest="RNN_TYPE",
        default="GRU",
        help="RNN type: LSTM/GRU (default: GRU)",
    )
    parser.add_argument(
        "--pd",
        dest="PADDING_NODE",
        type=int,
        default=0,
        help="Padding node ID (default: 0, must be non-negative and not recommended to modify)",
    )

    # 核心超参
    parser.add_argument(
        "--il",
        dest="IS_LEN",
        type=int,
        default=-1,
        help="Interaction sequence length (default: -1, use average sequence length automatically)",
    )
    parser.add_argument(
        "--wn",
        dest="WALK_NUM",
        type=int,
        default=-1,
        help="Random walks per node (default: -1=auto-estimate; ≥1 for manual setting)",
    )
    parser.add_argument(
        "--wl",
        dest="WALK_LEN",
        type=int,
        default=-1,
        help="Single walk length (default: -1=auto-estimate; ≥5 for manual setting)",
    )

    # 训练配置
    parser.add_argument(
        "--epoch",
        dest="EPOCH",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)",
    )
    parser.add_argument(
        "--bs", dest="BATCH_SIZE", type=int, default=64, help="Batch size (default: 64)"
    )
    parser.add_argument(
        "--lr",
        dest="LR",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--thread",
        dest="THREAD_NUM",
        type=int,
        default=5,
        help="Number of workers (default: 5)",
    )
    parser.add_argument(
        "--device",
        dest="device",
        type=int,
        default=-1,
        help="Device: -2=CPU, -1=MPS (default: -1)",
    )

    args = parser.parse_args()

    if args.PADDING_NODE < 0:
        raise ValueError(
            f"Padding node ID must be non-negative! Current: {args.PADDING_NODE}"
        )

    if args.TASK_TYPE == "I":
        if not (0 < args.MASK_RATIO < 1):
            raise ValueError(
                f"Inductive task requires mask_ratio in (0,1)! Current: {args.MASK_RATIO}"
            )

        mask_str = str(args.MASK_RATIO)
        if "." in mask_str:
            decimal_part = mask_str.split(".")[1]
            if len(decimal_part) > 2:
                raise ValueError(
                    f"mask_ratio must have at most 2 decimal places! "
                    f"Current: {args.MASK_RATIO} (decimal part: {decimal_part[:5]}...)"
                )

    main(args)
