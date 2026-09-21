"""评价指标计算

计算 NDCG@K, Recall@K 等推荐评价指标。
"""

import yaml
import numpy as np
import pandas as pd


def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dcg_at_k(relevances, k):
    """计算 DCG@K"""
    relevances = np.array(relevances)[:k]
    if len(relevances) == 0:
        return 0.0
    gains = 2 ** relevances - 1
    discounts = np.log2(np.arange(len(relevances)) + 2)
    return np.sum(gains / discounts)


def ndcg_at_k(relevances, k):
    """计算 NDCG@K"""
    dcg = dcg_at_k(relevances, k)
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = dcg_at_k(ideal_relevances, k)
    if idcg == 0:
        return 0.0
    return dcg / idcg


def recall_at_k(recommended, relevant, k):
    """计算 Recall@K"""
    recommended = set(recommended[:k])
    relevant = set(relevant)
    if len(relevant) == 0:
        return 0.0
    return len(recommended & relevant) / len(relevant)


def evaluate_model(recommendations, test_data, config):
    """评估模型整体表现"""
    k = config["recommendation"]["ndcg_k"]
    ndcg_scores = []
    recall_scores = []

    for user_id, rec_items in recommendations.items():
        user_test = test_data[test_data["userId"] == user_id]
        threshold = config["split"]["pos_threshold"]
        relevant = set(user_test[user_test["rating"] >= threshold]["movieId"])

        relevances = [1 if item in relevant else 0 for item in rec_items]
        ndcg_scores.append(ndcg_at_k(relevances, k))
        recall_scores.append(recall_at_k(rec_items, relevant, k))

    return {
        f"NDCG@{k}": np.mean(ndcg_scores),
        f"Recall@{k}": np.mean(recall_scores),
    }


def main():
    config = load_config()
    print("[TODO] 需要推荐结果文件进行评价")
    print("[Done] 评价模块就绪")


if __name__ == "__main__":
    main()
