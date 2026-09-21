"""评价指标计算

计算 NDCG@K, Recall@K 等推荐评价指标。
支持从各模块的推荐结果文件加载并评价。
"""

import os
import yaml
import json
import numpy as np
import pandas as pd
from utils import load_config, ensure_dir


def dcg_at_k(relevances, k):
    """计算 DCG@K"""
    relevances = np.array(relevances, dtype=float)[:k]
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


def precision_at_k(recommended, relevant, k):
    """计算 Precision@K"""
    recommended = set(recommended[:k])
    relevant = set(relevant)
    if len(recommended) == 0:
        return 0.0
    return len(recommended & relevant) / len(recommended)


def evaluate_model(recommendations, test_data, config):
    """评估模型整体表现
    
    recommendations: dict {user_id: [(item_id, score), ...] 或 [item_id, ...]}
    test_data: DataFrame with columns [userId, movieId, rating]
    """
    k = config["recommendation"]["ndcg_k"]
    threshold = config["split"]["pos_threshold"]

    # 构建测试集正样本
    user_pos_items = (
        test_data[test_data["rating"] >= threshold]
        .groupby("userId")["movieId"]
        .apply(set)
        .to_dict()
    )

    ndcg_scores = []
    recall_scores = []
    precision_scores = []
    covered_users = 0

    for user_id, rec_items in recommendations.items():
        # 处理不同格式的推荐结果
        if isinstance(rec_items, list) and len(rec_items) > 0:
            if isinstance(rec_items[0], tuple):
                rec_item_ids = [item for item, score in rec_items]
            elif isinstance(rec_items[0], dict):
                rec_item_ids = [item["movieId"] for item in rec_items]
            else:
                rec_item_ids = rec_items
        else:
            continue

        relevant = user_pos_items.get(user_id, set())
        if not relevant:
            continue

        relevances = [1 if item in relevant else 0 for item in rec_item_ids]
        ndcg_scores.append(ndcg_at_k(relevances, k))
        recall_scores.append(recall_at_k(rec_item_ids, relevant, k))
        precision_scores.append(precision_at_k(rec_item_ids, relevant, k))
        covered_users += 1

    results = {
        f"NDCG@{k}": round(np.mean(ndcg_scores), 4) if ndcg_scores else 0.0,
        f"Recall@{k}": round(np.mean(recall_scores), 4) if recall_scores else 0.0,
        f"Precision@{k}": round(np.mean(precision_scores), 4) if precision_scores else 0.0,
        "covered_users": covered_users,
        "total_test_users": len(user_pos_items),
    }
    return results


def load_recommendations(file_path):
    """从 JSON 文件加载推荐结果"""
    if not os.path.exists(file_path):
        return None
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def evaluate_all_models(config):
    """评价所有模型的推荐结果"""
    splits_dir = config["data"]["splits_dir"]
    test_path = os.path.join(splits_dir, "test.csv")

    if not os.path.exists(test_path):
        print("[ERROR] 测试集不存在，请先运行 02_data_split.py")
        return

    test_data = pd.read_csv(test_path)

    model_files = {
        "UserCF": "results/usercf_recommendations.json",
        "ItemCF": "results/itemcf_recommendations.json",
        "ALS": "results/als_recommendations.json",
        "Graph": "results/graph_recommendations.json",
    }

    all_results = {}
    for model_name, file_path in model_files.items():
        recs = load_recommendations(file_path)
        if recs is None:
            continue

        # 处理不同格式的推荐结果
        if isinstance(recs, dict) and "recommendations" in recs:
            recs = recs["recommendations"]

        # 转换格式
        formatted_recs = {}
        for user_id, items in recs.items():
            if isinstance(items, list) and len(items) > 0:
                if isinstance(items[0], dict):
                    formatted_recs[int(user_id)] = [(item["movieId"], item.get("score", 1.0)) for item in items]
                elif isinstance(items[0], (list, tuple)):
                    formatted_recs[int(user_id)] = items
                else:
                    formatted_recs[int(user_id)] = [(item, 1.0) for item in items]

        results = evaluate_model(formatted_recs, test_data, config)
        all_results[model_name] = results
        print(f"\n{model_name}:")
        for metric, value in results.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")

    # 保存对比结果
    if all_results:
        ensure_dir("results")
        results_df = pd.DataFrame(all_results).T
        results_df.to_csv("results/model_comparison.csv")
        print("\n对比结果已保存到 results/model_comparison.csv")

    return all_results


def main():
    config = load_config()
    evaluate_all_models(config)
    print("\n[Done] 评价完成")


if __name__ == "__main__":
    main()
