"""消融实验

对比不同方法的推荐性能，说明每个改进的贡献。
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from utils import load_config, ensure_dir

import sys
import importlib.util

# 动态加载 evaluate 模块
spec = importlib.util.spec_from_file_location("evaluate", os.path.join(os.path.dirname(os.path.abspath(__file__)), "09_evaluate.py"))
evaluate_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluate_module)

evaluate_model = evaluate_module.evaluate_model
dcg_at_k = evaluate_module.dcg_at_k
ndcg_at_k = evaluate_module.ndcg_at_k
recall_at_k = evaluate_module.recall_at_k
precision_at_k = evaluate_module.precision_at_k


def compute_popularity_baseline(train_data, test_data, config):
    """计算热门推荐基线（与 data_check.py 一致，防止数据泄漏）"""
    threshold = config["split"]["pos_threshold"]
    top_k = config["recommendation"]["ndcg_k"]

    # 从训练集计算热门电影（防止数据泄漏）
    pos_train = train_data[train_data["rating"] >= threshold]
    popular_movies = (
        pos_train.groupby("movieId")
        .size()
        .sort_values(ascending=False)
        .head(top_k)
        .index.tolist()
    )

    # 构建测试集正样本
    user_pos_items = (
        test_data[test_data["rating"] >= threshold]
        .groupby("userId")["movieId"]
        .apply(set)
        .to_dict()
    )

    # 计算 NDCG
    ndcg_scores = []
    recall_scores = []
    precision_scores = []
    covered_users = 0

    for user_id, pos_items in user_pos_items.items():
        relevances = [1 if m in pos_items else 0 for m in popular_movies]
        ndcg_scores.append(ndcg_at_k(relevances, top_k))
        recall_scores.append(recall_at_k(popular_movies, pos_items, top_k))
        precision_scores.append(precision_at_k(popular_movies, pos_items, top_k))
        covered_users += 1

    return {
        "method": "Baseline (Popular)",
        "NDCG@10": round(np.mean(ndcg_scores), 4) if ndcg_scores else 0.0,
        "Recall@10": round(np.mean(recall_scores), 4) if recall_scores else 0.0,
        "Precision@10": round(np.mean(precision_scores), 4) if precision_scores else 0.0,
        "covered_users": covered_users,
    }


def load_and_evaluate(model_name, file_path, test_data, config):
    """加载推荐结果并评估"""
    if not os.path.exists(file_path):
        return None

    with open(file_path, "r") as f:
        data = json.load(f)

    # 处理不同格式
    if isinstance(data, dict) and "recommendations" in data:
        recs = data["recommendations"]
    else:
        recs = data

    # 转换为 evaluate_model 格式
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
    results["method"] = model_name
    return results


def main():
    config = load_config()
    splits_dir = config["data"]["splits_dir"]
    test_path = os.path.join(splits_dir, "test.csv")
    train_path = os.path.join(splits_dir, "train.csv")

    if not os.path.exists(test_path):
        print("[ERROR] 测试集不存在，请先运行 02_data_split.py")
        return

    test_data = pd.read_csv(test_path)
    train_data = pd.read_csv(train_path)

    print("=" * 60)
    print("消融实验 (Ablation Study)")
    print("=" * 60)

    # E1: 基线（热门推荐）
    print("\n[E1] 基线: 热门推荐")
    baseline_results = compute_popularity_baseline(train_data, test_data, config)
    print(f"  NDCG@10: {baseline_results['NDCG@10']}")
    print(f"  Recall@10: {baseline_results['Recall@10']}")

    # E2-E4: 加载各模型推荐结果
    model_files = {
        "E2: ALS (传统CF)": "results/als_recommendations.json",
        "E3: NCF (深度学习)": "results/ncf_recommendations.json",
    }

    all_results = [baseline_results]

    for model_name, file_path in model_files.items():
        print(f"\n[{model_name}]")
        results = load_and_evaluate(model_name, file_path, test_data, config)
        if results:
            print(f"  NDCG@10: {results['NDCG@10']}")
            print(f"  Recall@10: {results['Recall@10']}")
            all_results.append(results)
        else:
            print(f"  结果文件不存在: {file_path}")

    # 保存对比结果
    ensure_dir("results")
    results_df = pd.DataFrame(all_results)
    results_df = results_df.set_index("method")
    results_df.to_csv("results/ablation_study.csv")

    print("\n" + "=" * 60)
    print("消融实验结果汇总")
    print("=" * 60)
    print(results_df.to_string())
    print("\n结果已保存到 results/ablation_study.csv")

    # 计算提升
    if len(all_results) >= 3:
        baseline_ndcg = all_results[0]["NDCG@10"]
        als_ndcg = all_results[1]["NDCG@10"] if len(all_results) > 1 else 0
        ncf_ndcg = all_results[2]["NDCG@10"] if len(all_results) > 2 else 0

        print("\n提升分析:")
        print(f"  ALS vs 基线: {((als_ndcg - baseline_ndcg) / baseline_ndcg * 100):+.1f}%")
        print(f"  NCF vs 基线: {((ncf_ndcg - baseline_ndcg) / baseline_ndcg * 100):+.1f}%")
        print(f"  NCF vs ALS: {((ncf_ndcg - als_ndcg) / als_ndcg * 100):+.1f}%")


if __name__ == "__main__":
    main()
