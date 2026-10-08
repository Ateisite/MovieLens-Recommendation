"""任务二：LSH 近似候选召回

使用局部敏感哈希（Locality-Sensitive Hashing）加速相似度计算。
对比精确相似度与 LSH 近似的效果与效率。
"""

import os
import sys
import time
import yaml
import json
import numpy as np
import pandas as pd
from collections import defaultdict
from datasketch import MinHash, MinHashLSH
from utils import load_config, ensure_dir


def build_user_item_sets(ratings, config):
    """构建用户-物品集合"""
    threshold = config["split"]["pos_threshold"]
    pos_ratings = ratings[ratings["rating"] >= threshold]

    user_item_sets = defaultdict(set)
    for _, row in pos_ratings.iterrows():
        user_item_sets[row["userId"]].add(row["movieId"])

    return user_item_sets


def build_lsh_index(user_item_sets, config):
    """构建 LSH 索引"""
    lsh_config = config["lsh"]
    lsh = MinHashLSH(
        threshold=lsh_config["threshold"],
        num_perm=lsh_config["num_perm"]
    )

    minhashes = {}
    for user_id, items in user_item_sets.items():
        m = MinHash(num_perm=lsh_config["num_perm"])
        for item in items:
            m.update(str(item).encode("utf8"))
        lsh.insert(str(user_id), m)
        minhashes[str(user_id)] = m

    return lsh, minhashes


def query_lsh_candidates(lsh, minhashes, user_item_sets, sample_size=100):
    """为采样用户查询 LSH 近似候选"""
    sample_users = list(minhashes.keys())[:sample_size]
    results = {}

    for user_id in sample_users:
        candidates = lsh.query(minhashes[user_id])
        # 排除自身
        candidates = [c for c in candidates if c != user_id]
        results[user_id] = candidates

    return results


def compute_exact_jaccard(user_item_sets, user_id, candidates, top_k=20):
    """计算精确的 Jaccard 相似度"""
    if user_id not in user_item_sets:
        return []

    target_set = user_item_sets[user_id]
    similarities = []

    for other_id in candidates:
        if other_id not in user_item_sets:
            continue
        other_set = user_item_sets[other_id]
        intersection = len(target_set & other_set)
        union = len(target_set | other_set)
        if union > 0:
            similarities.append((other_id, intersection / union))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]


def evaluate_lsh(user_item_sets, lsh_results, config):
    """评估 LSH 的准确率与召回率"""
    precisions = []
    recalls = []

    for user_id, lsh_candidates in lsh_results.items():
        if user_id not in user_item_sets:
            continue

        # LSH 候选中的真正相似用户 (Jaccard > 0.1)
        exact_sims = compute_exact_jaccard(user_item_sets, user_id, lsh_candidates)
        true_similar = {uid for uid, sim in exact_sims if sim > 0.1}

        if not true_similar:
            continue

        # LSH 返回的候选中有多少是真正相似的
        lsh_set = set(lsh_candidates)
        true_positives = len(lsh_set & true_similar)

        if len(lsh_set) > 0:
            precisions.append(true_positives / len(lsh_set))
        recalls.append(true_positives / len(true_similar) if true_similar else 0)

    avg_precision = np.mean(precisions) if precisions else 0
    avg_recall = np.mean(recalls) if recalls else 0

    print(f"\n=== LSH 评估 ===")
    print(f"  精确率 (Precision): {avg_precision:.4f}")
    print(f"  召回率 (Recall): {avg_recall:.4f}")
    print(f"  测试用户数: {len(lsh_results)}")

    return {"precision": avg_precision, "recall": avg_recall}


def main():
    config = load_config()
    splits_dir = config["data"]["splits_dir"]
    train_path = os.path.join(splits_dir, "train.csv")

    if not os.path.exists(train_path):
        print("[ERROR] 请先运行 02_data_split.py")
        sys.exit(1)

    train = pd.read_csv(train_path)
    user_item_sets = build_user_item_sets(train, config)
    print(f"用户数: {len(user_item_sets):,}")

    # 构建 LSH 索引
    start_time = time.time()
    lsh, minhashes = build_lsh_index(user_item_sets, config)
    build_time = time.time() - start_time
    print(f"LSH 索引构建时间: {build_time:.2f}s")

    # 查询候选
    start_time = time.time()
    lsh_results = query_lsh_candidates(lsh, minhashes, user_item_sets)
    query_time = time.time() - start_time
    print(f"LSH 查询时间: {query_time:.4f}s")

    # 评估
    eval_results = evaluate_lsh(user_item_sets, lsh_results, config)

    # 保存结果
    ensure_dir("results")
    output = {
        "build_time": build_time,
        "query_time": query_time,
        "evaluation": eval_results,
        "sample_results": {k: v[:10] for k, v in list(lsh_results.items())[:20]},
    }
    with open("results/lsh_evaluation.json", "w") as f:
        json.dump(output, f, indent=2)

    print("\n结果已保存到 results/lsh_evaluation.json")
    print("[Done] LSH 召回完成")


if __name__ == "__main__":
    main()
