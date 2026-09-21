"""任务三（算法模块2）：Item-based 协同过滤

基于物品的协同过滤推荐算法。
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from utils import load_config, ensure_dir


def build_item_similarity(ratings, config):
    """构建物品相似度矩阵（稀疏操作）"""
    threshold = config["split"]["pos_threshold"]
    pos_ratings = ratings[ratings["rating"] >= threshold]

    users = pos_ratings["userId"].unique()
    items = pos_ratings["movieId"].unique()
    user_map = {u: i for i, u in enumerate(users)}
    item_map = {m: i for i, m in enumerate(items)}

    row = pos_ratings["userId"].map(user_map).values
    col = pos_ratings["movieId"].map(item_map).values
    data = np.ones(len(pos_ratings), dtype=np.float32)

    user_item = csr_matrix((data, (row, col)), shape=(len(users), len(items)))

    # 归一化后计算余弦相似度
    norms = np.sqrt(user_item.multiply(user_item).sum(axis=0)).A1
    norms[norms == 0] = 1.0
    normalized = user_item.multiply(1.0 / norms)

    item_sim = normalized.T @ normalized
    return item_sim, user_map, item_map, users, items


def recommend_for_user(user_id, user_item_matrix, item_sim, user_map, item_map, items, top_k=10):
    """为单个用户生成推荐"""
    if user_id not in user_map:
        return []
    u_idx = user_map[user_id]

    # 用户历史交互
    interacted = user_item_matrix[u_idx].nonzero()[1]

    # 分数 = 用户历史物品的相似度加权和
    scores = np.zeros(item_sim.shape[0])
    for item_idx in interacted:
        scores += item_sim[item_idx].toarray().flatten()

    # 排除已交互
    scores[list(interacted)] = -1

    # Top-K
    top_indices = np.argpartition(scores, -top_k)[-top_k:]
    top_indices = top_indices[np.argsort(-scores[top_indices])]

    idx_to_item = {v: k for k, v in item_map.items()}
    return [(idx_to_item[idx], float(scores[idx])) for idx in top_indices if scores[idx] > 0]


def recommend_batch(user_item_matrix, item_sim, user_map, item_map, users, config, sample_size=1000):
    """批量生成推荐"""
    top_k = config["recommendation"]["top_k"]
    np.random.seed(config["random_seed"])

    sample_users = np.random.choice(users, size=min(sample_size, len(users)), replace=False)
    idx_to_item = {v: k for k, v in item_map.items()}

    all_recommendations = {}
    for user_id in sample_users:
        recs = recommend_for_user(user_id, user_item_matrix, item_sim, user_map, item_map, items, top_k)
        if recs:
            all_recommendations[user_id] = recs

    return all_recommendations


def main():
    config = load_config()
    splits_dir = config["data"]["splits_dir"]
    train_path = os.path.join(splits_dir, "train.csv")

    if not os.path.exists(train_path):
        print("[ERROR] 请先运行 02_data_split.py")
        sys.exit(1)

    train = pd.read_csv(train_path)
    item_sim, user_map, item_map, users, items = build_item_similarity(train, config)
    print(f"物品相似度矩阵: {item_sim.shape}, 非零元素: {item_sim.nnz:,}")

    # 构建用户-物品矩阵用于推荐
    threshold = config["split"]["pos_threshold"]
    pos_ratings = train[train["rating"] >= threshold]
    row = pos_ratings["userId"].map(user_map).values
    col = pos_ratings["movieId"].map(item_map).values
    data = np.ones(len(pos_ratings), dtype=np.float32)
    user_item_matrix = csr_matrix((data, (row, col)), shape=(len(users), len(items)))

    recommendations = recommend_batch(user_item_matrix, item_sim, user_map, item_map, users, config)

    ensure_dir("results")
    import json
    rec_for_save = {str(k): v for k, v in recommendations.items()}
    with open("results/itemcf_recommendations.json", "w") as f:
        json.dump(rec_for_save, f)

    print(f"已生成 {len(recommendations)} 位用户的推荐")
    print("结果已保存到 results/itemcf_recommendations.json")
    print("[Done] ItemCF 训练完成")


if __name__ == "__main__":
    main()
