"""任务三（算法模块1）：User-based 协同过滤

基于用户的协同过滤推荐算法。
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_user_item_matrix(ratings, config):
    """构建用户-物品评分矩阵"""
    threshold = config["split"]["pos_threshold"]
    pos_ratings = ratings[ratings["rating"] >= threshold]

    users = pos_ratings["userId"].unique()
    items = pos_ratings["movieId"].unique()
    user_map = {u: i for i, u in enumerate(users)}
    item_map = {m: i for i, m in enumerate(items)}

    row = pos_ratings["userId"].map(user_map).values
    col = pos_ratings["movieId"].map(item_map).values
    data = np.ones(len(pos_ratings))

    matrix = csr_matrix((data, (row, col)), shape=(len(users), len(items)))
    return matrix, user_map, item_map


def compute_user_similarity(matrix, config):
    """计算用户间余弦相似度"""
    max_neighbors = config["collaborative"]["max_neighbors"]
    sim = cosine_similarity(matrix)
    # 只保留 Top-K 最近邻
    for i in range(sim.shape[0]):
        top_k_idx = np.argsort(sim[i])[-max_neighbors - 1:-1]
        mask = np.zeros(sim.shape[1], dtype=bool)
        mask[top_k_idx] = True
        sim[i][~mask] = 0
    return sim


def recommend(user_id, sim_matrix, user_item_matrix, user_map, item_map, top_k=10):
    """为指定用户生成推荐"""
    if user_id not in user_map:
        return []
    u_idx = user_map[user_id]
    scores = sim_matrix[u_idx] @ user_item_matrix.toarray()
    seen = set(user_item_matrix[u_idx].nonzero()[1])
    item_scores = [(list(item_map.keys())[i], scores[i]) for i in range(len(scores)) if i not in seen]
    item_scores.sort(key=lambda x: x[1], reverse=True)
    return item_scores[:top_k]


def main():
    config = load_config()
    splits_dir = config["data"]["splits_dir"]
    train_path = os.path.join(splits_dir, "train.csv")

    if not os.path.exists(train_path):
        print("[ERROR] 请先运行 02_data_split.py")
        sys.exit(1)

    train = pd.read_csv(train_path)
    matrix, user_map, item_map = build_user_item_matrix(train, config)
    sim = compute_user_similarity(matrix, config)

    print(f"用户相似度矩阵: {sim.shape}")
    print("[Done] UserCF 训练完成")


if __name__ == "__main__":
    main()
