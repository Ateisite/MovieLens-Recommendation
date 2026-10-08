"""任务三（算法模块1）：User-based 协同过滤

基于用户的协同过滤推荐算法。
使用向量化操作避免 O(n^2) 循环，稀疏矩阵避免内存溢出。
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from utils import load_config, ensure_dir


def build_user_item_matrix(ratings, config):
    """构建用户-物品评分矩阵（稀疏）"""
    threshold = config["split"]["pos_threshold"]
    pos_ratings = ratings[ratings["rating"] >= threshold]

    users = pos_ratings["userId"].unique()
    items = pos_ratings["movieId"].unique()
    user_map = {u: i for i, u in enumerate(users)}
    item_map = {m: i for i, m in enumerate(items)}

    row = pos_ratings["userId"].map(user_map).values
    col = pos_ratings["movieId"].map(item_map).values
    data = np.ones(len(pos_ratings), dtype=np.float32)

    matrix = csr_matrix((data, (row, col)), shape=(len(users), len(items)))
    return matrix, user_map, item_map, users, items


def compute_user_similarity_vectorized(matrix, config):
    """向量化计算用户间余弦相似度，只保留 Top-K 最近邻（全程稀疏，分批计算）"""
    max_neighbors = config["collaborative"]["max_neighbors"]

    # 归一化后矩阵乘积 = 余弦相似度
    norms = np.sqrt(matrix.multiply(matrix).sum(axis=1)).A1
    norms[norms == 0] = 1.0  # 避免除零
    normalized = matrix.multiply(1.0 / norms[:, None]).tocsr()  # 转为 csr 支持切片

    # 分批计算避免内存爆炸（每次只算 5000 用户的相似度）
    batch_size = 5000
    n_users = matrix.shape[0]
    rows, cols, data = [], [], []

    for start in range(0, n_users, batch_size):
        end = min(start + batch_size, n_users)
        batch_sim = normalized[start:end] @ normalized.T  # (batch, n_users) 稀疏

        for i_local in range(batch_sim.shape[0]):
            i_global = start + i_local
            row = batch_sim[i_local].tocsr()
            row_data = row.data
            row_indices = row.indices

            # 排除自身
            mask = row_indices != i_global
            row_data = row_data[mask]
            row_indices = row_indices[mask]

            # Top-K 剪枝：只保留最相似的 max_neighbors 个
            if len(row_data) > max_neighbors:
                top_idx = np.argpartition(row_data, -max_neighbors)[-max_neighbors:]
                row_data = row_data[top_idx]
                row_indices = row_indices[top_idx]

            rows.extend([i_global] * len(row_data))
            cols.extend(row_indices)
            data.extend(row_data)

    sim = csr_matrix((data, (rows, cols)), shape=(n_users, n_users))
    return sim


def recommend(user_id, sim_matrix, user_item_matrix, user_map, item_map, items, top_k=10):
    """为指定用户生成推荐（全稀疏操作，无 toarray）"""
    if user_id not in user_map:
        return []
    u_idx = user_map[user_id]

    # 稀疏矩阵乘法计算推荐分数
    scores = sim_matrix[u_idx] @ user_item_matrix

    # 获取已交互物品
    seen = set(user_item_matrix[u_idx].nonzero()[1])

    # 提取分数并排序
    scores_dense = scores.toarray().flatten()
    item_scores = [
        (items[i], float(scores_dense[i]))
        for i in range(len(scores_dense))
        if i not in seen and scores_dense[i] > 0
    ]
    item_scores.sort(key=lambda x: x[1], reverse=True)
    return item_scores[:top_k]


def recommend_batch(sim_matrix, user_item_matrix, user_map, item_map, users, config, sample_size=1000):
    """批量生成推荐（向量化，高效）"""
    top_k = config["recommendation"]["top_k"]
    np.random.seed(config["random_seed"])

    # 采样用户进行推荐
    sample_users = np.random.choice(users, size=min(sample_size, len(users)), replace=False)

    all_recommendations = {}
    item_list = item_map  # item_map: movieId -> index
    idx_to_item = {v: k for k, v in item_map.items()}

    for user_id in sample_users:
        u_idx = user_map[user_id]
        scores = (sim_matrix[u_idx] @ user_item_matrix).toarray().flatten()
        seen = set(user_item_matrix[u_idx].nonzero()[1])

        # 排除已交互，取 Top-K
        scores[list(seen)] = -1
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(-scores[top_indices])]

        all_recommendations[user_id] = [
            (idx_to_item[idx], float(scores[idx])) for idx in top_indices if scores[idx] > 0
        ]

    return all_recommendations


def main():
    config = load_config()
    splits_dir = config["data"]["splits_dir"]
    train_path = os.path.join(splits_dir, "train.csv")

    if not os.path.exists(train_path):
        print("[ERROR] 请先运行 02_data_split.py")
        sys.exit(1)

    train = pd.read_csv(train_path)
    matrix, user_map, item_map, users, items = build_user_item_matrix(train, config)
    print(f"用户-物品矩阵: {matrix.shape}, 非零元素: {matrix.nnz:,}")

    sim = compute_user_similarity_vectorized(matrix, config)
    print(f"用户相似度矩阵: {sim.shape}, 非零元素: {sim.nnz:,}")

    # 批量生成推荐
    recommendations = recommend_batch(sim, matrix, user_map, item_map, users, config)

    # 保存结果
    ensure_dir("results")
    import json
    # 转换 numpy 类型为 Python 原生类型以便 JSON 序列化
    rec_for_save = {
        str(k): [(int(movie_id), float(score)) for movie_id, score in v]
        for k, v in recommendations.items()
    }
    with open("results/usercf_recommendations.json", "w") as f:
        json.dump(rec_for_save, f)

    print(f"已生成 {len(recommendations)} 位用户的推荐")
    print("结果已保存到 results/usercf_recommendations.json")
    print("[Done] UserCF 训练完成")


if __name__ == "__main__":
    main()
