"""任务三：ALS 矩阵分解（使用 implicit 库，无需 Spark/Java）"""

import os
import sys
import json
import time
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from utils import load_config, ensure_dir


def train_als_implicit(train_data, config, test_users=None):
    """使用 implicit 库训练 ALS 模型"""
    try:
        from implicit.als import AlternatingLeastSquares
    except ImportError:
        print("[ERROR] implicit 库未安装，请运行: pip install implicit")
        sys.exit(1)

    als_config = config["als"]
    top_k = config["recommendation"]["top_k"]
    threshold = config["split"]["pos_threshold"]

    # 构建用户-物品矩阵（只保留正反馈）
    pos_data = train_data[train_data["rating"] >= threshold]
    # 包含所有用户，不只是有正反馈的用户
    users = train_data["userId"].unique()
    items = pos_data["movieId"].unique()
    user_map = {u: i for i, u in enumerate(users)}
    item_map = {m: i for i, m in enumerate(items)}

    row = pos_data["userId"].map(user_map).values
    col = pos_data["movieId"].map(item_map).values
    data = np.ones(len(pos_data), dtype=np.float32)

    user_item_matrix = csr_matrix((data, (row, col)), shape=(len(users), len(items)))
    print(f"用户-物品矩阵: {user_item_matrix.shape}, 非零元素: {user_item_matrix.nnz:,}")

    # 训练 ALS（implicit 需要 user-item 矩阵，形状为 用户×物品）
    model = AlternatingLeastSquares(
        factors=als_config["rank"],
        iterations=als_config["max_iter"],
        regularization=als_config["reg_param"],
        random_state=config["random_seed"],
    )

    start_time = time.time()
    model.fit(user_item_matrix)
    train_time = time.time() - start_time
    print(f"训练时间: {train_time:.2f}s")

    # 为测试集用户生成推荐（如果提供），否则随机采样
    if test_users is not None:
        target_users = [u for u in test_users if u in user_map]
        print(f"为 {len(target_users)}/{len(test_users)} 位测试用户生成推荐")
    else:
        sample_size = min(5000, len(users))
        np.random.seed(config["random_seed"])
        target_users = np.random.choice(users, size=sample_size, replace=False)

    recommendations = {}
    for user_id in target_users:
        u_idx = user_map.get(user_id)
        if u_idx is None:
            continue
        recs = model.recommend(
            userid=u_idx,
            user_items=user_item_matrix[u_idx],
            N=top_k,
            filter_already_liked_items=True,
        )
        # recs 返回 (item_indices, scores)，item_indices 是模型内部索引
        recommendations[str(user_id)] = [
            {"movieId": int(items[idx]), "score": float(score)}
            for idx, score in zip(recs[0], recs[1]) if idx < len(items)
        ]

    return recommendations, train_time


def main():
    config = load_config()
    splits_dir = config["data"]["splits_dir"]
    train_path = os.path.join(splits_dir, "train.csv")
    test_path = os.path.join(splits_dir, "test.csv")

    if not os.path.exists(train_path):
        print("[ERROR] 请先运行 02_data_split.py")
        sys.exit(1)

    train = pd.read_csv(train_path)
    print(f"训练集: {len(train):,} 条")

    als_config = config["als"]
    print(f"参数: rank={als_config['rank']}, iterations={als_config['max_iter']}, reg={als_config['reg_param']}")

    # 加载测试集用户
    test_users = None
    if os.path.exists(test_path):
        test_users = pd.read_csv(test_path)["userId"].unique().tolist()

    recommendations, train_time = train_als_implicit(train, config, test_users=test_users)

    # 保存结果
    ensure_dir("results")
    output = {
        "train_time": train_time,
        "num_users": len(recommendations),
        "recommendations": recommendations,
    }

    with open("results/als_recommendations.json", "w") as f:
        json.dump(output, f)

    print(f"已生成 {len(recommendations)} 位用户的推荐")
    print("结果已保存到 results/als_recommendations.json")
    print("[Done] ALS 训练完成")


if __name__ == "__main__":
    main()
