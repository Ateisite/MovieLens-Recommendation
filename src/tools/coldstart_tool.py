"""冷启动推荐工具

为新用户提供推荐：输入喜欢的电影 → 基于 ALS 物品向量找相似电影
"""

import os
import json
import numpy as np
import pandas as pd
from langchain_core.tools import tool

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 全局缓存
_coldstart_data = None


def _load_coldstart_data():
    """加载 ALS 物品因子和映射"""
    global _coldstart_data
    if _coldstart_data is not None:
        return _coldstart_data

    factors_path = os.path.join(PROJECT_ROOT, "results", "als_item_factors.npy")
    map_path = os.path.join(PROJECT_ROOT, "results", "als_movie_map.json")

    if not os.path.exists(factors_path) or not os.path.exists(map_path):
        return None

    item_factors = np.load(factors_path)
    with open(map_path, "r") as f:
        movie_map = json.load(f)

    movies_df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "raw", "movies.csv"))
    movie_titles = movies_df.set_index("movieId")["title"].to_dict()
    movie_genres = movies_df.set_index("movieId")["genres"].to_dict()

    _coldstart_data = {
        "item_factors": item_factors,
        "movie_id_to_idx": movie_map["movie_id_to_idx"],
        "idx_to_movie_id": {int(k): v for k, v in movie_map["idx_to_movie_id"].items()},
        "movie_titles": movie_titles,
        "movie_genres": movie_genres,
    }
    return _coldstart_data


@tool
def coldstart_recommend(liked_movie_ids: list, top_k: int = 10) -> str:
    """为新用户提供推荐。新用户输入喜欢的电影 ID，返回相似电影推荐。

    Args:
        liked_movie_ids: 用户喜欢的电影 ID 列表
        top_k: 推荐数量（默认 10）

    Returns:
        推荐电影列表字符串
    """
    data = _load_coldstart_data()
    if data is None:
        return "冷启动模型未加载，请先运行 ALS 训练"

    item_factors = data["item_factors"]
    movie_id_to_idx = data["movie_id_to_idx"]
    idx_to_movie_id = data["idx_to_movie_id"]
    movie_titles = data["movie_titles"]
    movie_genres = data["movie_genres"]

    # 获取喜欢电影的向量
    vectors = []
    valid_ids = []
    for mid in liked_movie_ids:
        idx = movie_id_to_idx.get(str(mid))
        if idx is not None:
            vectors.append(item_factors[idx])
            valid_ids.append(mid)

    if not vectors:
        return "输入的电影 ID 都不在训练集中，请尝试其他电影"

    # 计算用户偏好向量（简单平均）
    user_vector = np.mean(vectors, axis=0)

    # 计算所有电影分数
    scores = item_factors @ user_vector

    # 排除已输入的电影
    exclude_idx = {movie_id_to_idx[str(mid)] for mid in valid_ids if str(mid) in movie_id_to_idx}

    # 取 Top-K
    sorted_indices = np.argsort(-scores)
    results = []
    for idx in sorted_indices:
        if idx in exclude_idx:
            continue
        movie_id = idx_to_movie_id[idx]
        title = movie_titles.get(movie_id, f"电影{movie_id}")
        genres = movie_genres.get(movie_id, "未知")
        score = scores[idx]
        results.append(f"{len(results)+1}. {title} | 类型: {genres} | 相似度: {score:.4f}")
        if len(results) >= top_k:
            break

    if not results:
        return "无法生成推荐"

    return f"基于你喜欢的 {len(valid_ids)} 部电影，为你推荐：\n\n" + "\n".join(results)


if __name__ == "__main__":
    # 测试
    print(coldstart_recommend.invoke({"liked_movie_ids": [1, 2, 3], "top_k": 5}))
