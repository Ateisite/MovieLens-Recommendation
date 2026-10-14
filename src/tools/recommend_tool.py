"""ALS 推荐工具"""

import os
import json
import numpy as np
import pandas as pd
from langchain_core.tools import tool

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# 全局变量存储 ALS 模型数据
_als_data = None


def _load_als_data():
    global _als_data
    if _als_data is not None:
        return _als_data

    # 加载推荐结果
    with open(os.path.join(PROJECT_ROOT, "results", "als_recommendations.json"), "r") as f:
        recs = json.load(f)

    # 加载电影数据
    movies_df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "raw", "movies.csv"))

    _als_data = {
        "recommendations": recs["recommendations"],
        "movies": movies_df.set_index("movieId")["title"].to_dict()
    }
    return _als_data


@tool
def recommend(user_id: int) -> str:
    """为指定用户生成 Top-10 个性化推荐。

    Args:
        user_id: 用户 ID

    Returns:
        推荐电影列表字符串
    """
    data = _load_als_data()
    recs = data["recommendations"].get(str(user_id))

    if not recs:
        return f"用户 {user_id} 没有推荐结果"

    output = [f"为用户 {user_id} 推荐：\n"]
    for i, rec in enumerate(recs[:10], 1):
        movie_id = rec["movieId"]
        title = data["movies"].get(movie_id, f"电影{movie_id}")
        score = rec.get("score", 0)
        output.append(f"{i}. {title} (ID: {movie_id}, 分数: {score:.4f})")

    return "\n".join(output)


@tool
def get_user_history(user_id: int) -> str:
    """获取用户用户历史评分记录。

    Args:
        user_id: 用户 ID

    Returns:
        用户历史评分字符串
    """
    try:
        train = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "splits", "train.csv"))
        user_ratings = train[train["userId"] == user_id]

        if user_ratings.empty:
            return f"用户 {user_id} 没有历史记录"

        movies = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "raw", "movies.csv"))
        movie_titles = movies.set_index("movieId")["title"].to_dict()

        output = [f"用户 {user_id} 的历史评分：\n"]
        for _, row in user_ratings.head(10).iterrows():
            title = movie_titles.get(row["movieId"], f"电影{row['movieId']}")
            output.append(f"- {title}: {row['rating']} 分")

        return "\n".join(output)
    except Exception as e:
        return f"获取历史失败: {str(e)}"


if __name__ == "__main__":
    # 测试
    print("=== recommend(1) ===")
    print(recommend.invoke({"user_id": 1}))
    print()
    print("=== get_user_history(1) ===")
    print(get_user_history.invoke({"user_id": 1}))



