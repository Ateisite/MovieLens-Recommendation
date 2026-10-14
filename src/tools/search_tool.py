"""电影搜索工具"""

import os
import pandas as pd
from langchain_core.tools import tool

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# 加载电影数据
_movies_df = None


def _get_movies_df():
    global _movies_df
    if _movies_df is None:
        _movies_df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "raw", "movies.csv"))
    return _movies_df


@tool
def search_movie(title: str) -> str:
    """按标题搜索电影，返回电影信息和 ID。

    Args:
        title: 电影标题（支持模糊匹配）

    Returns:
        电影信息字符串，包含 movieId、标题、类型
    """
    movies_df = _get_movies_df()

    # 模糊匹配
    mask = movies_df["title"].str.lower().str.contains(title.lower())
    results = movies_df[mask]

    if results.empty:
        return f"未找到包含 '{title}' 的电影"

    # 返回前 5 个结果
    output = []
    for _, row in results.head(5).iterrows():
        output.append(f"ID: {row['movieId']}, 标题: {row['title']}, 类型: {row['genres']}")

    return "\n".join(output)


@tool
def get_movie_details(movie_id: int) -> str:
    """获取电影详细信息。

    Args:
        movie_id: 电影 ID

    Returns:
        电影详细信息字符串
    """
    movies_df = _get_movies_df()

    movie = movies_df[movies_df["movieId"] == movie_id]
    if movie.empty:
        return f"未找到 ID 为 {movie_id} 的电影"

    row = movie.iloc[0]
    return f"标题: {row['title']}\n类型: {row['genres']}\nID: {row['movieId']}"


if __name__ == "__main__":
    # 测试（使用英文标题）
    print("=== search_movie('Interstellar') ===")
    print(search_movie.invoke({"title": "Interstellar"}))
    print()
    print("=== get_movie_details(109487) ===")
    print(get_movie_details.invoke({"movie_id": 109487}))
