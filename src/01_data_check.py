"""任务一：数据检查与基线

完成数据质量检查、描述性统计和热门推荐基线。
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np


def load_config(config_path="config.yaml"):
    """加载配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def check_data_files(config):
    """检查数据文件是否存在"""
    raw_dir = config["data"]["raw_dir"]
    required_files = [
        config["data"]["ratings_file"],
        config["data"]["movies_file"],
        config["data"]["tags_file"],
        config["data"]["links_file"],
    ]
    missing = []
    for f in required_files:
        path = os.path.join(raw_dir, f)
        if not os.path.exists(path):
            missing.append(f)
    if missing:
        print(f"[ERROR] 缺少数据文件: {missing}")
        print(f"请从 https://grouplens.org/datasets/movielens/32m/ 下载并放入 {raw_dir}/")
        return False
    print("[OK] 所有数据文件已就位")
    return True


def descriptive_statistics(ratings, movies, tags):
    """生成描述性统计报告"""
    print("\n=== 描述性统计 ===")
    print(f"评分记录数: {len(ratings):,}")
    print(f"用户数: {ratings['userId'].nunique():,}")
    print(f"电影数: {ratings['movieId'].nunique():,}")
    print(f"评分范围: {ratings['rating'].min()} - {ratings['rating'].max()}")
    print(f"平均评分: {ratings['rating'].mean():.2f}")
    print(f"标签记录数: {len(tags):,}")
    print(f"唯一标签数: {tags['tag'].nunique():,}")
    print(f"电影类型数: {movies['genres'].nunique():,}")


def popularity_baseline(ratings, config):
    """热门推荐基线"""
    threshold = config["split"]["pos_threshold"]
    pos_ratings = ratings[ratings["rating"] >= threshold]
    popular_movies = (
        pos_ratings.groupby("movieId")
        .size()
        .sort_values(ascending=False)
        .head(config["recommendation"]["top_k"])
    )
    print("\n=== 热门推荐基线 (Top 10) ===")
    print(popular_movies)
    return popular_movies


def main():
    config = load_config()
    if not check_data_files(config):
        sys.exit(1)

    raw_dir = config["data"]["raw_dir"]
    ratings = pd.read_csv(os.path.join(raw_dir, config["data"]["ratings_file"]))
    movies = pd.read_csv(os.path.join(raw_dir, config["data"]["movies_file"]))
    tags = pd.read_csv(os.path.join(raw_dir, config["data"]["tags_file"]))

    descriptive_statistics(ratings, movies, tags)
    popularity_baseline(ratings, config)

    print("\n[Done] 任务一完成")


if __name__ == "__main__":
    main()
