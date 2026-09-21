"""任务一：数据检查与基线

完成数据质量检查、描述性统计和热门推荐基线。
输出基线 NDCG@10 到 results/baseline_ndcg.txt
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from utils import load_config, check_data_files, setup_logger, ensure_dir

logger = setup_logger("data_check")


def data_quality_report(ratings, movies, tags):
    """数据质量检查：缺失值、重复值、类型校验"""
    print("\n=== 数据质量报告 ===")

    # 缺失值
    for name, df in [("ratings", ratings), ("movies", movies), ("tags", tags)]:
        missing = df.isnull().sum()
        if missing.any():
            print(f"  [{name}] 缺失值:\n{missing[missing > 0]}")
        else:
            print(f"  [{name}] 无缺失值 ✓")

    # 重复值
    dup_ratings = ratings.duplicated(subset=["userId", "movieId"]).sum()
    print(f"  [ratings] 重复评分记录: {dup_ratings}")

    # 类型校验
    assert ratings["rating"].between(0.5, 5.0).all(), "评分超出范围"
    assert (ratings["timestamp"] > 0).all(), "时间戳异常"
    print("  数据类型校验通过 ✓")


def descriptive_statistics(ratings, movies, tags):
    """生成描述性统计报告"""
    print("\n=== 描述性统计 ===")
    stats = {
        "评分记录数": f"{len(ratings):,}",
        "用户数": f"{ratings['userId'].nunique():,}",
        "电影数": f"{ratings['movieId'].nunique():,}",
        "评分范围": f"{ratings['rating'].min()} - {ratings['rating'].max()}",
        "平均评分": f"{ratings['rating'].mean():.2f}",
        "评分中位数": f"{ratings['rating'].median():.1f}",
        "标签记录数": f"{len(tags):,}",
        "唯一标签数": f"{tags['tag'].nunique():,}",
        "电影类型数": f"{movies['genres'].nunique():,}",
        "评分稀疏度": f"{1 - len(ratings) / (ratings['userId'].nunique() * ratings['movieId'].nunique()):.4%}",
    }
    for k, v in stats.items():
        print(f"  {k}: {v}")
    return stats


def compute_baseline_ndcg(ratings, config):
    """计算热门推荐基线的 NDCG@10"""
    threshold = config["split"]["pos_threshold"]
    top_k = config["recommendation"]["top_k"]

    # 热门电影（按正反馈次数排序）
    pos_ratings = ratings[ratings["rating"] >= threshold]
    popular_movies = (
        pos_ratings.groupby("movieId")
        .size()
        .sort_values(ascending=False)
        .head(top_k)
        .index.tolist()
    )

    # 按用户计算 NDCG@K
    user_pos_items = (
        ratings[ratings["rating"] >= threshold]
        .groupby("userId")["movieId"]
        .apply(set)
        .to_dict()
    )

    ndcg_scores = []
    for user_id, pos_items in user_pos_items.items():
        relevances = [1 if m in pos_items else 0 for m in popular_movies]
        dcg = sum((2**r - 1) / np.log2(i + 2) for i, r in enumerate(relevances[:top_k]))
        ideal = sorted(relevances, reverse=True)[:top_k]
        idcg = sum((2**r - 1) / np.log2(i + 2) for i, r in enumerate(ideal))
        if idcg > 0:
            ndcg_scores.append(dcg / idcg)

    mean_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0.0
    print(f"\n=== 热门推荐基线 ===")
    print(f"  NDCG@{top_k}: {mean_ndcg:.4f}")
    print(f"  目标: 提升 ≥ 10% (即 ≥ {mean_ndcg * 1.1:.4f})")

    # 保存结果
    ensure_dir("results")
    with open("results/baseline_ndcg.txt", "w", encoding="utf-8") as f:
        f.write(f"Baseline NDCG@{top_k}: {mean_ndcg:.4f}\n")
        f.write(f"Target NDCG@{top_k} (+10%): {mean_ndcg * 1.1:.4f}\n")
        f.write(f"Top {top_k} popular movies: {popular_movies}\n")
    print(f"  结果已保存到 results/baseline_ndcg.txt")

    return mean_ndcg, popular_movies


def main():
    config = load_config()
    if not check_data_files(config):
        sys.exit(1)

    raw_dir = config["data"]["raw_dir"]
    ratings = pd.read_csv(os.path.join(raw_dir, config["data"]["ratings_file"]))
    movies = pd.read_csv(os.path.join(raw_dir, config["data"]["movies_file"]))
    tags = pd.read_csv(os.path.join(raw_dir, config["data"]["tags_file"]))

    data_quality_report(ratings, movies, tags)
    descriptive_statistics(ratings, movies, tags)
    compute_baseline_ndcg(ratings, config)

    print("\n[Done] 任务一完成")


if __name__ == "__main__":
    main()
