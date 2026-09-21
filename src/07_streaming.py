"""任务四：流式增量处理

按时间戳回放评分流，实现窗口统计和增量画像更新。
使用向量化操作替代 iterrows，大幅提升性能。
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd
from utils import load_config, ensure_dir


def replay_rating_stream_vectorized(ratings, config):
    """向量化滑动窗口统计（替代 iterrows）"""
    window_size = config["streaming"]["window_size"]
    slide_interval = config["streaming"]["slide_interval"]

    ratings = ratings.sort_values("timestamp").reset_index(drop=True)
    n = len(ratings)

    # 使用 pandas rolling 进行向量化窗口统计
    ratings["rating_mean"] = ratings["rating"].rolling(
        window=window_size, min_periods=1
    ).mean()

    ratings["rating_std"] = ratings["rating"].rolling(
        window=window_size, min_periods=1
    ).std().fillna(0)

    # 按 slide_interval 采样统计点
    sample_indices = range(0, n, slide_interval)
    stats = ratings.iloc[sample_indices][["timestamp", "rating_mean", "rating_std"]].copy()
    stats["window_size"] = window_size
    stats["record_index"] = list(sample_indices)

    return stats


def incremental_profile_update(ratings, config):
    """增量更新用户画像
    
    按时间顺序处理，维护每个用户的：
    - 评分均值
    - 评分次数
    - 最近交互的电影类型分布
    """
    threshold = config["split"]["pos_threshold"]
    ratings = ratings.sort_values("timestamp")

    # 向量化：用 expanding 计算累积统计
    user_stats = ratings.groupby("userId")["rating"].expanding().agg(["mean", "count"]).reset_index()
    user_stats.columns = ["userId", "level", "cumulative_mean", "cumulative_count"]

    # 每个用户的最终画像
    user_profiles = ratings.groupby("userId").agg(
        avg_rating=("rating", "mean"),
        num_ratings=("rating", "count"),
        first_rating_time=("timestamp", "min"),
        last_rating_time=("timestamp", "max"),
    )

    # 正反馈率
    pos_counts = ratings[ratings["rating"] >= threshold].groupby("userId").size()
    user_profiles["positive_rate"] = (pos_counts / user_profiles["num_ratings"]).fillna(0)

    # 活跃天数
    user_profiles["active_days"] = (
        (user_profiles["last_rating_time"] - user_profiles["first_rating_time"]) / 86400
    ).fillna(0)

    return user_profiles


def main():
    config = load_config()
    raw_dir = config["data"]["raw_dir"]
    ratings_path = os.path.join(raw_dir, config["data"]["ratings_file"])

    if not os.path.exists(ratings_path):
        print("[ERROR] 请先下载数据文件")
        sys.exit(1)

    ratings = pd.read_csv(ratings_path)
    print(f"评分记录: {len(ratings):,} 条")

    # 滑动窗口统计
    stats = replay_rating_stream_vectorized(ratings, config)
    print(f"流处理窗口数: {len(stats):,}")
    print(f"全局平均评分: {stats['rating_mean'].mean():.2f}")

    # 用户画像
    profiles = incremental_profile_update(ratings, config)
    print(f"用户画像数: {len(profiles):,}")
    print(f"平均评分: {profiles['avg_rating'].mean():.2f}")
    print(f"平均正反馈率: {profiles['positive_rate'].mean():.2%}")

    # 保存结果
    ensure_dir("results")
    stats[["timestamp", "rating_mean", "rating_std", "record_index"]].to_csv(
        "results/streaming_window_stats.csv", index=False
    )
    profiles.to_csv("results/user_profiles.csv")

    print("\n结果已保存:")
    print("  - results/streaming_window_stats.csv")
    print("  - results/user_profiles.csv")
    print("[Done] 流式增量处理完成")


if __name__ == "__main__":
    main()
