"""任务四：流式增量处理

按时间戳回放评分流，实现窗口统计和增量画像更新。
"""

import os
import sys
import yaml
import pandas as pd
from collections import deque


def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def replay_rating_stream(ratings, config):
    """按时间戳回放评分流"""
    window_size = config["streaming"]["window_size"]
    slide_interval = config["streaming"]["slide_interval"]

    ratings = ratings.sort_values("timestamp")
    window = deque(maxlen=window_size)
    stats_log = []

    for i, row in ratings.iterrows():
        window.append(row)
        if len(window) % slide_interval == 0:
            # 窗口统计
            window_df = pd.DataFrame(list(window))
            avg_rating = window_df["rating"].mean()
            stats_log.append({
                "timestamp": row["timestamp"],
                "window_avg_rating": avg_rating,
                "window_size": len(window)
            })

    return pd.DataFrame(stats_log)


def incremental_profile_update(user_profiles, new_ratings, config):
    """增量更新用户画像"""
    # TODO: 实现增量更新逻辑
    pass


def main():
    config = load_config()
    raw_dir = config["data"]["raw_dir"]
    ratings_path = os.path.join(raw_dir, config["data"]["ratings_file"])

    if not os.path.exists(ratings_path):
        print("[ERROR] 请先下载数据文件")
        sys.exit(1)

    ratings = pd.read_csv(ratings_path)
    stats = replay_rating_stream(ratings, config)

    print(f"流处理窗口数: {len(stats)}")
    print(f"平均窗口评分: {stats['window_avg_rating'].mean():.2f}")
    print("[Done] 流式增量处理完成")


if __name__ == "__main__":
    main()
