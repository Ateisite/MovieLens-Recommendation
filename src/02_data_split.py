"""数据划分

按每个用户的时间戳排序划分训练/验证/测试集，防止数据泄漏。
生成候选集：测试正样本 + 固定负样本（随机种子 20260907）。
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
from utils import load_config, check_data_files, ensure_dir

import warnings
warnings.filterwarnings("ignore")


def temporal_split_per_user(ratings, config):
    """按每个用户的时间顺序划分数据
    
    每位用户按时间戳排序：前 80% 训练 / 中 10% 验证 / 后 10% 测试
    """
    train_ratio = config["split"]["train_ratio"]
    val_ratio = config["split"]["val_ratio"]

    train_list, val_list, test_list = [], [], []

    for user_id, group in ratings.groupby("userId"):
        group = group.sort_values("timestamp")
        n = len(group)
        train_end = max(1, int(n * train_ratio))
        val_end = max(train_end + 1, int(n * (train_ratio + val_ratio)))

        train_list.append(group.iloc[:train_end])
        if train_end < n:
            val_list.append(group.iloc[train_end:min(val_end, n)])
        if val_end < n:
            test_list.append(group.iloc[val_end:])

    train = pd.concat(train_list, ignore_index=True)
    val = pd.concat(val_list, ignore_index=True) if val_list else pd.DataFrame()
    test = pd.concat(test_list, ignore_index=True) if test_list else pd.DataFrame()

    print(f"训练集: {len(train):,} 条")
    print(f"验证集: {len(val):,} 条")
    print(f"测试集: {len(test):,} 条")
    return train, val, test


def generate_candidates_with_negatives(train, test, config):
    """生成候选集：测试正样本 + 100 个固定负样本
    
    正反馈定义: 评分 >= 4
    负样本随机种子: 20260907
    负样本数量: 100
    """
    np.random.seed(config["split"]["neg_sample_seed"])
    threshold = config["split"]["pos_threshold"]
    neg_count = config["split"]["neg_sample_count"]

    # 训练集中所有电影
    all_movies = set(train["movieId"].unique())

    # 每位测试用户的候选集
    candidates = {}
    test_users = test["userId"].unique()

    for user_id in test_users:
        user_test = test[test["userId"] == user_id]
        pos_items = set(user_test[user_test["rating"] >= threshold]["movieId"])

        # 用户已交互的电影（训练集 + 测试集）
        user_train_movies = set(train[train["userId"] == user_id]["movieId"])
        interacted = user_train_movies | set(user_test["movieId"])

        # 负采样：从用户未交互的电影中随机选取
        neg_candidates = list(all_movies - interacted)
        if len(neg_candidates) <= neg_count:
            neg_items = neg_candidates
        else:
            neg_items = np.random.choice(neg_candidates, size=neg_count, replace=False).tolist()

        candidates[user_id] = {
            "positive": list(pos_items),
            "negative": neg_items,
            "all": list(pos_items) + neg_items,
        }

    print(f"生成候选集: {len(candidates)} 位用户")
    print(f"  平均正样本: {np.mean([len(c['positive']) for c in candidates.values()]):.1f}")
    print(f"  负样本数: {neg_count}")

    return candidates


def main():
    config = load_config()
    if not check_data_files(config):
        sys.exit(1)

    splits_dir = config["data"]["splits_dir"]
    ensure_dir(splits_dir)

    raw_dir = config["data"]["raw_dir"]
    ratings = pd.read_csv(os.path.join(raw_dir, config["data"]["ratings_file"]))

    train, val, test = temporal_split_per_user(ratings, config)

    train.to_csv(os.path.join(splits_dir, "train.csv"), index=False)
    val.to_csv(os.path.join(splits_dir, "val.csv"), index=False)
    test.to_csv(os.path.join(splits_dir, "test.csv"), index=False)

    # 生成候选集
    candidates = generate_candidates_with_negatives(train, test, config)

    import json
    with open(os.path.join(splits_dir, "candidates.json"), "w") as f:
        json.dump(candidates, f)

    print("\n[Done] 数据划分与候选集生成完成")


if __name__ == "__main__":
    main()
