"""任务二：LSH 近似候选召回

使用局部敏感哈希（Locality-Sensitive Hashing）加速相似度计算。
"""

import os
import sys
import yaml
import numpy as np
from datasketch import MinHash, MinHashLSH


def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_lsh_index(user_item_sets, config):
    """构建 LSH 索引"""
    lsh_config = config["lsh"]
    lsh = MinHashLSH(threshold=lsh_config["threshold"], num_perm=lsh_config["num_perm"])

    minhashes = {}
    for user_id, items in user_item_sets.items():
        m = MinHash(num_perm=lsh_config["num_perm"])
        for item in items:
            m.update(str(item).encode("utf8"))
        lsh.insert(user_id, m)
        minhashes[user_id] = m

    return lsh, minhashes


def query_candidates(lsh, minhashes, user_id, config):
    """查询近似最近邻候选"""
    if user_id not in minhashes:
        return []
    result = lsh.query(minhashes[user_id])
    return result


def evaluate_lsh(exact_sim, lsh_results, sample_size=1000):
    """评估 LSH 的准确率与效率"""
    # TODO: 实现准确率对比
    pass


def main():
    config = load_config()
    splits_dir = config["data"]["splits_dir"]
    train_path = os.path.join(splits_dir, "train.csv")

    if not os.path.exists(train_path):
        print("[ERROR] 请先运行 02_data_split.py")
        sys.exit(1)

    print("[TODO] LSH 召回模块待实现")
    print("[Done] 框架就绪")


if __name__ == "__main__":
    main()
