"""任务五：图推荐实验

建立用户-电影二部图，使用链接分析或社区特征改进推荐。
"""

import os
import sys
import yaml
import pandas as pd
import networkx as nx


def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_bipartite_graph(ratings, config):
    """构建用户-电影二部图"""
    threshold = config["split"]["pos_threshold"]
    pos_ratings = ratings[ratings["rating"] >= threshold]

    B = nx.Graph()
    users = set(pos_ratings["userId"].unique())
    movies = set(pos_ratings["movieId"].unique())

    B.add_nodes_from(users, bipartite=0)
    B.add_nodes_from(movies, bipartite=1)

    edges = list(zip(pos_ratings["userId"], pos_ratings["movieId"]))
    B.add_edges_from(edges)

    print(f"二部图: {len(users)} 用户 + {len(movies)} 电影 = {len(edges)} 边")
    return B


def community_detection(B, config):
    """社区发现"""
    # TODO: 使用 Louvain 或标签传播算法
    pass


def pagerank_recommend(B, user_id, top_k=10):
    """基于 PageRank 的推荐"""
    pr = nx.pagerank(B)
    # TODO: 筛选用户未交互的电影
    pass


def main():
    config = load_config()
    splits_dir = config["data"]["splits_dir"]
    train_path = os.path.join(splits_dir, "train.csv")

    if not os.path.exists(train_path):
        print("[ERROR] 请先运行 02_data_split.py")
        sys.exit(1)

    train = pd.read_csv(train_path)
    B = build_bipartite_graph(train, config)

    print(f"图密度: {nx.density(B):.6f}")
    print("[Done] 图推荐实验框架就绪")


if __name__ == "__main__":
    main()
