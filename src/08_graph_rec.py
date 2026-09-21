"""任务五：图推荐实验

建立用户-电影二部图，使用社区发现和 PageRank 改进推荐。
"""

import os
import sys
import yaml
import json
import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict
from utils import load_config, ensure_dir


def build_bipartite_graph(ratings, config):
    """构建用户-电影二部图"""
    threshold = config["split"]["pos_threshold"]
    pos_ratings = ratings[ratings["rating"] >= threshold]

    B = nx.Graph()

    users = set(pos_ratings["userId"].unique())
    movies = set(pos_ratings["movieId"].unique())

    B.add_nodes_from(users, bipartite=0, node_type="user")
    B.add_nodes_from(movies, bipartite=1, node_type="movie")

    edges = list(zip(pos_ratings["userId"], pos_ratings["movieId"]))
    B.add_edges_from(edges)

    print(f"二部图: {len(users)} 用户 + {len(movies)} 电影 = {B.number_of_edges()} 边")
    print(f"图密度: {nx.density(B):.6f}")
    return B, users, movies


def community_detection(B, users, config):
    """社区发现（标签传播算法）"""
    # 使用连通分量作为社区（对于二部图）
    # 也可以使用更复杂的算法如 Louvain（需要 python-louvain 库）
    communities = list(nx.connected_components(B))

    # 将社区分配给用户节点
    user_communities = {}
    for comm_id, comm in enumerate(communities):
        for node in comm:
            if node in users:
                user_communities[node] = comm_id

    print(f"社区数量: {len(communities)}")
    print(f"  最大社区大小: {max(len(c) for c in communities)}")
    print(f"  平均社区大小: {np.mean([len(c) for c in communities]):.1f}")

    return communities, user_communities


def pagerank_features(B, config):
    """计算 PageRank 作为节点重要性特征"""
    pr = nx.pagerank(B, alpha=0.85)

    # 分离用户和电影的 PageRank 值
    user_pr = {k: v for k, v in pr.items() if B.nodes[k].get("node_type") == "user"}
    movie_pr = {k: v for k, v in pr.items() if B.nodes[k].get("node_type") == "movie"}

    print(f"PageRank 计算完成")
    print(f"  用户 PageRank - 均值: {np.mean(list(user_pr.values())):.6f}, 最大: {max(user_pr.values()):.6f}")
    print(f"  电影 PageRank - 均值: {np.mean(list(movie_pr.values())):.6f}, 最大: {max(movie_pr.values()):.6f}")

    return user_pr, movie_pr


def graph_based_recommend(B, user_id, movie_pr, user_communities, top_k=10):
    """基于图特征的推荐
    
    策略：
    1. 获取用户已交互的电影
    2. 找到这些电影的邻居用户（同社区优先）
    3. 这些用户交互过的其他电影按 PageRank 排序
    """
    if user_id not in B:
        return []

    # 用户已交互的电影
    watched = set(B.neighbors(user_id))
    user_comm = user_communities.get(user_id, -1)

    # 候选电影：邻居用户看过的电影
    candidates = defaultdict(float)
    for neighbor in B.neighbors(user_id):
        for movie in B.neighbors(neighbor):
            if movie not in watched:
                # 分数 = 邻居的 PageRank * 电影 PageRank
                score = movie_pr.get(movie, 0)
                candidates[movie] += score

    # 排序取 Top-K
    sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    return sorted_candidates[:top_k]


def recommend_batch_graph(B, users, movie_pr, user_communities, config, sample_size=500):
    """批量生成图推荐"""
    np.random.seed(config["random_seed"])
    sample_users = np.random.choice(list(users), size=min(sample_size, len(users)), replace=False)

    all_recommendations = {}
    for user_id in sample_users:
        recs = graph_based_recommend(B, user_id, movie_pr, user_communities)
        if recs:
            all_recommendations[user_id] = recs

    return all_recommendations


def main():
    config = load_config()
    splits_dir = config["data"]["splits_dir"]
    train_path = os.path.join(splits_dir, "train.csv")

    if not os.path.exists(train_path):
        print("[ERROR] 请先运行 02_data_split.py")
        sys.exit(1)

    train = pd.read_csv(train_path)
    B, users, movies = build_bipartite_graph(train, config)

    # 社区发现
    communities, user_communities = community_detection(B, users, config)

    # PageRank
    user_pr, movie_pr = pagerank_features(B, config)

    # 图推荐
    recommendations = recommend_batch_graph(B, users, movie_pr, user_communities, config)

    # 保存结果
    ensure_dir("results")
    output = {
        "num_communities": len(communities),
        "num_users_with_recs": len(recommendations),
        "sample_recommendations": {
            str(k): [(int(m), float(s)) for m, s in v[:5]]
            for k, v in list(recommendations.items())[:20]
        },
    }
    with open("results/graph_recommendations.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    # 保存 PageRank 特征
    pd.DataFrame(
        [{"movieId": k, "pagerank": v} for k, v in movie_pr.items()]
    ).to_csv("results/movie_pagerank.csv", index=False)

    print(f"\n已生成 {len(recommendations)} 位用户的图推荐")
    print("结果已保存到 results/graph_recommendations.json")
    print("[Done] 图推荐实验完成")


if __name__ == "__main__":
    main()
