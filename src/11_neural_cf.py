"""Neural Collaborative Filtering (NCF) 模型

结合 GMF（广义矩阵分解）和 MLP（多层感知机）的深度学习推荐模型。
修复：全局负采样 + 候选集过滤 + 配置段
"""

import os
import sys
import json
import time
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from utils import load_config, ensure_dir

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class NCFDataset(Dataset):
    """NCF 数据集"""

    def __init__(self, user_ids, item_ids, labels):
        self.user_ids = torch.LongTensor(user_ids)
        self.item_ids = torch.LongTensor(item_ids)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.user_ids[idx], self.item_ids[idx], self.labels[idx]


class NCF(nn.Module):
    """Neural Collaborative Filtering 模型"""

    def __init__(self, n_users, n_items, embed_dim=64, mlp_layers=[128, 64, 32], dropout=0.2):
        super(NCF, self).__init__()

        # GMF 部分
        self.gmf_user_embed = nn.Embedding(n_users, embed_dim)
        self.gmf_item_embed = nn.Embedding(n_items, embed_dim)

        # MLP 部分
        self.mlp_user_embed = nn.Embedding(n_users, embed_dim)
        self.mlp_item_embed = nn.Embedding(n_items, embed_dim)

        # MLP 层
        mlp_input_dim = embed_dim * 2
        self.mlp_layers = nn.ModuleList()
        for layer_dim in mlp_layers:
            self.mlp_layers.append(nn.Linear(mlp_input_dim, layer_dim))
            self.mlp_layers.append(nn.ReLU())
            self.mlp_layers.append(nn.Dropout(dropout))
            mlp_input_dim = layer_dim

        # 最终预测层（GMF + MLP 拼接）
        self.output_layer = nn.Linear(embed_dim + mlp_layers[-1], 1)
        self.sigmoid = nn.Sigmoid()

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, user_ids, item_ids):
        # GMF 部分
        gmf_user = self.gmf_user_embed(user_ids)
        gmf_item = self.gmf_item_embed(item_ids)
        gmf_out = gmf_user * gmf_item  # 逐元素乘积

        # MLP 部分
        mlp_user = self.mlp_user_embed(user_ids)
        mlp_item = self.mlp_item_embed(item_ids)
        mlp_out = torch.cat([mlp_user, mlp_item], dim=-1)
        for layer in self.mlp_layers:
            mlp_out = layer(mlp_out)

        # 拼接 GMF + MLP
        concat = torch.cat([gmf_out, mlp_out], dim=-1)
        output = self.output_layer(concat)
        return self.sigmoid(output).squeeze()


def train_ncf(train_data, config):
    """训练 NCF 模型"""
    ncf_config = config.get("ncf", {})
    threshold = config["split"]["pos_threshold"]
    embed_dim = ncf_config.get("embed_dim", 64)
    mlp_layers = ncf_config.get("mlp_layers", [128, 64, 32])
    epochs = ncf_config.get("epochs", 20)
    batch_size = ncf_config.get("batch_size", 256)
    lr = ncf_config.get("learning_rate", 0.001)
    neg_sample_ratio = ncf_config.get("neg_sample_ratio", 1.0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"设备: {device}")

    # 构建用户-物品映射
    pos_data = train_data[train_data["rating"] >= threshold]
    users = train_data["userId"].unique()
    items = pos_data["movieId"].unique()
    user_map = {u: i for i, u in enumerate(users)}
    item_map = {m: i for i, m in enumerate(items)}
    idx_to_item = {i: m for m, i in item_map.items()}

    print(f"用户数: {len(users):,}, 物品数: {len(items):,}")

    # 准备正样本
    pos_user_ids = pos_data["userId"].map(user_map).values
    pos_item_ids = pos_data["movieId"].map(item_map).values
    pos_labels = np.ones(len(pos_data))

    # 全局负采样（修复：避免逐用户创建大集合）
    np.random.seed(config["random_seed"])
    n_neg = int(len(pos_data) * neg_sample_ratio)
    neg_user_ids = np.random.choice(list(user_map.values()), size=n_neg)
    neg_item_ids = np.random.choice(list(item_map.values()), size=n_neg)
    neg_labels = np.zeros(n_neg)

    # 合并正负样本
    all_user_ids = np.concatenate([pos_user_ids, neg_user_ids])
    all_item_ids = np.concatenate([pos_item_ids, neg_item_ids])
    all_labels = np.concatenate([pos_labels, neg_labels])

    # 打乱顺序
    shuffle_idx = np.random.permutation(len(all_labels))
    all_user_ids = all_user_ids[shuffle_idx]
    all_item_ids = all_item_ids[shuffle_idx]
    all_labels = all_labels[shuffle_idx]

    print(f"训练样本: 正样本 {len(pos_labels):,}, 负样本 {n_neg:,}")

    # 创建数据加载器
    dataset = NCFDataset(all_user_ids, all_item_ids, all_labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # 创建模型
    model = NCF(
        n_users=len(users),
        n_items=len(items),
        embed_dim=embed_dim,
        mlp_layers=mlp_layers
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    # 训练
    print(f"训练 NCF: {epochs} epochs, batch_size={batch_size}, lr={lr}")
    start_time = time.time()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        n_batches = 0
        for batch_users, batch_items, batch_labels in dataloader:
            batch_users = batch_users.to(device)
            batch_items = batch_items.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            predictions = model(batch_users, batch_items)
            loss = criterion(predictions, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    train_time = time.time() - start_time
    print(f"训练时间: {train_time:.2f}s")

    return model, user_map, item_map, idx_to_item, users, items, train_time


def generate_recommendations(model, user_map, item_map, idx_to_item, users, items, test_users, config):
    """为测试用户生成推荐（修复：候选集过滤）"""
    device = next(model.parameters()).device
    top_k = config["recommendation"]["top_k"]
    splits_dir = config["data"]["splits_dir"]
    candidates_path = os.path.join(splits_dir, "candidates.json")

    # 加载候选集
    candidates = {}
    if os.path.exists(candidates_path):
        with open(candidates_path, "r") as f:
            cand_raw = json.load(f)
        for uid_str, items_dict in cand_raw.items():
            uid = int(uid_str)
            candidate_items = [item_map[m] for m in items_dict["all"] if m in item_map]
            if candidate_items:
                candidates[uid] = candidate_items
        print(f"加载候选集: {len(candidates)} 用户")

    model.eval()
    recommendations = {}
    n_items = len(items)

    with torch.no_grad():
        for user_id in test_users:
            u_idx = user_map.get(user_id)
            if u_idx is None:
                continue

            # 候选集内评分（修复：不在全量物品上评分）
            if user_id in candidates:
                cand_indices = candidates[user_id]
                item_tensor = torch.LongTensor(cand_indices).to(device)
                user_tensor = torch.LongTensor([u_idx] * len(cand_indices)).to(device)
                scores = model(user_tensor, item_tensor).cpu().numpy()
                # 从候选集中取 top-k
                actual_k = min(top_k, len(cand_indices))
                top_local_idx = np.argsort(-scores)[:actual_k]
                recommendations[str(user_id)] = [
                    {"movieId": int(idx_to_item[cand_indices[i]]), "score": float(scores[i])}
                    for i in top_local_idx
                ]
            else:
                # 无候选集时回退到全量评分
                batch_size = 2048
                all_scores = []
                for i in range(0, n_items, batch_size):
                    batch_items = torch.LongTensor(list(range(i, min(i+batch_size, n_items)))).to(device)
                    batch_users = torch.LongTensor([u_idx] * len(batch_items)).to(device)
                    scores = model(batch_users, batch_items)
                    all_scores.append(scores.cpu().numpy())
                all_scores = np.concatenate(all_scores)
                top_indices = np.argsort(-all_scores)[:top_k]
                recommendations[str(user_id)] = [
                    {"movieId": int(idx_to_item[idx]), "score": float(all_scores[idx])}
                    for idx in top_indices
                ]

    return recommendations


def main():
    config = load_config()
    splits_dir = config["data"]["splits_dir"]
    train_path = os.path.join(splits_dir, "train.csv")
    test_path = os.path.join(splits_dir, "test.csv")

    if not os.path.exists(train_path):
        print("[ERROR] 请先运行 02_data_split.py")
        sys.exit(1)

    train = pd.read_csv(train_path)
    print(f"训练集: {len(train):,} 条")

    # 训练 NCF
    model, user_map, item_map, idx_to_item, users, items, train_time = train_ncf(train, config)

    # 加载测试集用户
    test_users = None
    if os.path.exists(test_path):
        test_users = pd.read_csv(test_path)["userId"].unique().tolist()
        print(f"测试用户: {len(test_users):,}")

    # 生成推荐
    recommendations = generate_recommendations(model, user_map, item_map, idx_to_item, users, items, test_users, config)

    # 保存结果
    ensure_dir("results")
    output = {
        "train_time": train_time,
        "num_users": len(recommendations),
        "recommendations": recommendations,
    }

    with open("results/ncf_recommendations.json", "w") as f:
        json.dump(output, f)

    print(f"已生成 {len(recommendations)} 位用户的推荐")
    print("结果已保存到 results/ncf_recommendations.json")
    print("[Done] NCF 训练完成")


if __name__ == "__main__":
    main()