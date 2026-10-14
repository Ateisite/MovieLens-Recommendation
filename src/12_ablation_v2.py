"""消融实验 V2：使用候选集过滤，大幅加速"""

import os, sys, json, time, yaml
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from utils import load_config, ensure_dir


class NCFDataset(Dataset):
    def __init__(self, user_ids, item_ids, labels):
        self.user_ids = torch.LongTensor(user_ids)
        self.item_ids = torch.LongTensor(item_ids)
        self.labels = torch.FloatTensor(labels)

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        return self.user_ids[idx], self.item_ids[idx], self.labels[idx]


class NCF(nn.Module):
    def __init__(self, n_users, n_items, embed_dim=32, mlp_layers=[64, 32], dropout=0.2):
        super().__init__()
        self.gmf_user = nn.Embedding(n_users, embed_dim)
        self.gmf_item = nn.Embedding(n_items, embed_dim)
        self.mlp_user = nn.Embedding(n_users, embed_dim)
        self.mlp_item = nn.Embedding(n_items, embed_dim)
        mlp_dim = embed_dim * 2
        self.mlp = nn.ModuleList()
        for l in mlp_layers:
            self.mlp.append(nn.Linear(mlp_dim, l))
            self.mlp.append(nn.ReLU())
            self.mlp.append(nn.Dropout(dropout))
            mlp_dim = l
        self.output = nn.Linear(embed_dim + mlp_layers[-1], 1)
        self.sigmoid = nn.Sigmoid()
        for m in self.modules():
            if isinstance(m, nn.Embedding): nn.init.normal_(m.weight, std=0.01)
            elif isinstance(m, nn.Linear): nn.init.xavier_uniform_(m.weight)

    def forward(self, u, i):
        gmf = self.gmf_user(u) * self.gmf_item(i)
        mlp = torch.cat([self.mlp_user(u), self.mlp_item(i)], dim=-1)
        for layer in self.mlp: mlp = layer(mlp)
        return self.sigmoid(self.output(torch.cat([gmf, mlp], dim=-1))).squeeze()


def run(mode, epochs=15):
    config = load_config()
    ncf_cfg = config.get("ncf", {})
    threshold = config["split"]["split"]["pos_threshold"] if "split" in config.get("ncf", {}) else config["split"]["pos_threshold"]
    k = config["recommendation"]["top_k"]
    device = torch.device("cpu")

    splits_dir = config["data"]["splits_dir"]
    train = pd.read_csv(os.path.join(splits_dir, "train.csv"), usecols=["userId","movieId","rating"])
    test = pd.read_csv(os.path.join(splits_dir, "test.csv"), usecols=["userId","movieId","rating"])

    # 加载候选集
    cand_path = os.path.join(splits_dir, "candidates.json")
    with open(cand_path) as f:
        cand_raw = json.load(f)
    candidates = {}
    for uid_str, items_dict in cand_raw.items():
        candidates[int(uid_str)] = items_dict["all"]

    # 映射
    pos_data = train[train["rating"] >= threshold]
    users = train["userId"].unique()
    items = pos_data["movieId"].unique()
    user_map = {u: i for i, u in enumerate(users)}
    item_map = {m: i for i, m in enumerate(items)}
    idx_to_item = {i: m for m, i in item_map.items()}

    # 正样本
    pos_u = pos_data["userId"].map(user_map).values
    pos_i = pos_data["movieId"].map(item_map).values

    # 全局负采样
    np.random.seed(42)
    n_neg = len(pos_data)
    neg_u = np.random.choice(list(user_map.values()), size=n_neg)
    neg_i = np.random.choice(list(item_map.values()), size=n_neg)

    all_u = np.concatenate([pos_u, neg_u])
    all_i = np.concatenate([pos_i, neg_i])
    all_l = np.concatenate([np.ones(len(pos_u)), np.zeros(n_neg)])
    idx = np.random.permutation(len(all_l))
    all_u, all_i, all_l = all_u[idx], all_i[idx], all_l[idx]

    dataset = NCFDataset(all_u, all_i, all_l)
    loader = DataLoader(dataset, batch_size=512, shuffle=True)

    model = NCF(len(users), len(items), embed_dim=32, mlp_layers=[64, 32]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    # 训练
    print(f"  训练 {mode}...")
    model.train()
    for epoch in range(epochs):
        total_loss, n_b = 0, 0
        for bu, bi, bl in loader:
            bu, bi, bl = bu.to(device), bi.to(device), bl.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bu, bi), bl)
            loss.backward()
            optimizer.step()
            total_loss += loss.item(); n_b += 1
        if (epoch+1) % 5 == 0:
            print(f"    Epoch {epoch+1}/{epochs}, Loss: {total_loss/n_b:.4f}")

    # 推荐（候选集内）
    model.eval()
    recs = {}
    with torch.no_grad():
        test_users = test["userId"].unique()
        for uid in test_users:
            u_idx = user_map.get(uid)
            if u_idx is None: continue
            cand_items = candidates.get(uid, [])
            cand_idx = [item_map[m] for m in cand_items if m in item_map]
            if not cand_idx: continue
            item_t = torch.LongTensor(cand_idx).to(device)
            user_t = torch.LongTensor([u_idx] * len(cand_idx)).to(device)
            scores = model(user_t, item_t).cpu().numpy()
            top_k_idx = np.argsort(-scores)[:k]
            recs[str(uid)] = [{"movieId": int(idx_to_item[cand_idx[i]]), "score": float(scores[i])} for i in top_k_idx]

    # 评估
    user_pos = test[test["rating"] >= threshold].groupby("userId")["movieId"].apply(set).to_dict()
    ndcg_list = []
    import math
    for uid_str, items_list in recs.items():
        uid = int(uid_str)
        rel = user_pos.get(uid, set())
        if not rel: continue
        rec_ids = [i["movieId"] for i in items_list]
        rel_vec = [1 if i in rel else 0 for i in rec_ids[:10]]
        dcg = sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(rel_vec))
        idcg = sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(sorted(rel_vec, reverse=True)))
        if idcg > 0: ndcg_list.append(dcg / idcg)

    ndcg = np.mean(ndcg_list) if ndcg_list else 0
    print(f"  {mode}: NDCG@10 = {ndcg:.4f} ({len(ndcg_list)} users)")
    return {"mode": mode, "ndcg": round(ndcg, 4)}


def main():
    results = {}
    for mode in ["ncf"]:
        results[mode] = run(mode, epochs=15)

    ensure_dir("results")
    with open("results/ablation_v2.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n结果: {results}")


if __name__ == "__main__":
    main()