import pandas as pd, numpy as np, yaml

config = yaml.safe_load(open('config.yaml'))
raw_dir = config['data']['raw_dir']
ratings = pd.read_csv(raw_dir + '/' + config['data']['ratings_file'])

threshold = config['split']['pos_threshold']
top_k = config['recommendation']['top_k']

# 当前代码：在全量数据上算基线
pos_all = ratings[ratings['rating'] >= threshold]
popular = pos_all.groupby('movieId').size().sort_values(ascending=False).head(top_k).index.tolist()
user_pos_all = pos_all.groupby('userId')['movieId'].apply(set).to_dict()

ndcg_all = []
for uid, pos in user_pos_all.items():
    rel = [1 if m in pos else 0 for m in popular]
    dcg = sum((2**r - 1) / np.log2(i + 2) for i, r in enumerate(rel[:top_k]))
    ideal = sorted(rel, reverse=True)[:top_k]
    idcg = sum((2**r - 1) / np.log2(i + 2) for i, r in enumerate(ideal))
    if idcg > 0:
        ndcg_all.append(dcg / idcg)

print(f'全量数据基线: NDCG@10 = {np.mean(ndcg_all):.4f}, 用户数={len(user_pos_all)}')
print(f'热门电影前5: {popular[:5]}')
