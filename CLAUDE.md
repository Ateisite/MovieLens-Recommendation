# MovieLens 推荐系统项目

> **修改记录：**
> - 2026-10-08 | Claude (Opus 5) | 初始创建，包含基础流水线结果
> - 2026-10-08 | GPT (Codex) | 全面代码审查 + 根因分析，定位 NDCG/Recall 极低问题
> - 2026-10-08 | Claude (Opus 5) | 修复推荐生成逻辑：改为只为测试集用户生成推荐，覆盖率从个位数提升到 29K/29K

## 项目概述

基于 MovieLens 32M 数据集的智能电影推荐系统。核心目标：NDCG@10 相对热门推荐基线提升 ≥ 10%。

**当前状态：** 基础流水线已跑通，正在向 Agent 架构改造。

## 技术栈

- **数据处理**: Pandas, NumPy, PySpark
- **推荐算法**: UserCF, ItemCF, ALS (implicit 库), LSH
- **图分析**: NetworkX (二部图、PageRank)
- **Agent**: LangChain (Tool Use + RAG + Memory)
- **服务**: FastAPI
- **可视化**: Matplotlib, Seaborn

## 项目结构

```
MovieLens-Recommendation/
├── README.md                  # 项目说明
├── config.yaml                # 配置文件（参数、路径、种子）
├── requirements.txt           # Python 依赖
├── data/
│   ├── raw/                   # 原始数据（不提交到 git）
│   ├── processed/             # 清洗后数据
│   └── splits/                # 训练/验证/测试划分
├── src/
│   ├── 01_data_check.py       # 数据检查与基线
│   ├── 02_data_split.py       # 数据划分（时序划分+负采样）
│   ├── 03_usercf.py           # UserCF（分批计算，避免内存爆炸）
│   ├── 04_itemcf.py           # ItemCF
│   ├── 05_lsh_recall.py       # LSH 近似召回
│   ├── 06_spark_als.py        # ALS 矩阵分解（implicit 库，无需 Java）
│   ├── 07_streaming.py        # 流式增量处理
│   ├── 08_graph_rec.py        # 图推荐
│   ├── 09_evaluate.py         # 评价指标（NDCG@K, Recall@K）
│   ├── 10_pipeline.py         # 系统集成
│   ├── 11_neural_cf.py        # (待实现) 深度学习模型
│   ├── agent.py               # (待实现) Agent 主逻辑
│   ├── tools/                 # (待实现) Agent 工具
│   │   ├── cf_tool.py
│   │   ├── semantic_tool.py
│   │   └── detail_tool.py
│   ├── memory.py              # (待实现) 用户记忆模块
│   └── api.py                 # (待实现) FastAPI 服务
├── results/                   # 实验结果（不提交到 git）
├── reports/                   # 报告和图表
└── 学习与升级指南.md           # 学习文档
```

## 运行方式

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 下载数据

从 https://grouplens.org/datasets/movielens/32m/ 下载 `ml-32m.zip`，解压到 `data/raw/`。

### 3. 运行

```bash
# 一键运行完整流水线
python run_all.py

# 或分步执行
python src/01_data_check.py
python src/02_data_split.py
python src/03_usercf.py
# ... 以此类推
```

## 关键设计决策

### 数据划分
- **时序划分**：每个用户按时间戳排序，前 80% 训练、中 10% 验证、后 10% 测试
- **负采样**：每个正样本配 100 个随机负样本（种子固定为 20260907）
- **采样**：config.yaml 中 `sample_ratings: 1000000` 控制数据量，便于快速演示

### UserCF 内存优化
- 分批计算相似度矩阵（每次 5000 用户），避免 63GB 内存爆炸
- 全程使用稀疏矩阵（scipy.sparse.csr_matrix）
- Top-K 剪枝：每个用户只保留最相似的 50 个邻居

### ALS 实现
- 使用 `implicit` 库而非 PySpark（无需 Java 环境）
- 训练时间约 7 秒（100 万条数据）

### 评价指标
- NDCG@10：核心指标，衡量排序质量
- Recall@10：Top-10 推荐中命中正样本的比例
- 基线：热门推荐 NDCG@10 = 0.7344

## 当前结果

| 模型 | NDCG@10 | Recall@10 | 覆盖用户 |
|------|---------|-----------|----------|
| 基线（热门） | 0.7344 | - | - |
| UserCF | 0.0022 | 0.0032 | 19224/29920 |
| ItemCF | 0.0002 | 0.0003 | 29448/29920 |
| ALS | 0.0070 | 0.0127 | 29470/29920 |
| Graph | 0.0049 | 0.0076 | 29504/29920 |

**问题：** NDCG 远低于基线，主要原因是数据稀疏（100 万采样 + 80/20 时序划分后正反馈极少）。

**修复历史：**
- 已修复：推荐生成改为为测试集用户生成（覆盖率从个位数提升到 29K/29K）
- 待修复：增加数据量、调整正反馈阈值、模型调参

## 下一步（Agent 改造）

1. **修复评价逻辑** —— 只为测试集用户生成推荐
2. **实现 Agent 架构** —— Tool Use + RAG + Memory
3. **添加可视化** —— 模型对比图、消融实验
4. **实现 API 服务** —— FastAPI 推荐接口

详见 `学习与升级指南.md`。

## 常见问题

### JSON 序列化错误
numpy.int64 不能直接序列化为 JSON，需要转为 Python 原生类型：
```python
# 错误
json.dump(recs, f)

# 正确
recs_clean = {str(k): [(int(m), float(s)) for m, s in v] for k, v in recs.items()}
json.dump(recs_clean, f)
```

### 内存不足
- 使用 `config.yaml` 中的 `sample_ratings` 控制数据量
- UserCF 已做分批计算优化

### ALS 无法运行
- 确保安装了 `implicit` 库：`pip install implicit`
- 不需要 Java 环境

## Git 规范

- 数据文件（`data/raw/*.csv`, `data/splits/*.csv`）不提交
- 结果文件（`results/*.json`, `results/*.csv`）不提交
- 提交前确保代码能跑通
