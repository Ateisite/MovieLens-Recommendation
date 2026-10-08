# MovieLens-Recommendation

基于 MovieLens 32M 数据集的分布式增量电影推荐系统。使用 Spark、LSH、图分析等技术构建完整推荐流水线，NDCG@10 相对热门推荐基线提升 >= 10%。

## 技术栈

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![PySpark](https://img.shields.io/badge/PySpark-3.5+-orange) ![NetworkX](https://img.shields.io/badge/NetworkX-3.1+-green) ![scikit--learn](https://img.shields.io/badge/scikit--learn-1.3+-red)

- **数据处理**: Pandas, NumPy, PySpark
- **推荐算法**: UserCF, ItemCF, ALS 矩阵分解
- **近似召回**: LSH (datasketch)
- **图分析**: NetworkX (二部图、社区发现)
- **评价指标**: NDCG@10, Recall@10

## 数据集

[MovieLens 32M](https://grouplens.org/datasets/movielens/32m/) - 约 3200 万条评分、200 万条标签、20 万用户、8.7 万部电影。

下载后将 atings.csv, movies.csv, 	ags.csv, links.csv 放入 data/raw/ 目录。

## 快速开始

### 1. 安装依赖

`ash
pip install -r requirements.txt
`

### 2. 下载数据

从 [MovieLens 32M](https://grouplens.org/datasets/movielens/32m/) 下载并解压到 data/raw/。

### 3. 运行

`ash
# 一键运行完整流水线
python run_all.py

# 或分步执行
python src/01_data_check.py    # 数据检查与基线
python src/02_data_split.py    # 数据划分
python src/03_usercf.py        # UserCF
python src/04_itemcf.py        # ItemCF
python src/05_lsh_recall.py    # LSH 召回
python src/06_spark_als.py     # Spark ALS
python src/07_streaming.py     # 流式增量处理
python src/08_graph_rec.py     # 图推荐
python src/09_evaluate.py      # 评价指标
python src/10_pipeline.py      # 系统集成
`

## 项目结构

`
MovieLens-Recommendation/
├── README.md
├── requirements.txt
├── config.yaml
├── data/
│   ├── raw/              # 原始数据
│   ├── processed/        # 清洗后数据
│   └── splits/           # 训练/验证/测试划分
├── src/
│   ├── 01_data_check.py  # 数据检查与基线
│   ├── 02_data_split.py  # 数据划分
│   ├── 03_usercf.py      # UserCF
│   ├── 04_itemcf.py      # ItemCF
│   ├── 05_lsh_recall.py  # LSH 召回
│   ├── 06_spark_als.py   # Spark ALS
│   ├── 07_streaming.py   # 流式增量处理
│   ├── 08_graph_rec.py   # 图推荐
│   ├── 09_evaluate.py    # 评价指标计算
│   └── 10_pipeline.py    # 系统集成
├── results/              # 实验结果
├── reports/              # 项目报告
└── run_all.py            # 一键运行脚本
`

## 实验结果

| 模型 | NDCG@10 | Recall@10 |
|------|---------|-----------|
| 热门推荐基线 | - | - |
| UserCF | - | - |
| ItemCF | - | - |
| ALS (Spark) | - | - |
| LSH 召回 + 排序 | - | - |
| 图推荐 | - | - |
| 最终集成模型 | - | - |

## 许可

本项目仅用于课程学习与求职展示。
