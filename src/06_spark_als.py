"""任务三（分布式）：Spark ALS 矩阵分解

使用 PySpark MLlib 的 ALS 算法进行分布式推荐。
"""

import os
import sys
import yaml


def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    splits_dir = config["data"]["splits_dir"]
    train_path = os.path.join(splits_dir, "train.csv")

    if not os.path.exists(train_path):
        print("[ERROR] 请先运行 02_data_split.py")
        sys.exit(1)

    try:
        from pyspark.sql import SparkSession
        from pyspark.ml.recommendation import ALS
        from pyspark.ml.evaluation import RegressionEvaluator
    except ImportError:
        print("[ERROR] PySpark 未安装，请运行: pip install pyspark")
        sys.exit(1)

    als_config = config["als"]

    spark = SparkSession.builder \
        .appName("MovieLens-ALS") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()

    # TODO: 加载数据、训练 ALS 模型、生成推荐
    print(f"ALS 参数: rank={als_config['rank']}, maxIter={als_config['max_iter']}")

    spark.stop()
    print("[Done] Spark ALS 框架就绪")


if __name__ == "__main__":
    main()
