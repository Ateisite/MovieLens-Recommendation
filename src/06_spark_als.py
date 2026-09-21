"""任务三（分布式）：Spark ALS 矩阵分解

使用 PySpark MLlib 的 ALS 算法进行分布式推荐。
完整流程：加载数据 -> 训练模型 -> 生成推荐 -> 保存结果。
"""

import os
import sys
import yaml
import json
import time
from utils import load_config, ensure_dir


def main():
    config = load_config()
    splits_dir = config["data"]["splits_dir"]
    train_path = os.path.join(splits_dir, "train.csv")
    test_path = os.path.join(splits_dir, "test.csv")

    if not os.path.exists(train_path):
        print("[ERROR] 请先运行 02_data_split.py")
        sys.exit(1)

    try:
        from pyspark.sql import SparkSession
        from pyspark.ml.recommendation import ALS
        from pyspark.ml.evaluation import RegressionEvaluator
        from pyspark.sql.functions import col, explode
    except ImportError:
        print("[ERROR] PySpark 未安装，请运行: pip install pyspark")
        sys.exit(1)

    als_config = config["als"]
    top_k = config["recommendation"]["top_k"]

    # 创建 Spark Session
    spark = SparkSession.builder \
        .appName("MovieLens-ALS") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "8") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    print("=== Spark ALS 训练 ===")
    print(f"参数: rank={als_config['rank']}, maxIter={als_config['max_iter']}, regParam={als_config['reg_param']}")

    # 加载数据
    train_df = spark.read.csv(train_path, header=True, inferSchema=True)
    train_df = train_df.select(
        col("userId").cast("integer"),
        col("movieId").cast("integer"),
        col("rating").cast("float"),
    )

    if os.path.exists(test_path):
        test_df = spark.read.csv(test_path, header=True, inferSchema=True)
        test_df = test_df.select(
            col("userId").cast("integer"),
            col("movieId").cast("integer"),
            col("rating").cast("float"),
        )
    else:
        test_df = None

    print(f"训练集: {train_df.count():,} 条")
    if test_df:
        print(f"测试集: {test_df.count():,} 条")

    # 训练 ALS 模型
    start_time = time.time()

    als = ALS(
        rank=als_config["rank"],
        maxIter=als_config["max_iter"],
        regParam=als_config["reg_param"],
        alpha=als_config["alpha"],
        userCol="userId",
        itemCol="movieId",
        ratingCol="rating",
        coldStartStrategy="drop",
        implicitPrefs=False,
    )

    model = als.fit(train_df)
    train_time = time.time() - start_time
    print(f"训练时间: {train_time:.2f}s")

    # 评估
    if test_df:
        predictions = model.transform(test_df)
        evaluator = RegressionEvaluator(
            metricName="rmse",
            labelCol="rating",
            predictionCol="prediction",
        )
        rmse = evaluator.evaluate(predictions)
        print(f"测试集 RMSE: {rmse:.4f}")

    # 为所有用户生成 Top-K 推荐
    user_recs = model.recommendForAllUsers(top_k)

    # 展开推荐结果
    recs_exploded = user_recs.select(
        "userId",
        explode("recommendations").alias("rec"),
    ).select(
        "userId",
        col("rec.movieId").alias("movieId"),
        col("rec.rating").alias("predicted_rating"),
    )

    # 收集结果
    recs_pd = recs_exploded.toPandas()

    # 保存结果
    ensure_dir("results")
    recs_dict = {}
    for user_id, group in recs_pd.groupby("userId"):
        recs_dict[str(user_id)] = [
            {"movieId": int(row["movieId"]), "score": float(row["predicted_rating"])}
            for _, row in group.iterrows()
        ]

    output = {
        "train_time": train_time,
        "rmse": rmse if test_df else None,
        "num_users": len(recs_dict),
        "recommendations": recs_dict,
    }

    with open("results/als_recommendations.json", "w") as f:
        json.dump(output, f)

    print(f"已生成 {len(recs_dict)} 位用户的推荐")
    print("结果已保存到 results/als_recommendations.json")

    spark.stop()
    print("[Done] Spark ALS 完成")


if __name__ == "__main__":
    main()
