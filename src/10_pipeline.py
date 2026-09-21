"""任务六：系统集成

组合召回、特征和排序模块，通过消融实验解释性能变化。
"""

import os
import sys
import yaml


def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_pipeline(config):
    """运行完整推荐流水线"""
    print("=== 推荐系统流水线 ===")
    print("1. 数据检查...")
    # os.system("python src/01_data_check.py")

    print("2. 数据划分...")
    # os.system("python src/02_data_split.py")

    print("3. 召回阶段 (UserCF + ItemCF + LSH)...")
    # TODO: 多路召回融合

    print("4. 排序阶段 (ALS + 特征工程)...")
    # TODO: 精排模型

    print("5. 生成 Top-10 推荐...")

    print("6. 评价指标计算...")

    print("7. 消融实验...")


def ablation_study(config):
    """消融实验"""
    experiments = {
        "baseline": "仅热门推荐",
        "+usercf": "加入 UserCF",
        "+itemcf": "加入 ItemCF",
        "+lsh": "加入 LSH 召回",
        "+als": "加入 ALS 排序",
        "+graph": "加入图特征",
    }
    for name, desc in experiments.items():
        print(f"  {name}: {desc}")


def main():
    config = load_config()
    run_pipeline(config)
    ablation_study(config)
    print("\n[Done] 系统集成框架就绪")


if __name__ == "__main__":
    main()
