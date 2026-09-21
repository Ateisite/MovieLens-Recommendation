"""一键运行完整推荐流水线

依次执行所有任务模块，生成推荐结果和评价报告。
"""

import os
import sys
import subprocess
import yaml
from datetime import datetime


def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_step(step_name, script_path):
    """执行单个步骤"""
    print(f"\n{'='*60}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {step_name}")
    print(f"{'='*60}")

    if not os.path.exists(script_path):
        print(f"[SKIP] 脚本不存在: {script_path}")
        return True

    result = subprocess.run(
        [sys.executable, script_path],
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    if result.returncode != 0:
        print(f"[ERROR] {step_name} 执行失败")
        return False
    return True


def main():
    config = load_config()

    steps = [
        ("任务一：数据检查与基线", "src/01_data_check.py"),
        ("数据划分", "src/02_data_split.py"),
        ("UserCF", "src/03_usercf.py"),
        ("ItemCF", "src/04_itemcf.py"),
        ("LSH 召回", "src/05_lsh_recall.py"),
        ("Spark ALS", "src/06_spark_als.py"),
        ("流式增量处理", "src/07_streaming.py"),
        ("图推荐", "src/08_graph_rec.py"),
        ("评价指标", "src/09_evaluate.py"),
        ("系统集成", "src/10_pipeline.py"),
    ]

    print("MovieLens 分布式增量电影推荐系统")
    print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    success_count = 0
    for step_name, script_path in steps:
        if run_step(step_name, script_path):
            success_count += 1
        else:
            print(f"\n流水线中断于: {step_name}")
            break

    print(f"\n{'='*60}")
    print(f"执行完成: {success_count}/{len(steps)} 步成功")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
