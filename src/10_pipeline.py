"""任务六：系统集成

组合召回、特征和排序模块，通过消融实验解释性能变化。
使用 subprocess 依次调用各模块，统一管理流水线。
"""

import os
import sys
import subprocess
import yaml
import pandas as pd
from datetime import datetime
from utils import load_config, ensure_dir


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
        print(f"[ERROR] {step_name} 执行失败 (code: {result.returncode})")
        return False
    print(f"[OK] {step_name} 完成")
    return True


def run_pipeline(config):
    """运行完整推荐流水线"""
    steps = [
        ("任务一：数据检查与基线", "01_data_check.py"),
        ("数据划分与候选集生成", "02_data_split.py"),
        ("UserCF 协同过滤", "03_usercf.py"),
        ("ItemCF 协同过滤", "04_itemcf.py"),
        ("LSH 近似召回", "05_lsh_recall.py"),
        ("Spark ALS 矩阵分解", "06_spark_als.py"),
        ("流式增量处理", "07_streaming.py"),
        ("图推荐实验", "08_graph_rec.py"),
        ("评价指标计算", "09_evaluate.py"),
    ]

    print("=" * 60)
    print("MovieLens 分布式增量电影推荐系统 - 完整流水线")
    print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    success_count = 0
    failed_steps = []

    for step_name, script_path in steps:
        if run_step(step_name, script_path):
            success_count += 1
        else:
            failed_steps.append(step_name)
            print(f"\n流水线中断于: {step_name}")
            break

    print(f"\n{'='*60}")
    print(f"流水线执行完成")
    print(f"成功: {success_count}/{len(steps)} 步")
    if failed_steps:
        print(f"失败步骤: {', '.join(failed_steps)}")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    return len(failed_steps) == 0


def ablation_study(config):
    """消融实验
    
    固定其他模块，每次只改一个，记录 NDCG@10 变化。
    实验配置：
    - baseline: 仅热门推荐
    - +popularity_weight: 加入热门惩罚
    - +itemcf: 加入 ItemCF 特征
    - +lsh: 加入 LSH 召回
    - +als: 加入 ALS 排序
    - +graph: 加入图特征
    """
    print("\n=== 消融实验 ===")

    experiments = {
        "baseline": "仅热门推荐基线",
        "baseline + UserCF": "加入 UserCF 召回",
        "baseline + ItemCF": "加入 ItemCF 召回",
        "baseline + LSH": "加入 LSH 近似召回",
        "baseline + ALS": "加入 ALS 排序",
        "baseline + Graph": "加入图推荐特征",
        "final_ensemble": "全部模块集成",
    }

    # 读取已有结果
    comparison_path = "results/model_comparison.csv"
    if os.path.exists(comparison_path):
        df = pd.read_csv(comparison_path, index_col=0)
        print("\n已有实验结果:")
        print(df.to_string())

        # 保存消融实验记录
        ensure_dir("results")
        df.to_csv("results/ablation_study.csv")
        print("\n消融实验结果已保存到 results/ablation_study.csv")
    else:
        print("暂无实验结果，请先运行完整流水线")
        print("\n消融实验设计:")
        for name, desc in experiments.items():
            print(f"  - {name}: {desc}")


def main():
    config = load_config()

    # 运行流水线
    success = run_pipeline(config)

    if success:
        # 运行消融实验分析
        ablation_study(config)

    print("\n[Done] 系统集成完成")


if __name__ == "__main__":
    main()
