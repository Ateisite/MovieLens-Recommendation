"""一键运行完整推荐流水线

依次执行所有任务模块，生成推荐结果和评价报告。
"""

import sys
import os

# 将 src 目录加入路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from pipeline import main

if __name__ == "__main__":
    main()
