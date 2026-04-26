# core/config.py
import os
from dotenv import load_dotenv

# 加载根目录的 api_key.env
load_dotenv("api_key.env", override=True)

# 定义所有绝对路径或相对路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 1. 数据库路径（Chroma 文件夹）
DB_PATH = os.path.join(BASE_DIR, "db")

# 2. 知识库原文路径（data 文件夹）
DATA_DIR = os.path.join(BASE_DIR, "data")

# 3. 输出报告路径（outputs 文件夹）
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# 4. 测试集输入路径（testdata 文件夹）
TESTDATA_DIR = os.path.join(BASE_DIR, "testdata")

# 5. 模型路径
MODEL_PATH = r"/root/DATA_COMPLIANCE_RAG/sentence-transformer"
