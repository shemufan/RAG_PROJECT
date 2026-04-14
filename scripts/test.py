from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# 1. 模拟一段假数据（代替队友还没发来的法律）
mock_text = "这是一段用于测试向量入库的模拟法律条文。任何组织和个人不得窃取或者以其他非法方式获取数据。"
doc = [Document(page_content=mock_text)]

# 2. 初始化开源免费的向量化模型
print("正在加载 Embedding 模型...")
embeddings = HuggingFaceEmbeddings() 

# 3. 尝试将其存入本地数据库
print("正在执行向量化并写入 Chroma...")
db = Chroma.from_documents(
    documents=doc, 
    embedding=embeddings, 
    persist_directory="./db" # 存在你刚刚建的 db 文件夹里
)

print("入库成功！快去看看 db 文件夹里是不是多了一些文件？")