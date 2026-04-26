# main.py
import os

# 这一句会自动触发 engine.py 里的模型加载，并把搭好的网页拿过来
from ui.app import demo

if __name__ == "__main__":
    print("🚀 启动智能合规定级系统...")
    # 检查并自动创建输出文件夹
    if not os.path.exists("./outputs"):
        os.makedirs("./outputs")

    # 启动！(正式摆脱 Jupyter)
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=False)
