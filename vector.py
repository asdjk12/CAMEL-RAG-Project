# 该文件用于存储向量数据

import json
import os
from dotenv import load_dotenv
from example_codes import qdrant, vector_retriever
from camel.models import ModelFactory
from camel.types import ModelPlatformType
from camel.agents import ChatAgent

def load_JSON(path):
    """
    path 指向一个结构化的json文件。 当前以SMALL_OCR_CONTENT_LIST.json 结构为准
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)     # 读取json文件内容
    return data

def data_storage(data):
    db.text2vector(data,json_path)
    return 


if __name__ == "__main__":
    load_dotenv()
    json_path = os.getenv("SMALL_OCR_CONTENT_LIST")     # 读取环境变量

    data = load_JSON(json_path)             # 读取json 文件
    model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
        model_type="ZhipuAI/GLM-5",
        url='https://api-inference.modelscope.cn/v1/',
        api_key= os.getenv('MODELSCOPE_SDK_TOKEN'),
    )

    # 存储process
    db = qdrant.QdrantDB()
    data_storage(data)
    print("文本储存完成！")


    # 召回process
    retriever = vector_retriever.VecRetriever(db)
    try:
        result = retriever.search("交换价值是什么？", top_k=1)
            # 4. 查看结果
        for item in result:
            print(f"文件：{item['file_name']}\n")       # JSON_PATH, editable
            print(f"页码：{item['page_idx']}")
            print(f"应召回内容:{item['content']}\n")
            
            # 改写召回内容
            prompt = f"""
                提取上传内容的关键信息,并改写成更精炼的回答
                [上传内容]:\n
                {item['content']}
                """

            agent = ChatAgent(
                model= model,
                output_language="中文",
            )
            print(f"回答:{agent.step(prompt).msgs[0].content}" )

            print("-" * 50)

    finally:
        retriever.close()
