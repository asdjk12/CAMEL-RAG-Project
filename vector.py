# 该文件用于存储向量数据

import json
import os
from dotenv import load_dotenv
from example_codes import qdrant, vector_retriever
from camel.models import ModelFactory
from camel.types import ModelPlatformType
from camel.agents import ChatAgent
from agent_role import build_knowledge_agent, build_analyst_agent

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

def build_union_content(results):
    parts = []
    for idx, item in enumerate(results, start=1):
        parts.append(
            f"""[召回片段 {idx}]
                来源文件: {item['file_name']}
                页码: {item['page_idx']}
                内容:
                {item['content']}"""
            )
    return "\n\n".join(parts)

def rewrite_query(question, model):
    """
    改写向量检查问题
    """

    rewriter = ChatAgent(
        model=model,
        output_language="中文",
    )

    prompt = f"""
        你是RAG检索查询改写助手。
        你的任务是把用户问题改写成更适合向量检索的查询语句。

        要求：
        1. 保留原问题核心含义，不要改变问题意图。
        2. 尽量补充原问题里的关键概念同义表达。
        3. 输出只保留一条适合检索的中文问题，不要解释，不要分点，不要加引号。

        用户原问题：
        {question}
    """

    try:
        query_question = rewriter.step(prompt).msgs[0].content.strip()
        return query_question if query_question else question
    except Exception:
        return question


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

    # agent 初始化
    Knowledge_agent = build_knowledge_agent(model)
    Analyst_agent = build_analyst_agent(model)


    # 存储process
    db = qdrant.QdrantDB()
    data_storage(data)
    print("文本储存完成！")


    # 召回process
    retriever = vector_retriever.VecRetriever(db)
    
    try:
        question = f"交换价值是什么？"

        # query改写
        query_question = rewrite_query(question=question, model=model)
        print(f"原始问题: {question}")
        print(f"检索问题: {query_question}")
        print("-" * 50)

        # 检索
        result = retriever.search(query_question, top_k=3)
        
        # 查看结果
        for item in result:
            print(f"文件：{item['file_name']}\n")       # JSON_PATH, editable
            print(f"页码：{item['page_idx']}")
            print(f"应召回内容:{item['content']}\n")
        
        # 检索聚合
        union_content = build_union_content(result) 
            
        # 5. 知识解读师
        knowledge_prompt = f"""
            请基于下面的检索内容，回答用户问题，并完成知识解读。

            [用户问题]
            {question}

            [检索内容]
            {union_content}

            要求：
            1. 只能依据检索内容回答。
            2. 先给出核心概念解释。
            3. 如果检索内容涉及相近概念，请指出区别与联系。
            4. 如果材料不足，请明确说“材料中未直接说明”。
        """ 

        knowledge_response = Knowledge_agent.step(knowledge_prompt)
        knowledge_answer = knowledge_response.msgs[0].content

        print("KnowledgeReader 输出:")
        print(knowledge_answer)
        print("=" * 50)

        # 6. 内容分析师
        analyst_prompt = f"""
            请基于下面的信息，生成最终分析回答。

            [用户问题]
            {question}

            [检索内容]
            {union_content}

            [知识解读师输出]
            {knowledge_answer}

            你的任务：
            1. 总结这次问题对应文本的主题。
            2. 提炼主要观点。
            3. 说明文本中的论证结构或信息组织方式。
            4. 最后给出一个面向用户的最终回答。

            输出格式建议：
            - 主题概括
            - 主要观点
            - 论证结构
            - 最终回答
        """

        analyst_response = Analyst_agent.step(analyst_prompt)
        final_answer = analyst_response.msgs[0].content

        print("ContentAnalyst 输出 / 最终答案:")
        print(final_answer)
        print("=" * 50)

    finally:
        retriever.close()

