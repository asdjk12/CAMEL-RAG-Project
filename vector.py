import json
import os
import sys

from dotenv import load_dotenv

from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.types import ModelPlatformType
from example_codes import qdrant, vector_retriever
from session_memory import SessionMemoryManager
from strong_agent_pipeline import StrongAgentPipeline
from workforce_rag import WorkforceRAGPipeline


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def data_storage(data, db, json_path):
    db.text2vector(data, json_path)
    print("vector index rebuilt")


def create_pipeline(model, db, json_path, rebuild_index=False):
    if rebuild_index:
        data = load_json(json_path)
        data_storage(data, db, json_path)

    retriever = vector_retriever.VecRetriever(db)
    rag_pipeline = WorkforceRAGPipeline(
        model=model,
        retriever=retriever,
        top_k=3,
    )
    pipeline = StrongAgentPipeline(
        model=model,
        rag_pipeline=rag_pipeline,
    )
    return pipeline


class MultiTurnRAGSession:
    def __init__(self, model, pipeline, max_history_turns=5):
        self.model = model
        self.pipeline = pipeline
        self.memory = SessionMemoryManager(max_history_turns=max_history_turns)

    def clear_history(self):
        self.memory.clear()

    def format_history(self):
        return self.memory.format_history()

    def rewrite_follow_up(self, user_input):
        question = user_input.strip()
        if not self.memory.has_history():
            return question

        rewriter = ChatAgent(
            model=self.model,
            output_language="chinese",
        )
        prompt = f"""
        你要把多轮对话中的最新用户问题改写成一个可以直接用于检索的独立问题。

        要求：
        1. 结合历史对话补全代词、省略和上下文。
        2. 不能编造历史中没有的信息。
        3. 如果最新问题已经完整，保留原意即可。
        4. 只输出改写后的问题，不要解释。

    [历史对话]
    {self.memory.build_context_summary()}

    [最新用户问题]
    {question}
        """

        try:
            rewritten = rewriter.step(prompt).msgs[0].content.strip()
            return rewritten if rewritten else question
        except Exception:
            return question

    def ask(self, user_input):
        standalone_query = self.rewrite_follow_up(user_input)
        result = self.pipeline.run_one(
            raw_input=standalone_query,
            index=self.memory.next_turn_id(),
            session_memory=self.memory,
        )
        verification = result.get("verification") or {}
        self.memory.add_turn(
            user_input=user_input,
            standalone_query=standalone_query,
            route=result.get("route", ""),
            route_reason=result.get("route_reason", ""),
            rewritten_query=result.get("rewritten_query", ""),
            final_answer=result.get("final_answer", ""),
            verification_status=verification.get("status", ""),
            verification_reason=verification.get("reason", ""),
        )
        return result, standalone_query


def main(model, db, json_path, raw_inputs=None, rebuild_index=False):
    pipeline = create_pipeline(
        model=model,
        db=db,
        json_path=json_path,
        rebuild_index=rebuild_index,
    )

    try:
        results = pipeline.run_batch(raw_inputs)

        for result in results:
            print("=" * 80)
            print(f"index: {result.get('index')}")
            print(f"raw_input: {result.get('raw_input')}")
            print(f"route: {result.get('route')}")
            print(f"state: {result.get('state')}")
            print(f"error: {result.get('error')}")
            rewritten_query = result.get("rewritten_query")
            if rewritten_query:
                print(f"rewritten_query: {rewritten_query}")
            verification = result.get("verification") or {}
            if verification:
                print(
                    "verification:"
                    f" {verification.get('status')} / {verification.get('suggested_action')}"
                )
            print("final_answer:")
            print(result.get("final_answer", ""))

        return results
    finally:
        pipeline.close()


def interactive_main(model, db, json_path, rebuild_index=False):
    pipeline = create_pipeline(
        model=model,
        db=db,
        json_path=json_path,
        rebuild_index=rebuild_index,
    )
    session = MultiTurnRAGSession(model=model, pipeline=pipeline)

    print("进入多轮 RAG 对话模式。")
    print("命令: /exit 退出, /clear 清空历史, /history 查看历史")

    try:
        while True:
            user_input = input("\n请输入问题: ").strip()

            if not user_input:
                print("输入为空，请重新输入。")
                continue

            command = user_input.lower()
            if command in {"/exit", "exit", "quit"}:
                print("已退出。")
                break

            if command == "/clear":
                session.clear_history()
                print("历史对话已清空。")
                continue

            if command == "/history":
                print(session.format_history())
                continue

            result, standalone_query = session.ask(user_input)

            print("=" * 80)
            if standalone_query != user_input:
                print(f"standalone_query: {standalone_query}")
            print(f"route: {result.get('route')}")
            print(f"state: {result.get('state')}")
            rewritten_query = result.get("rewritten_query")
            if rewritten_query and rewritten_query != standalone_query:
                print(f"rewritten_query: {rewritten_query}")
            verification = result.get("verification") or {}
            if verification:
                print(
                    "verification:"
                    f" {verification.get('status')} / {verification.get('suggested_action')}"
                )
            print("final_answer:")
            print(result.get("final_answer", ""))
    finally:
        pipeline.close()


if __name__ == "__main__":
    # 环境变量
    load_dotenv()

    json_path = os.getenv("SMALL_OCR_CONTENT_LIST")
    if not json_path:
        raise ValueError("SMALL_OCR_CONTENT_LIST is not set")

    # 初始化
    model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
        model_type="ZhipuAI/GLM-5",
        url="https://api-inference.modelscope.cn/v1/",
        api_key=os.getenv("MODELSCOPE_SDK_TOKEN"),
    )

    db = qdrant.QdrantDB()

    # terminal imput控制
    cli_inputs = sys.argv[1:]
    if cli_inputs:
        main(
            # terminal 问题
            model=model,
            db=db,
            json_path=json_path,
            raw_inputs=cli_inputs,
            rebuild_index=False,
        )
    else:
        interactive_main(
            # 交互页面
            model=model,
            db=db,
            json_path=json_path,
            rebuild_index=False,
        )
