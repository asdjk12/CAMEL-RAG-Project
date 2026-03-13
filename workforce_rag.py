import json
import os
import re

from agent_role import build_analyst_agent, build_knowledge_agent
from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.societies.workforce import Workforce, WorkforceMode
from camel.tasks import Task
from camel.toolkits import FunctionTool
from tools import build_query_bundle, pdf_question_to_text


class WorkforceRAGPipeline:
    def __init__(self, model, retriever, top_k=3):
        self.model = model
        self.retriever = retriever
        self.top_k = top_k

    def close(self):
        self.retriever.close()

    def _type_checking(self, raw_input):
        question_path = os.path.abspath(raw_input)
        if question_path.lower().endswith(".pdf"):
            return "pdf"
        return "string"

    def _normalize_one(self, raw_input, index):
        try:
            input_type = self._type_checking(raw_input)
            if input_type == "pdf":
                pdf_text = pdf_question_to_text(raw_input)
                query_bundle = build_query_bundle(pdf_text)
            else:
                question_text = raw_input.strip()
                query_bundle = {
                    "raw_question": question_text,
                    "clean_question": question_text,
                    "query_type": "string_question",
                    "search_queries": [question_text] if question_text else [],
                }

            if not query_bundle.get("raw_question"):
                raise ValueError("empty question")

            return {
                "index": index,
                "raw_input": raw_input,
                "input_type": input_type,
                "query_bundle": query_bundle,
                "error": None,
            }
        except Exception as exc:
            return {
                "index": index,
                "raw_input": raw_input,
                "input_type": self._type_checking(raw_input),
                "query_bundle": None,
                "error": str(exc),
            }

    def _vector_search(self, query, top_k=3):
        top_k = max(1, min(int(top_k), 10))
        results = self.retriever.search(query, top_k=top_k)
        simplified = []
        for item in results:
            simplified.append(
                {
                    "file_name": item.get("file_name", ""),
                    "page_idx": item.get("page_idx"),
                    "source_type": item.get("source_type", "text"),
                    "modality": item.get("modality", "text"),
                    "content": item.get("content", ""),
                }
            )
        return json.dumps(
            {
                "query": query,
                "top_k": top_k,
                "results": simplified,
            },
            ensure_ascii=False,
            indent=2,
        )

    def _make_query_rewriter(self):
        return ChatAgent(
            system_message=BaseMessage.make_assistant_message(
                role_name="检索问题改写器",
                content=(
                    "你负责把用户问题改写成适合向量检索的独立查询。"
                    "必须保留原问题约束，不得编造，不要输出解释。"
                ),
            ),
            model=self.model,
            output_language="chinese",
        )

    def _make_retrieval_agent(self):
        search_tool = FunctionTool(self._vector_search)
        return ChatAgent(
            system_message=BaseMessage.make_assistant_message(
                role_name="向量检索员",
                content=(
                    "你负责调用向量检索工具完成知识库检索。"
                    "你必须优先调用工具，再基于工具返回结果整理输出。"
                    "如果没有检索到足够证据，要明确说明。"
                ),
            ),
            model=self.model,
            output_language="chinese",
            tools=[search_tool],
        )

    def _build_workforce(self):
        workforce = Workforce(
            description="RAG Workforce",
            coordinator_agent=ChatAgent(model=self.model, output_language="chinese"),
            task_agent=ChatAgent(model=self.model, output_language="chinese"),
            new_worker_agent=ChatAgent(model=self.model, output_language="chinese"),
            mode=WorkforceMode.PIPELINE,
            task_timeout_seconds=900,
        )

        workforce.add_single_agent_worker(
            "负责将用户问题改写为适合检索的独立查询",
            worker=self._make_query_rewriter(),
        ).add_single_agent_worker(
            "负责调用向量数据库检索相关证据",
            worker=self._make_retrieval_agent(),
        ).add_single_agent_worker(
            "负责基于检索结果提炼关键事实、证据与不足点",
            worker=build_knowledge_agent(self.model),
        ).add_single_agent_worker(
            "负责根据问题和证据生成最终答案",
            worker=build_analyst_agent(self.model),
        )

        return workforce

    def _extract_task_result(self, aggregated_result, task_id):
        if not aggregated_result:
            return ""

        pattern = (
            rf"--- Task {re.escape(task_id)} Result ---\n"
            rf"(.*?)(?=\n--- Task [^\n]+ Result ---|\Z)"
        )
        match = re.search(pattern, aggregated_result, re.S)
        if not match:
            return ""
        return match.group(1).strip()

    def run_one(self, raw_input, index=0):
        normalized_task = self._normalize_one(raw_input, index)
        if normalized_task.get("error"):
            return {
                "index": index,
                "raw_input": raw_input,
                "input_type": normalized_task.get("input_type"),
                "query_bundle": normalized_task.get("query_bundle"),
                "state": "FAILED",
                "rewritten_query": "",
                "retrieval_result": "",
                "knowledge_answer": "",
                "final_answer": "",
                "workforce_result": "",
                "error": normalized_task.get("error"),
            }

        question_text = normalized_task["query_bundle"]["clean_question"]
        workforce = self._build_workforce()

        workforce.pipeline_add(
            Task(
                id="rewrite_query",
                content=(
                    "请把下面的问题改写成适合向量检索的独立查询，只输出改写结果：\n"
                    f"{question_text}"
                ),
                additional_info={
                    "raw_question": question_text,
                },
            )
        ).pipeline_add(
            Task(
                id="vector_retrieve",
                content=(
                    "请读取依赖任务中的改写查询，并调用向量检索工具检索知识库。"
                    "返回最相关的证据，至少包含文件名、页码、内容、相关性说明。"
                ),
                additional_info={
                    "top_k": self.top_k,
                },
            )
        ).pipeline_add(
            Task(
                id="evidence_summarize",
                content=(
                    "请基于依赖任务中的检索证据，提炼回答当前问题所需的关键事实、"
                    "证据链和信息不足点。\n"
                    f"当前问题：{question_text}"
                ),
            )
        ).pipeline_add(
            Task(
                id="final_answer",
                content=(
                    "请综合依赖任务中的事实和证据，直接回答用户问题。"
                    "要求先给结论，再给依据；资料不足时必须明确说明。\n"
                    f"用户问题：{question_text}"
                ),
            )
        ).pipeline_build()

        main_task = Task(
            content=f"使用固定 RAG Workforce 流水线回答问题：{question_text}",
            id=f"rag-{index}",
            additional_info={
                "raw_input": raw_input,
                "input_type": normalized_task.get("input_type"),
            },
        )

        result_task = workforce.process_task(main_task)
        workforce_result = result_task.result or ""
        rewritten_query = self._extract_task_result(workforce_result, "rewrite_query")
        retrieval_result = self._extract_task_result(workforce_result, "vector_retrieve")
        knowledge_answer = self._extract_task_result(workforce_result, "evidence_summarize")
        final_answer = self._extract_task_result(workforce_result, "final_answer")

        error = None
        if result_task.state.value != "DONE" and not final_answer:
            error = f"workforce pipeline state={result_task.state.value}"

        return {
            "index": index,
            "raw_input": raw_input,
            "input_type": normalized_task.get("input_type"),
            "query_bundle": normalized_task.get("query_bundle"),
            "state": result_task.state.value,
            "rewritten_query": rewritten_query,
            "retrieval_result": retrieval_result,
            "knowledge_answer": knowledge_answer,
            "final_answer": final_answer,
            "workforce_result": workforce_result,
            "error": error,
        }

    def run_batch(self, raw_inputs):
        if not raw_inputs:
            return []

        results = []
        for index, raw_input in enumerate(raw_inputs):
            results.append(self.run_one(raw_input, index=index))
        return results
