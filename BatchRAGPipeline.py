import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from camel.agents import ChatAgent

from agent_role import build_analyst_agent, build_knowledge_agent
from tools import build_query_bundle, pdf_question_to_text


def timed_call(timings, name, func, *args, **kwargs):
    start = time.perf_counter()
    try:
        return func(*args, **kwargs)
    finally:
        timings[name] = round(time.perf_counter() - start, 4)


class InputNormalizer:
    def type_checking(self, raw_input):
        # 判断question的type
        # 1. string 类的question / 2. pdf 文件 
        
        question_path = os.path.abspath(raw_input)
        if question_path.lower().endswith(".pdf"):      # 末尾后缀检测
            return "pdf"
        else:
            return "string"

    def build_string_task(self, raw_input, index):
        # 普通字符串清洗后作为问题本身
        question_text = raw_input.strip()
        search_queries = [question_text] if question_text else []

        return {
            "index": index,
            "raw_input": raw_input,
            "input_type": "string",
            "query_bundle": {
                "raw_question": question_text,
                "clean_question": question_text,
                "query_type": "string_question",
                "search_queries": search_queries,
            },
            "error": None,
        }

    def build_pdf_task(self, raw_input, index):
        # PDF文本先提取文字
        # 再构建 query bundle
        pdf_text = pdf_question_to_text(raw_input)
        query_bundle = build_query_bundle(pdf_text)

        return {
            "index": index,
            "raw_input": raw_input,
            "input_type": "pdf",
            "query_bundle": query_bundle,
            "error": None,
        }

    def normalize_one(self, raw_input, index):
        # 将用户输入问题统一成 结构化数据
        try:
            input_type = self.type_checking(raw_input)

            if input_type == "pdf":
                task = self.build_pdf_task(raw_input, index)
            else:
                task = self.build_string_task(raw_input, index)

            item = task["query_bundle"]["raw_question"]
            if not item:
                raise ValueError("empty question")
            return task

        except Exception as e:
            return {
                "index": index,
                "raw_input": raw_input,
                "input_type": self.type_checking(raw_input),
                "question_text": "",
                "search_queries": [],
                "query_bundle": None,
                "error": str(e),
            }

    def normalize_batch(self, raw_inputs):
        # 批量处理时保留 index，方便最后按顺序汇总
        tasks = []
        for index, raw_input in enumerate(raw_inputs):
            tasks.append(self.normalize_one(raw_input, index))
        return tasks


class RetrievalWorkerA:
    # 改写问题后，走向量召回
    def __init__(self, model, retriever, top_k: int = 3):
        self.model = model
        self.retriever = retriever
        self.top_k = top_k

    def rewrite_query(self, question):
        # 原问题改写
        rewriter = ChatAgent(
            model=self.model,
            output_language="chinese",
        )

        prompt = f"""
            请将下面的问题改写为更适合向量检索的查询语句。
            要求：
            1. 保留原问题的关键约束与核心名词。
            2. 不要扩写出原文没有的信息。
            3. 只输出改写后的查询，不要解释。
            原问题：
            {question}
            """

        try:
            rewritten_question = rewriter.step(prompt).msgs[0].content.strip()
            return rewritten_question if rewritten_question else question
        except Exception:
            return question

    def run(self, normalized_task):
        query_bundle = normalized_task.get("query_bundle") or {}
        question_text = query_bundle.get("clean_question", "")
        timings = {}

        if not question_text:
            return {
                "index": normalized_task.get("index"),
                "worker_name": "vector",
                "question_text": "",
                "rewritten_query": "",
                "used_queries": [],
                "results": [],
                "timings": timings,
                "error": "Empty question text",
            }

        rewritten_query = timed_call(
            timings,
            "rewrite_query",
            self.rewrite_query,
            question_text,
        )

        try:
            retrieval_results = timed_call(
                timings,
                "vector_search",
                self.retriever.search,
                rewritten_query,
                top_k=self.top_k,
            )

            return {
                "index": normalized_task.get("index"),
                "worker_name": "vector",
                "question_text": question_text,
                "rewritten_query": rewritten_query,
                "used_queries": [rewritten_query],
                "results": retrieval_results,
                "timings": timings,
                "error": None,
            }

        except Exception as e:
            return {
                "index": normalized_task.get("index"),
                "worker_name": "vector",
                "question_text": question_text,
                "rewritten_query": rewritten_query,
                "used_queries": [rewritten_query],
                "results": [],
                "timings": timings,
                "error": str(e),
            }


class MergeWorker:
    # 召回结果，给后面的 AnswerWorker 使用
    def build_union_content(self, results):
        parts = []

        for idx, item in enumerate(results, start=1):
            location_text = ""
            if item.get("page_idx") not in (None, ""):
                location_text = f"页码: {item['page_idx']}"

            part = (
                f"[检索结果 {idx}]\n"
                f"文件名: {item.get('file_name', '')}\n"
                f"来源类型: {item.get('source_type', 'text')}\n"
                f"{location_text}\n"
                f"模态: {item.get('modality', 'text')}\n"
                f"内容: {item.get('content', '')}"
            )
            parts.append(part)

        return "\n\n".join(parts)

    def merge_results(self, retrieval_outputs):
        # 合并多路结果，并按 文件名 + 页码 + 内容 去重
        merged_results = []
        seen = set()

        for output in retrieval_outputs:
            current_results = output.get("results", [])

            for item in current_results:
                unique_key = (
                    item.get("file_name"),
                    item.get("page_idx"),
                    item.get("content"),
                )

                if unique_key in seen:
                    continue

                seen.add(unique_key)
                merged_results.append(item)

        return merged_results

    def run(self, normalized_task, retrieval_outputs):
        # 统一运行入口
        query_bundle = normalized_task.get("query_bundle") or {}
        question_text = query_bundle.get("clean_question", "")

        retrievals = {}
        errors = []

        for output in retrieval_outputs:
            worker_name = output.get("worker_name", "unknown")
            retrievals[worker_name] = output

            if output.get("error"):
                errors.append(
                    {
                        "worker_name": worker_name,
                        "error": output.get("error"),
                    }
                )

        merged_results = self.merge_results(retrieval_outputs)
        union_content = self.build_union_content(merged_results)

        return {
            "index": normalized_task.get("index"),
            "raw_input": normalized_task.get("raw_input"),
            "input_type": normalized_task.get("input_type"),
            "question_text": question_text,
            "query_bundle": query_bundle,
            "retrievals": retrievals,
            "merged_results": merged_results,
            "union_content": union_content,
            "error": errors,
        }


class AnswerWorker:
    # 根据 merge 后的上下文生成最终答案
    def __init__(self, model):
        self.model = model

    def build_knowledge_prompt(self, question_text, union_content):
        # knowledg_agent 提炼找回内容的关键信息
        return f"""
            你是知识分析助手，请基于检索到的资料回答问题。
            [问题]
            {question_text}

            [资料]
            {union_content}

            要求：
            1. 只根据资料作答，不能编造。
            2. 若资料不足，请明确说明不足之处。
            3. 先给出结论，再给出依据。
            """

    def build_analyst_prompt(self, question_text, union_content, knowledge_answer):
        # analyst_agent 基于整理后的知识输出最终答案
        return f"""
            你是分析助手，请基于已有知识回答给出最终答案。
            [问题]
            {question_text}

            [资料]
            {union_content}

            [知识回答]
            {knowledge_answer}

            要求：
            1. 直接回答问题。
            2. 保持条理清晰、逻辑严谨。
            3. 若资料不足，请明确说明。
            """

    def run(self, merged_task):
        # 获取到最终要回答的问题
        question_text = merged_task.get("question_text", "")
        union_content = merged_task.get("union_content", "")
        timings = {}

        if not question_text:
            return {
                "index": merged_task.get("index"),
                "question_text": "",
                "knowledge_answer": "",
                "final_answer": "",
                "retrievals": merged_task.get("retrievals", {}),
                "merged_results": merged_task.get("merged_results", []),
                "union_content": union_content,
                "timings": timings,
                "error": "Empty question text",
            }

        if not union_content:
            return {
                "index": merged_task.get("index"),
                "question_text": question_text,
                "knowledge_answer": "",
                "final_answer": "",
                "retrievals": merged_task.get("retrievals", {}),
                "merged_results": merged_task.get("merged_results", []),
                "union_content": "",
                "timings": timings,
                "error": "Empty union content",
            }

        try:
            # 初始化 协调智能体
            knowledge_agent = build_knowledge_agent(self.model)
            analyst_agent = build_analyst_agent(self.model)

            # 理解process
            knowledge_prompt = self.build_knowledge_prompt(
                question_text=question_text,
                union_content=union_content,
            )
            knowledge_response = timed_call(
                timings,
                "knowledge_agent_step",
                knowledge_agent.step,
                knowledge_prompt,
            )
            knowledge_answer = knowledge_response.msgs[0].content

            # 分析process
            analyst_prompt = self.build_analyst_prompt(
                question_text=question_text,
                union_content=union_content,
                knowledge_answer=knowledge_answer,
            )
            analyst_response = timed_call(
                timings,
                "analyst_agent_step",
                analyst_agent.step,
                analyst_prompt,
            )
            final_answer = analyst_response.msgs[0].content

            return {
                "index": merged_task.get("index"),
                "question_text": question_text,
                "knowledge_answer": knowledge_answer,
                "final_answer": final_answer,
                "retrievals": merged_task.get("retrievals", {}),
                "merged_results": merged_task.get("merged_results", []),
                "union_content": union_content,
                "timings": timings,
                "error": None,
            }

        except Exception as e:
            return {
                "index": merged_task.get("index"),
                "question_text": question_text,
                "knowledge_answer": "",
                "final_answer": "",
                "retrievals": merged_task.get("retrievals", {}),
                "merged_results": merged_task.get("merged_results", []),
                "union_content": union_content,
                "timings": timings,
                "error": str(e),
            }


class BatchRAGPipeline:
    def __init__(self, model, retriever_a, retrieval_workers=None, top_k: int = 3):
        # pipeline 初始化
        self.model = model
        self.normalizer = InputNormalizer()
        self.merge_worker = MergeWorker()
        self.answer_worker = AnswerWorker(model)

        # 检索agent 
        if retrieval_workers is None:
            self.retrieval_workers = [
                RetrievalWorkerA(model=model, retriever=retriever_a, top_k=top_k)
            ]
        else:
            self.retrieval_workers = retrieval_workers

    def retrieve_one(self, normalized_task):
        # 执行召回
        retrieval_outputs = []
        timings = {}

        for worker in self.retrieval_workers:
            worker_name = getattr(worker, "__class__", type(worker)).__name__
            result = timed_call(
                timings,
                f"worker.{worker_name}",
                worker.run,
                normalized_task,
            )
            retrieval_outputs.append(result)

        return retrieval_outputs, timings

    def run_one(self, raw_input, index: int = 0):
        # 单个输入的完整 pipeline
        # InputNor =>  retrievalA => merge => answer
        timings = {}
        normalized_task = timed_call(
            timings,
            "normalize_one",
            self.normalizer.normalize_one,
            raw_input,
            index,
        )

        if normalized_task.get("error"):
            return {
                "index": index,
                "raw_input": raw_input,
                "question_text": "",
                "knowledge_answer": "",
                "final_answer": "",
                "timings": timings,
                "error": normalized_task.get("error"),
            }

        retrieval_outputs, retrieve_timings = timed_call(
            timings,
            "retrieve_one",
            self.retrieve_one,
            normalized_task,
        )
        timings["retrieve_workers"] = retrieve_timings

        merged_task = timed_call(
            timings,
            "merge",
            self.merge_worker.run,
            normalized_task,
            retrieval_outputs,
        )
        answer_result = timed_call(
            timings,
            "answer",
            self.answer_worker.run,
            merged_task,
        )
        timings["answer_detail"] = answer_result.get("timings", {})

        return {
            "index": index,
            "raw_input": raw_input,
            "input_type": normalized_task.get("input_type"),
            "query_bundle": normalized_task.get("query_bundle"),
            "retrieval_outputs": retrieval_outputs,
            "merged_task": merged_task,
            "answer_result": answer_result,
            "final_answer": answer_result.get("final_answer", ""),
            "timings": timings,
            "error": answer_result.get("error"),
        }

    def run_batch(self, raw_inputs):
        # 批量输入入口：
        # 1. 先统一标准化
        # 2. 并发执行召回
        # 3. 按输入顺序依次 merge 和 answer

        normalized_tasks = []
        normalize_timings = {}
        for index, raw_input in enumerate(raw_inputs):
            task_timings = {}
            task = timed_call(
                task_timings,
                "normalize_one",
                self.normalizer.normalize_one,
                raw_input,
                index,
            )
            normalized_tasks.append(task)
            normalize_timings[index] = task_timings

        ordered_retrieval_outputs = [None] * len(normalized_tasks)
        results = [None] * len(normalized_tasks)

        valid_tasks = []
        for task in normalized_tasks:
            task_index = task.get("index", 0)
            if task.get("error"):
                results[task_index] = {
                    "index": task_index,
                    "raw_input": task.get("raw_input"),
                    "input_type": task.get("input_type"),
                    "query_bundle": task.get("query_bundle"),
                    "retrieval_outputs": [],
                    "merged_task": None,
                    "answer_result": None,
                    "final_answer": "",
                    "timings": normalize_timings.get(task_index, {}).copy(),
                    "error": task.get("error"),
                }
            else:
                valid_tasks.append(task)

        if valid_tasks:
            # 最多4个task并行，设定上限
            max_workers = min(len(valid_tasks), 4)

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_index = {
                    executor.submit(self.retrieve_one, task): task["index"]
                    for task in valid_tasks
                }

                for future in as_completed(future_to_index):
                    task_index = future_to_index[future]
                    try:
                        ordered_retrieval_outputs[task_index] = future.result()
                    except Exception as e:
                        ordered_retrieval_outputs[task_index] = (
                            [
                                {
                                    "index": task_index,
                                    "worker_name": "unknown",
                                    "question_text": "",
                                    "rewritten_query": "",
                                    "used_queries": [],
                                    "results": [],
                                    "timings": {},
                                    "error": str(e),
                                }
                            ],
                            {},
                        )

        for task in normalized_tasks:
            task_index = task.get("index", 0)

            if results[task_index] is not None:
                continue

            timings = normalize_timings.get(task_index, {}).copy()
            retrieval_outputs, retrieve_timings = (
                ordered_retrieval_outputs[task_index] or ([], {})
            )
            timings["retrieve_workers"] = retrieve_timings

            merged_task = timed_call(
                timings,
                "merge",
                self.merge_worker.run,
                task,
                retrieval_outputs,
            )
            answer_result = timed_call(
                timings,
                "answer",
                self.answer_worker.run,
                merged_task,
            )
            timings["answer_detail"] = answer_result.get("timings", {})

            results[task_index] = {
                "index": task_index,
                "raw_input": task.get("raw_input"),
                "input_type": task.get("input_type"),
                "query_bundle": task.get("query_bundle"),
                "retrieval_outputs": retrieval_outputs,
                "merged_task": merged_task,
                "answer_result": answer_result,
                "final_answer": answer_result.get("final_answer", ""),
                "timings": timings,
                "error": answer_result.get("error"),
            }

        return results
