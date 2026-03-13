from camel.agents import ChatAgent
from camel.messages import BaseMessage

from task_router import TaskRouter
from verifier_agent import VerificationResult
from verifier_agent import AnswerVerifier


class StrongAgentPipeline:
    # 强 agent 总控层：
    # 1. 先路由
    # 2. 再执行
    # 3. 最后校验
    # 4. 必要时自动重试
    def __init__(self, model, rag_pipeline):
        self.model = model
        self.rag_pipeline = rag_pipeline
        self.router = TaskRouter(model=model)
        self.verifier = AnswerVerifier(model=model)
        self.direct_agent = self._build_direct_agent()

    def _build_direct_agent(self):
        return ChatAgent(
            system_message=BaseMessage.make_assistant_message(
                role_name="直接回答助手",
                content=(
                    "你负责处理无需检索的简单问题。"
                    "如果信息不足，不要编造，要明确说明边界。"
                ),
            ),
            model=self.model,
            output_language="chinese",
        )

    def close(self):
        self.rag_pipeline.close()

    def _ask_directly(self, raw_input, history_summary=""):
        prompt = f"""
请直接回答下面的问题。

[history_summary]
{history_summary or "当前没有历史上下文。"}

[question]
{raw_input}
"""
        response = self.direct_agent.step(prompt)
        return response.msgs[0].content.strip()

    def _build_clarify_answer(self, raw_input):
        return (
            "你的问题目前还不够完整。"
            f"请补充你具体想问的对象、范围或目标：{raw_input}"
        )

    def _build_retry_input(self, raw_input, previous_result, retry_round):
        # 重试时显式补充失败信息，让下一轮检索更聚焦
        retry_hint = previous_result.get("verification", {}).get("reason", "")
        rewritten_query = previous_result.get("rewritten_query", "")
        return (
            f"{raw_input}\n"
            f"补充要求：这是第 {retry_round} 次重试。"
            "请扩大召回范围并更关注关键约束、缺失信息和直接证据。"
            f"\n上一轮检索改写: {rewritten_query}"
            f"\n上一轮失败原因: {retry_hint}"
        )

    def _verify_result(self, raw_input, result, history_summary, route):
        verification = self.verifier.verify(
            question=raw_input,
            answer=result.get("final_answer", ""),
            retrieval_result=result.get("retrieval_result", ""),
            history_summary=history_summary,
            route=route,
        )
        result["verification"] = verification.to_dict()
        return verification

    def _accept_without_verify(self, reason):
        return VerificationResult(
            status="pass",
            score=100,
            reason=reason,
            suggested_action="accept",
        )

    def run_one(self, raw_input, index=0, session_memory=None):
        history_summary = ""
        if session_memory is not None:
            history_summary = session_memory.build_context_summary()

        route_decision = self.router.route(
            raw_input=raw_input,
            history_summary=history_summary,
        )

        # 先处理无需检索的路由分支
        if route_decision.route == "clarify":
            final_answer = self._build_clarify_answer(raw_input)
            result = {
                "index": index,
                "raw_input": raw_input,
                "input_type": "string",
                "state": "DONE",
                "route": route_decision.route,
                "route_reason": route_decision.reason,
                "rewritten_query": "",
                "retrieval_result": "",
                "knowledge_answer": "",
                "final_answer": final_answer,
                "error": None,
            }
            verification = self._accept_without_verify("当前路由要求先澄清用户需求。")
            result["verification"] = verification.to_dict()
            result["attempt_count"] = 1
            result["verification_status"] = verification.status
            return result

        if route_decision.route == "direct_answer":
            final_answer = self._ask_directly(raw_input, history_summary=history_summary)
            result = {
                "index": index,
                "raw_input": raw_input,
                "input_type": "string",
                "state": "DONE",
                "route": route_decision.route,
                "route_reason": route_decision.reason,
                "rewritten_query": "",
                "retrieval_result": "",
                "knowledge_answer": "",
                "final_answer": final_answer,
                "error": None,
            }
            if route_decision.should_verify:
                verification = self._verify_result(
                    raw_input=raw_input,
                    result=result,
                    history_summary=history_summary,
                    route=route_decision.route,
                )
            else:
                verification = self._accept_without_verify("当前路由不要求额外校验。")
                result["verification"] = verification.to_dict()
            result["attempt_count"] = 1
            result["verification_status"] = verification.status
            return result

        # 检索型路由统一先走底层 RAG 执行器
        # 注意：multi_step 目前只是被 router 标记出来，
        # 真正的多子任务规划/并行执行仍可在这里继续扩展。
        result = self.rag_pipeline.run_one(raw_input=raw_input, index=index)
        result["route"] = route_decision.route
        result["route_reason"] = route_decision.reason

        if route_decision.should_verify:
            verification = self._verify_result(
                raw_input=raw_input,
                result=result,
                history_summary=history_summary,
                route=route_decision.route,
            )
        else:
            verification = self._accept_without_verify("当前路由不要求额外校验。")
            result["verification"] = verification.to_dict()

        attempt_count = 1
        while (
            route_decision.should_retrieve
            and route_decision.should_verify
            and verification.suggested_action == "retry"
            and attempt_count <= route_decision.max_retry
        ):
            retry_input = self._build_retry_input(
                raw_input=raw_input,
                previous_result=result,
                retry_round=attempt_count,
            )
            result = self.rag_pipeline.run_one(raw_input=retry_input, index=index)
            result["route"] = route_decision.route
            result["route_reason"] = (
                f"{route_decision.reason} 当前为自动重试第 {attempt_count} 轮。"
            )
            verification = self._verify_result(
                raw_input=raw_input,
                result=result,
                history_summary=history_summary,
                route=route_decision.route,
            )
            attempt_count += 1

        result["attempt_count"] = attempt_count
        result["verification_status"] = verification.status
        return result

    def run_batch(self, raw_inputs, session_memory=None):
        if not raw_inputs:
            return []

        results = []
        for index, raw_input in enumerate(raw_inputs):
            results.append(
                self.run_one(
                    raw_input=raw_input,
                    index=index,
                    session_memory=session_memory,
                )
            )
        return results
