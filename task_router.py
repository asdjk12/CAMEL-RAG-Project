import json
import os
import re
from dataclasses import asdict, dataclass

from camel.agents import ChatAgent
from camel.messages import BaseMessage


@dataclass
class RouteDecision:
    # 路由决策结果，决定后续是直接回答、澄清还是走检索
    route: str
    reason: str
    should_retrieve: bool
    should_verify: bool
    max_retry: int = 1

    def to_dict(self):
        return asdict(self)


class TaskRouter:
    def __init__(self, model):
        self.model = model

    def _extract_json(self, text):
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            raise ValueError("router json not found")
        return json.loads(match.group(0))

    def _heuristic_route(self, raw_input, history_summary=""):
        stripped = raw_input.strip()
        input_path = os.path.abspath(stripped)
        has_history = history_summary and "没有历史" not in history_summary

        # PDF 输入优先走文档问答
        if input_path.lower().endswith(".pdf"):
            return RouteDecision(
                route="pdf_rag",
                reason="检测到 PDF 输入，优先走文档检索问答。",
                should_retrieve=True,
                should_verify=True,
                max_retry=2,
            )

        # 过短问题或明显省略表达，先澄清
        if len(stripped) < 4:
            return RouteDecision(
                route="clarify",
                reason="问题过短，缺少足够约束。",
                should_retrieve=False,
                should_verify=False,
                max_retry=0,
            )

        if re.search(r"(这个|那个|它|上面|前面|继续|刚才)", stripped) and not has_history:
            return RouteDecision(
                route="clarify",
                reason="问题依赖上下文，但当前没有足够历史。",
                should_retrieve=False,
                should_verify=False,
                max_retry=0,
            )

        # 包含明显多步骤意图，标记为 multi_step
        if re.search(r"(先.*再|并且|同时|步骤|计划|方案|比较)", stripped):
            return RouteDecision(
                route="multi_step",
                reason="检测到复合任务，后续应进入多步处理。",
                should_retrieve=True,
                should_verify=True,
                max_retry=2,
            )

        # 默认走知识库检索回答
        return RouteDecision(
            route="rag_answer",
            reason="默认走知识库检索与证据回答。",
            should_retrieve=True,
            should_verify=True,
            max_retry=1,
        )

    def _llm_route(self, raw_input, history_summary=""):
        # 用模型做二次判断，让路由不完全依赖规则
        router_agent = ChatAgent(
            system_message=BaseMessage.make_assistant_message(
                role_name="任务路由器",
                content=(
                    "你负责把用户输入路由到最合适的处理模式。"
                    "可选 route 只有 clarify/direct_answer/rag_answer/pdf_rag/multi_step。"
                    "必须返回 JSON，不要输出解释性文本。"
                ),
            ),
            model=self.model,
            output_language="chinese",
        )
        prompt = f"""
请根据用户输入和会话摘要做路由决策。

返回 JSON:
{{
  "route": "clarify/direct_answer/rag_answer/pdf_rag/multi_step",
  "reason": "简短原因",
  "should_retrieve": true,
  "should_verify": true,
  "max_retry": 1
}}

[会话摘要]
{history_summary or "当前没有历史上下文。"}

[用户输入]
{raw_input}
"""
        response = router_agent.step(prompt).msgs[0].content.strip()
        data = self._extract_json(response)
        return RouteDecision(
            route=data.get("route", "rag_answer"),
            reason=data.get("reason", "使用 LLM 路由。"),
            should_retrieve=bool(data.get("should_retrieve", True)),
            should_verify=bool(data.get("should_verify", True)),
            max_retry=max(0, int(data.get("max_retry", 1))),
        )

    def route(self, raw_input, history_summary=""):
        heuristic = self._heuristic_route(raw_input, history_summary)

        # 明显规则场景直接返回，避免把基础判断也交给模型
        if heuristic.route in {"pdf_rag", "clarify", "multi_step"}:
            return heuristic

        try:
            return self._llm_route(raw_input, history_summary)
        except Exception:
            return heuristic
