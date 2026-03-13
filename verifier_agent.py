import json
import re
from dataclasses import asdict, dataclass

from camel.agents import ChatAgent
from camel.messages import BaseMessage


@dataclass
class VerificationResult:
    # 校验结果：决定是否接受、重试还是先澄清
    status: str
    score: int
    reason: str
    suggested_action: str

    def to_dict(self):
        return asdict(self)


class AnswerVerifier:
    def __init__(self, model):
        self.model = model

    def _extract_json(self, text):
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            raise ValueError("verifier json not found")
        return json.loads(match.group(0))

    def _heuristic_verify(self, question, answer, retrieval_result="", route=""):
        # 对极端情况做兜底，避免校验模块本身变成阻塞点
        if route == "clarify":
            return VerificationResult(
                status="pass",
                score=100,
                reason="当前策略是先澄清用户需求。",
                suggested_action="accept",
            )

        if not answer.strip():
            return VerificationResult(
                status="fail",
                score=0,
                reason="回答为空。",
                suggested_action="retry",
            )

        if retrieval_result and len(answer.strip()) < 20:
            return VerificationResult(
                status="fail",
                score=35,
                reason="已有检索证据，但回答过短，疑似未充分利用证据。",
                suggested_action="retry",
            )

        return VerificationResult(
            status="pass",
            score=80,
            reason="启发式校验通过。",
            suggested_action="accept",
        )

    def verify(
        self,
        question,
        answer,
        retrieval_result="",
        history_summary="",
        route="rag_answer",
    ):
        verifier = ChatAgent(
            system_message=BaseMessage.make_assistant_message(
                role_name="答案校验器",
                content=(
                    "你负责检查当前回答是否真正回答了问题、是否依赖了证据、"
                    "以及是否需要重试或先澄清。必须返回 JSON。"
                ),
            ),
            model=self.model,
            output_language="chinese",
        )
        prompt = f"""
请校验下面这次回答。

返回 JSON:
{{
  "status": "pass/fail",
  "score": 0,
  "reason": "简短原因",
  "suggested_action": "accept/retry/clarify"
}}

[route]
{route}

[history_summary]
{history_summary or "当前没有历史上下文。"}

[question]
{question}

[retrieval_result]
{retrieval_result or "当前没有显式检索证据。"}

[answer]
{answer}
"""

        try:
            response = verifier.step(prompt).msgs[0].content.strip()
            data = self._extract_json(response)
            return VerificationResult(
                status=data.get("status", "fail"),
                score=max(0, min(100, int(data.get("score", 0)))),
                reason=data.get("reason", "校验器未提供原因。"),
                suggested_action=data.get("suggested_action", "retry"),
            )
        except Exception:
            return self._heuristic_verify(
                question=question,
                answer=answer,
                retrieval_result=retrieval_result,
                route=route,
            )
