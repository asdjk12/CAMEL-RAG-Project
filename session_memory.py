from dataclasses import dataclass, field


@dataclass
class SessionTurn:
    # 单轮会话记录，保留路由、检索与校验痕迹
    turn_id: int
    user_input: str
    standalone_query: str = ""
    route: str = ""
    route_reason: str = ""
    rewritten_query: str = ""
    final_answer: str = ""
    verification_status: str = ""
    verification_reason: str = ""


@dataclass
class SessionMemoryManager:
    # 强 agent 的记忆层：不只存聊天，还存任务状态与决策痕迹
    max_history_turns: int = 5
    current_goal: str = ""
    turns: list[SessionTurn] = field(default_factory=list)

    def clear(self):
        self.current_goal = ""
        self.turns.clear()

    def next_turn_id(self):
        return len(self.turns)

    def has_history(self):
        return bool(self.turns)

    def add_turn(
        self,
        user_input,
        standalone_query="",
        route="",
        route_reason="",
        rewritten_query="",
        final_answer="",
        verification_status="",
        verification_reason="",
    ):
        turn = SessionTurn(
            turn_id=len(self.turns) + 1,
            user_input=user_input,
            standalone_query=standalone_query,
            route=route,
            route_reason=route_reason,
            rewritten_query=rewritten_query,
            final_answer=final_answer,
            verification_status=verification_status,
            verification_reason=verification_reason,
        )
        self.turns.append(turn)
        if not self.current_goal:
            self.current_goal = user_input

    def format_history(self):
        if not self.turns:
            return "无历史对话。"

        parts = []
        for turn in self.turns[-self.max_history_turns :]:
            parts.append(
                "\n".join(
                    [
                        f"[第{turn.turn_id}轮]",
                        f"用户原始问题: {turn.user_input}",
                        f"独立问题: {turn.standalone_query}",
                        f"路由: {turn.route}",
                        f"回答: {turn.final_answer}",
                        f"校验状态: {turn.verification_status}",
                    ]
                )
            )
        return "\n\n".join(parts)

    def build_context_summary(self):
        # 给 router / verifier 用的简洁摘要，避免直接塞全部长历史
        if not self.turns:
            return "当前没有历史上下文。"

        recent_turns = self.turns[-self.max_history_turns :]
        parts = []
        if self.current_goal:
            parts.append(f"当前目标: {self.current_goal}")

        for turn in recent_turns:
            parts.append(
                "\n".join(
                    [
                        f"[第{turn.turn_id}轮摘要]",
                        f"问题: {turn.user_input}",
                        f"路由: {turn.route}",
                        f"结论: {turn.final_answer}",
                    ]
                )
            )

        return "\n\n".join(parts)
