from dataclasses import dataclass

SYSTEM_PREFIX = "<|turn>system\n<|think|>\n"
MODEL_OPEN    = "<|turn>model\n"
TURN_CLOSE    = "<turn|>"

@dataclass
class TurnMeta:
    turn: int
    label: str
    user_start: int
    user_end: int
    model_start: int
    model_end: int
    stop_type: str
    tokens_predicted: int
    seed_used: int
    gen_tokens: list[int] | None
    
    def to_dict(self) -> dict:
        return {
            'turn': self.turn,
            'label': self.label,
            'user_start': self.user_start,
            'user_end': self.user_end,
            'model_start': self.model_start,
            'model_end': self.model_end,
            'stop_type': self.stop_type,
            'tokens_predicted': self.tokens_predicted,
            'seed_used': self.seed_used,
            'gen_tokens': self.gen_tokens
        }

def render_header(system: str | None) -> str:
    if system is None or system == "":
        return SYSTEM_PREFIX + TURN_CLOSE + "\n"
    else:
        return SYSTEM_PREFIX + f"{system}{TURN_CLOSE}\n"

def render_user_turn(user: str) -> str:
    return f"<|turn>user\n{user}{TURN_CLOSE}\n"

def finalize_answer(answer: str) -> str:
    t = answer.rstrip()
    if not t.endswith("<turn|>"):
        return t + "<turn|>"
    else:
        return answer

class Transcript:
    body: str
    turns: list[TurnMeta]

    def __init__(self, system: str | None) -> None:
        self.body = render_header(system)
        self.turns = []

    def open_user_turn(self, user: str) -> tuple[int, int, int]:
        user_start = len(self.body)
        self.body += render_user_turn(user)
        user_end = len(self.body)
        self.body += MODEL_OPEN
        model_start = len(self.body)
        return (user_start, user_end, model_start)

    def rollback(self, user_start: int) -> None:
        self.body = self.body[:user_start]

    def close_model_turn(self, *, turn: int, label: str, user_start: int, user_end: int,
                         model_start: int, answer: str, stop_type: str,
                         tokens_predicted: int, seed_used: int,
                         gen_tokens: list[int] | None) -> TurnMeta:
        self.body += finalize_answer(answer)
        model_end = len(self.body)
        self.body += "\n"
        meta = TurnMeta(
            turn=turn,
            label=label,
            user_start=user_start,
            user_end=user_end,
            model_start=model_start,
            model_end=model_end,
            stop_type=stop_type,
            tokens_predicted=tokens_predicted,
            seed_used=seed_used,
            gen_tokens=gen_tokens
        )
        self.turns.append(meta)
        return meta
