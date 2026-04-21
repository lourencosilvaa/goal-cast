from dataclasses import asdict, dataclass


@dataclass
class EvaluationEntry:
    match_date: str
    home_team: str
    away_team: str
    predicted_outcome: str
    home_win_prob: float
    draw_prob: float
    away_win_prob: float
    confidence: float

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "EvaluationEntry":
        return cls(
            match_date=data["match_date"],
            home_team=data["home_team"],
            away_team=data["away_team"],
            predicted_outcome=data["predicted_outcome"],
            home_win_prob=data["home_win_prob"],
            draw_prob=data["draw_prob"],
            away_win_prob=data["away_win_prob"],
            confidence=data["confidence"],
        )
