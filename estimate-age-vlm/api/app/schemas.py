from typing import Optional

from pydantic import BaseModel


class FinalClassification(BaseModel):
    estimated_age: str
    confidence_score: float
    logic_path: str


class EstimationResponse(BaseModel):
    priority_analysis: dict
    rule_applied: str
    final_classification: FinalClassification
    cost: float  # injected by estimator.py from token usage metadata, not LLM output
    observation: Optional[dict] = None  # populated only in two_step mode