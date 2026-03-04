from typing import List, Optional
from pydantic import BaseModel, Field

class EvaluationCriterion(BaseModel):
    """Evaluation of a single performance requirement."""
    name: str = Field(description="Name of the criterion (e.g., groundedness, relevance)")
    passed: bool = Field(description="Whether the criterion was met")
    score: float = Field(description="Score from 0.0 to 1.0", ge=0.0, le=1.0)
    reasoning: str = Field(description="Brief explanation of the score based on evidence")


class EvaluationResult(BaseModel):
    """The complete evaluation report for an agent interaction."""
    session_id: str = Field(description="The session ID being evaluated")
    overall_score: float = Field(description="Average score across all criteria")
    criteria: List[EvaluationCriterion] = Field(description="Individual scores for each criterion")
    summary: str = Field(description="A high-level summary of the agent's performance in this interaction")
    improvement_suggestions: Optional[str] = Field(description="Actionable advice for improving the agent")
