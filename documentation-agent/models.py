from typing import Literal
from pydantic import BaseModel, Field


class RAGResponse(BaseModel):
    """
    This model provides a structured answer with metadata about the response,
    including confidence, categorization, and follow-up suggestions.
    """

    answer: str = Field(description="The main answer to the user's question in markdown")
    found_answer: bool = Field(description="True if relevant information was found in the documentation")
    confidence: float = Field(description="Confidence score from 0.0 to 1.0 indicating how certain the answer is")
    confidence_explanation: str = Field(description="Explanation about the confidence level")
    answer_type: Literal["how-to", "explanation", "troubleshooting", "comparison", "reference"] = Field(description="The category of the answer")
    followup_questions: list[str] = Field(description="Suggested follow-up questions the user might want to ask")