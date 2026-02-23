from typing import Literal
from pydantic import BaseModel, Field


class Check(BaseModel):
    rule: str = Field(description="The rule being verified")
    passed: bool = Field(description="Whether the response complies with this rule")
    explanation: str = Field(description="Evidence: what in the response passes or violates this rule")


class Reference(BaseModel):
    """
    A reference to a document that was used to generate the answer.
    """

    file_path: str = Field(description="The path to the document")
    explanation: str = Field(description="The explanation of how this document is relevant and was used in the answer")


class RAGResponse(BaseModel):
    """
    This model provides a structured answer with metadata about the response,
    including confidence, categorization, and follow-up suggestions.
    """

    answer: str = Field(description="The main answer to the user's question in markdown")
    found_answer: bool = Field(description="True if relevant information was found in the documentation")
    references: list[Reference] = Field(description="List of references to documents used to generate the answer")
    confidence: float = Field(description="Confidence score from 0.0 to 1.0 indicating how certain the answer is")
    confidence_explanation: str = Field(description="Explanation about the confidence level")
    answer_type: Literal["how-to", "explanation", "troubleshooting", "comparison", "reference"] = Field(description="The category of the answer")
    followup_questions: list[str] = Field(description="Suggested follow-up questions the user might want to ask")
    checks: list[Check] = Field(
        description="Self-verification checklist. For each applicable rule, verify your response complies. If any check fails, fix your response before submitting."
    )

    def to_string(self):
        parts = []

        parts.append(self.answer)
        parts.append("")
        parts.append(f"found_answer: {self.found_answer}")
        parts.append(f"confidence: {self.confidence}")
        parts.append(f"confidence_explanation: {self.confidence_explanation}")
        parts.append(f"answer_type: {self.answer_type}")

        for ref in self.references:
            parts.append(f"reference: {ref.file_path} — {ref.explanation}")

        for q in self.followup_questions:
            parts.append(f"follow_up_question: {q}")

        for check in self.checks:
            parts.append(f"self_check: [{'+' if check.passed else '-'}] {check.rule} — {check.explanation}")

        return "\n".join(parts)

    def __str__(self):
        return self.to_string()