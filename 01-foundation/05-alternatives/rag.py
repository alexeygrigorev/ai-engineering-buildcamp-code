
import json

from typing import Literal
from pydantic import BaseModel, Field

from minsearch import Index
from gitsource import GithubRepositoryDataReader, chunk_documents


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


def initialize_index():
    reader = GithubRepositoryDataReader(
        repo_owner="evidentlyai",
        repo_name="docs",
        allowed_extensions={"md", "mdx"},
    )
    files = reader.read()

    parsed_docs = [doc.parse() for doc in files]
    chunked_docs = chunk_documents(parsed_docs, size=3000, step=1500)

    index = Index(
        text_fields=["title", "description", "content"],
        keyword_fields=["filename"]
    )
    index.fit(chunked_docs)

    print(f"Indexed {len(chunked_docs)} chunks from {len(files)} documents")
    return index


RAG_INSTRUCTIONS = """
You're a documentation assistant. Answer the QUESTION based on the CONTEXT from our documentation.

Use only facts from the CONTEXT when answering.
If the answer isn't in the CONTEXT, say so.
"""

RAG_PROMPT_TEMPLATE = """
<QUESTION>
{question}
</QUESTION>

<CONTEXT>
{context}
</CONTEXT>
""".strip()


class RAG:

    def __init__(self, index, llm_client):
        self.index = index
        self.llm_client = llm_client

        self.output_type = RAGResponse
        self.rag_instructions = RAG_INSTRUCTIONS

    def search(self, query):
        results = self.index.search(
            query=query,
            num_results=5
        )
        return results

    def build_prompt(self, question, search_results):
        context = json.dumps(search_results, indent=2)
        return RAG_PROMPT_TEMPLATE.format(
            question=question,
            context=context
        )

    def llm(self, user_prompt, model="gpt-4o-mini"):
        messages = [
            {"role": "system", "content": self.rag_instructions},
            {"role": "user", "content": user_prompt}
        ]

        response = self.llm_client.responses.parse(
            model=model,
            input=messages,
            text_format=self.output_type
        )

        return response.output_parsed

    def rag(self, question):
        search_results = self.search(question)
        prompt = self.build_prompt(question, search_results)
        answer = self.llm(prompt)
        return answer
