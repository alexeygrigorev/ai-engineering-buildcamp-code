"""
Test Groq models for feature completeness:
- Responses API support
- Chat Completions support
- Streaming support
- Structured Output support
"""

import os
from typing import Literal
from pydantic import BaseModel
from openai import OpenAI

groq_client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY")
)


class TestResponse(BaseModel):
    summary: str
    word_count: int


# Models to test from Groq free plan
MODELS_TO_TEST = [
    "allam-2-7b",
    "canopylabs/orpheus-arabic-saudi",
    "canopylabs/orpheus-v1-english",
    "groq/compound",
    "groq/compound-mini",
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "meta-llama/llama-guard-4-12b",
    "meta-llama/llama-prompt-guard-2-22m",
    "meta-llama/llama-prompt-guard-2-86m",
    "moonshotai/kimi-k2-instruct",
    "moonshotai/kimi-k2-instruct-0905",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-safeguard-20b",
    "qwen/qwen3-32b",
]


def test_chat_completions(model: str) -> bool:
    """Test if model supports chat.completions"""
    try:
        response = groq_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=10
        )
        return response.choices[0].message.content is not None
    except Exception:
        return False


def test_responses(model: str) -> bool:
    """Test if model supports responses API"""
    try:
        response = groq_client.responses.create(
            model=model,
            input="hi"
        )
        return response.output_text is not None
    except Exception:
        return False


def test_streaming(model: str) -> bool:
    """Test if model supports streaming"""
    try:
        stream = groq_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
            max_tokens=10
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                return True
        return False
    except Exception:
        return False


def test_structured_output(model: str) -> bool:
    """Test if model supports structured output via chat.completions.parse"""
    try:
        response = groq_client.chat.completions.parse(
            model=model,
            messages=[{"role": "user", "content": "say hello in 3 words"}],
            response_format=TestResponse
        )
        return response.choices[0].message.parsed is not None
    except Exception:
        return False


def test_structured_output_responses(model: str) -> bool:
    """Test if model supports structured output via responses.parse"""
    try:
        response = groq_client.responses.parse(
            model=model,
            input="say hello in 3 words",
            text_format=TestResponse
        )
        return response.output_parsed is not None
    except Exception:
        return False


def main():
    print(f"{'Model':<45} {'Chat':<6} {'Resp':<6} {'Stream':<7} {'Struct':<7} {'SO-Resp':<9}")
    print("-" * 85)

    results = []

    for model in MODELS_TO_TEST:
        chat = test_chat_completions(model)
        resp = test_responses(model)
        stream = test_streaming(model)
        struct = test_structured_output(model)
        struct_resp = test_structured_output_responses(model)

        results.append({
            "model": model,
            "chat_completions": chat,
            "responses": resp,
            "streaming": stream,
            "structured_output": struct,
            "structured_output_responses": struct_resp
        })

        print(f"{model:<45} {'OK' if chat else 'NO':<6} {'OK' if resp else 'NO':<6} {'OK' if stream else 'NO':<7} {'OK' if struct else 'NO':<7} {'OK' if struct_resp else 'NO':<9}")

    print("\n" + "=" * 85)
    print("Legend: OK = Supported, NO = Not Supported")
    print("\nFeature Complete Models (all features supported):")

    for r in results:
        if all([
            r["chat_completions"],
            r["responses"],
            r["streaming"],
            r["structured_output"],
            r["structured_output_responses"]
        ]):
            print(f"  - {r['model']}")

    print("\nModels with Structured Output (chat.completions only):")
    for r in results:
        if r["structured_output"] and not r["structured_output_responses"]:
            print(f"  - {r['model']}")

    print("\nModels with Structured Output (responses only):")
    for r in results:
        if r["structured_output_responses"] and not r["structured_output"]:
            print(f"  - {r['model']}")


if __name__ == "__main__":
    main()
