ALIASES = {
    'gpt-4o-mini-2024-07-18': "gpt-4o-mini",
    'gpt-5-mini-2025-08-07': "gpt-5-nano",
}
    

MODEL_PRICES = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-5-nano": {"input": 0.075, "output": 0.30},
    "gpt-5-mini": {"input": 0.25, "output": 2.00},
    "gpt-5.2": {"input": 1.75, "output": 14.00},
    "gpt-5.2-pro": {"input": 21.00, "output": 168.00},
}

def calculate_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
    if model_name in ALIASES:
        model_name = ALIASES[model_name]

    prices = MODEL_PRICES[model_name.lower()]
    input_cost = (input_tokens / 1_000_000) * prices["input"]
    output_cost = (output_tokens / 1_000_000) * prices["output"]
    return input_cost + output_cost


def calculate_cost_response(response) -> float:
    usage = response.usage

    return calculate_cost(
        model_name=response.model,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
    )
    