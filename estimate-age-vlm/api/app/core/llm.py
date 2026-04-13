from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI

from app.loggers.logger import logger

# Pricing per 1M tokens (USD) — source: OpenAI pricing page
# - https://developers.openai.com/api/docs/pricing
PRICING: dict[str, dict[str, float]] = {
    "gpt-5.4":      {"input": 2.50,  "cached_input": 0.25,  "output": 15.00},
    "gpt-5.4-mini": {"input": 0.75,  "cached_input": 0.075, "output": 4.50},
    "gpt-5.4-nano": {"input": 0.20,  "cached_input": 0.02,  "output": 1.25},
    "gpt-5.4-pro":  {"input": 30.00, "cached_input": 0.0,   "output": 180.00},
    "gpt-4o":       {"input": 5.00,  "cached_input": 1.25,  "output": 10.00},
}


class LLMClient(ChatOpenAI):
    def __init__(
        self,
        model: str,
        api_key: str,
        max_tokens: int = 4096,
        timeout: int = 180,
        temperature: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(
            model=model,
            api_key=api_key,
            max_tokens=max_tokens,
            timeout=timeout,
            temperature=temperature,
            **kwargs,
        )

    def invoke(self, input, config=None, **kwargs) -> BaseMessage:
        response = super().invoke(input, config=config, **kwargs)
        usage = getattr(response, "usage_metadata", None)
        if usage:
            cost_data = self._calculate_cost(
                usage.get("input_tokens", 0),
                usage.get("output_tokens", 0),
                usage.get("input_token_details", {}).get("cache_read", 0),
            )
            if cost_data:
                response.cost = cost_data  # 👈 attach

        self._log_usage(response)
        return response

    def _log_usage(self, response: BaseMessage) -> None:
        usage = getattr(response, "usage_metadata", None)
        if not usage:
            logger.warning("No usage metadata in LLM response")
            return

        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)
        cache_read = usage.get("input_token_details", {}).get("cache_read", 0)
        cost = self._calculate_cost(input_tokens, output_tokens, cache_read)

        if cost is not None:
            logger.info(
                "Token usage — input: %d (cached: %d) | output: %d | total: %d | cost: $%.6f",
                input_tokens, cache_read, output_tokens, total_tokens, cost,
            )
        else:
            logger.info(
                "Token usage — input: %d (cached: %d) | output: %d | total: %d | cost: N/A (model not in pricing table)",
                input_tokens, cache_read, output_tokens, total_tokens,
            )

    def _calculate_cost(self, input_tokens: int, output_tokens: int, cache_read: int = 0) -> float | None:
        pricing = PRICING.get(self.model_name)
        if not pricing:
            return None
        non_cached_input = input_tokens - cache_read
        input_cost = (non_cached_input / 1_000_000) * pricing["input"]
        cached_cost = (cache_read / 1_000_000) * pricing["cached_input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        return input_cost + cached_cost + output_cost
