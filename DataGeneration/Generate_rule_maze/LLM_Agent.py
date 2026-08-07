from generate_maze import REGULAR
import os
import time
import json
import yaml
from openai import OpenAI

DEFAULT_API_BASE = "https://yunwu.ai/v1"
# Supported model aliases on the proxy: gemini-3-flash-preview, gemini-3-pro-preview, gemini-2.5-flash, gemini-2.5-pro, etc.
DEFAULT_MODEL = "gemini-3-pro-preview"


class GEMINI_AGENT:
    def __init__(self, api_key, model_name=DEFAULT_MODEL, api_base=DEFAULT_API_BASE):
        self.model_name = model_name
        self.client = OpenAI(
            base_url=api_base,
            api_key=api_key,
        )

    def chat_completion(self, messages, timeout=60, temperature=0.7, max_tokens=2048,
                        max_retries=3, retry_delay=5):
        last_exc = None
        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    timeout=timeout,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return response
            except Exception as e:
                last_exc = e
                print(f"[LLM] Attempt {attempt}/{max_retries} failed: {e}")
                if attempt < max_retries:
                    time.sleep(retry_delay)
        raise RuntimeError(f"LLM call failed after {max_retries} retries") from last_exc


if __name__ == "__main__":
    _apikey_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "apikey.yaml")
    with open(_apikey_path, "r") as f:
        api_key_data = yaml.safe_load(f)
    api_key = api_key_data["API_KEY"]
    api_base = api_key_data.get("API_BASE", DEFAULT_API_BASE)
    gemini_agent = GEMINI_AGENT(api_key=api_key, api_base=api_base)
