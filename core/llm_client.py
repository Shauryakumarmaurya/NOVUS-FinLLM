"""
novus_v3/core/llm_client.py — Unified LLM Client with Tool Calling

Model routing (top-notch equity analysis mode):
    DEEPSEEK_R1  = deepseek-reasoner   → ALL analytical agents (default)
    DEEPSEEK_V3  = deepseek-chat       → extraction.py only (fast + cheap)

Note: R1 does NOT accept a temperature parameter — it uses its own internal
chain-of-thought. This client automatically omits temperature when using R1.

Supports:
    1. Simple calls (backward compatible)
    2. Tool/function calling (DeepSeek + OpenAI compatible)
    3. Multi-turn conversations
    4. Reasoning trace extraction (<think> tags)
    5. Automatic retry with exponential backoff
"""

import json
import time
import re
import os
from typing import Optional, Any
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class LLMResponse:
    """Structured response from any LLM call."""
    content: str = ""
    thinking: Optional[str] = None        # DeepSeek R1 <think> trace
    tool_calls: list[dict] = field(default_factory=list)
    finish_reason: str = "stop"           # "stop" | "tool_calls" | "length"
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0

    @property
    def is_final(self) -> bool:
        """True if the model produced a final answer (no more tool calls needed)."""
        return not self.has_tool_calls


class LLMClient:
    """
    Unified client for DeepSeek R1 / V3 with function calling.
    
    Uses the OpenAI-compatible API that DeepSeek provides.
    Drop-in replacement for your current call_deepseek().
    """

    # Model name constants
    DEEPSEEK_R1 = "deepseek-reasoner"   # Deep reasoning — use for ALL analysis agents
    DEEPSEEK_V3 = "deepseek-chat"       # Fast/cheap    — use for extraction only

    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        model: str = "deepseek-reasoner",  # R1 is the default now — top-notch analysis
        max_retries: int = 3,
        timeout: int = 180,                # R1 needs more time for chain-of-thought
    ):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
        self.base_url = base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout
        self._client = None

    @property
    def is_r1(self) -> bool:
        """True if this client is using the R1 reasoning model."""
        return "reasoner" in self.model

    def _get_client(self):
        if self._client is None:
            try:
                import openai
                self._client = openai.OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                    timeout=self.timeout,
                )
            except ImportError:
                raise ImportError("pip install openai — required for DeepSeek API access")
        return self._client

    # ── Simple call (backward compatible with your current call_deepseek) ──

    def call_simple(self, system_prompt: str, user_content: str) -> str:
        """Drop-in replacement for call_deepseek(system_prompt, user_content)."""
        response = self.call(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]
        )
        return response.content

    # ── Full call with tool support ──

    def call(
        self,
        messages: list[dict],
        tools: list[dict] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,   # defaults to 16k for R1, 4k for V3
    ) -> LLMResponse:
        """
        Full LLM call with tool/function calling support.
        
        Args:
            messages: Conversation history [{role, content}, ...]
            tools: Tool definitions for function calling
            temperature: 0.0-1.0 — IGNORED for R1 (automatically omitted)
            max_tokens: Max output tokens. Defaults to 16000 for R1, 4096 for V3.
            
        Returns:
            LLMResponse with content, tool_calls, thinking trace, etc.
        """
        client = self._get_client()

        # R1 does not accept temperature — omit it entirely
        effective_max_tokens = max_tokens or (16000 if self.is_r1 else 4096)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": effective_max_tokens,
        }
        if not self.is_r1:
            kwargs["temperature"] = temperature  # only set for V3
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                start = time.time()
                raw = client.chat.completions.create(**kwargs)
                latency = int((time.time() - start) * 1000)

                msg = raw.choices[0].message
                content = msg.content or ""
                finish = raw.choices[0].finish_reason or "stop"

                # Extract <think> reasoning trace
                thinking = None
                think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
                if think_match:
                    thinking = think_match.group(1).strip()
                    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

                # Extract tool calls
                tool_calls = []
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        try:
                            args = json.loads(tc.function.arguments)
                        except (json.JSONDecodeError, TypeError):
                            args = {}
                        tool_calls.append({
                            "id": tc.id,
                            "name": tc.function.name,
                            "arguments": args,
                        })

                usage = raw.usage
                return LLMResponse(
                    content=content,
                    thinking=thinking,
                    tool_calls=tool_calls,
                    finish_reason="tool_calls" if tool_calls else finish,
                    input_tokens=usage.prompt_tokens if usage else 0,
                    output_tokens=usage.completion_tokens if usage else 0,
                    latency_ms=latency,
                )

            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    wait = 2 ** attempt
                    print(f"[LLM] Attempt {attempt} failed: {e}. Retrying in {wait}s...")
                    time.sleep(wait)

        return LLMResponse(
            content=f"[LLM ERROR after {self.max_retries} retries] {last_error}",
            finish_reason="error",
        )


# ── Module-level singletons ─────────────────────────────────────────────

_r1_client:  Optional[LLMClient] = None   # R1 — all analytical agents
_v3_client:  Optional[LLMClient] = None   # V3 — extraction only

def get_llm_client(use_r1: bool = True) -> LLMClient:
    """
    Returns the appropriate LLM client.
    
    use_r1=True  (default) → DeepSeek-R1 (deepseek-reasoner) for top-notch equity analysis
    use_r1=False           → DeepSeek-V3 (deepseek-chat) for fast extraction
    """
    global _r1_client, _v3_client
    if use_r1:
        if _r1_client is None:
            _r1_client = LLMClient(model=LLMClient.DEEPSEEK_R1)
        return _r1_client
    else:
        if _v3_client is None:
            _v3_client = LLMClient(model=LLMClient.DEEPSEEK_V3)
        return _v3_client


def get_r1_client() -> LLMClient:
    """Shortcut: DeepSeek-R1 for deep analytical reasoning."""
    return get_llm_client(use_r1=True)


def get_v3_client() -> LLMClient:
    """Shortcut: DeepSeek-V3 for fast extraction and formatting."""
    return get_llm_client(use_r1=False)
