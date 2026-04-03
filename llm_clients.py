# llm_clients.py — LLM API clients for Novus FinLLM
"""
Centralized LLM client configuration and call wrappers.
Supports DeepSeek V3.2 (fast/cheap) and DeepSeek R1 (deep reasoning) via OpenAI-compatible API.
Also supports Google Gemini.

Routing strategy:
  - extraction.py          → deepseek-chat (V3.2)     — fast, structured JSON
  - analytical agents      → deepseek-reasoner (R1)   — deep equity reasoning
  - pm_synthesis.py        → deepseek-reasoner (R1)   — final investment thesis
"""

import os
import time
import json as _json
from dotenv import load_dotenv
from openai import OpenAI
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Configure the DeepSeek API client
_ENABLE_DEEPSEEK_DEBUG_LOGS = os.getenv("ENABLE_DEEPSEEK_DEBUG_LOGS", "false").lower() in ("1", "true", "yes", "on")
client = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")

# Model names
DEEPSEEK_V3  = 'deepseek-chat'       # Fast, cheap — use for extraction & formatting
DEEPSEEK_R1  = 'deepseek-reasoner'   # Slow, deep — use for all equity analysis agents

# Legacy alias (keeps backward compatibility)
deepseek_model_name = DEEPSEEK_V3

# Configure the Gemini API client
_ENABLE_GEMINI_DEBUG_LOGS = os.getenv("ENABLE_GEMINI_DEBUG_LOGS", "false").lower() in ("1", "true", "yes", "on")
if gemini_api_key and gemini_api_key != "replace_me":
    genai.configure(api_key=gemini_api_key)
    gemini_client = True
else:
    gemini_client = None


def call_gemini(prompt, text_to_analyze, send_financials=False, financial_data=None, extra_context=None):
    """Calls the Gemini API with a specific prompt and text, optionally including financial data."""
    if gemini_client is None:
        return "Error: Gemini API key is not configured."

    # --- Refined Prompt Construction ---
    if send_financials and financial_data:
        description = "transcripts and historical financial statements"
        content_to_send = f"{text_to_analyze}\n\n---\n\n{financial_data}"
    else:
        description = "transcripts"
        content_to_send = text_to_analyze

    if extra_context:
        content_to_send = f"{content_to_send}\n\n---\n\nPrior context from previous step:\n{extra_context}"

    full_prompt = f"System:\n{prompt}\n\nUser Data:\n{content_to_send}"

    try:
        if _ENABLE_GEMINI_DEBUG_LOGS:
            try:
                _debug_path = os.path.join(os.path.dirname(__file__), "gemini_input_debug.txt")
                with open(_debug_path, "a", encoding="utf-8") as _f:
                    _f.write("\n\n=== GEMINI_CALL INPUT @ " + time.strftime("%Y-%m-%d %H:%M:%S") + " ===\n")
                    _f.write("-- send_financials: " + str(bool(send_financials)) + "\n")
                    _f.write("-- extra_context: " + ("yes" if bool(extra_context) else "no") + "\n")
                    _f.write("-- prompt:\n" + (prompt if isinstance(prompt, str) else _json.dumps(prompt, ensure_ascii=False)) + "\n")
                    _f.write("-- content_to_analyze (possibly markdown):\n" + (text_to_analyze if isinstance(text_to_analyze, str) else _json.dumps(text_to_analyze, ensure_ascii=False)) + "\n")
                    if send_financials and financial_data is not None:
                        _f.write("-- financial_data (truncated to 10k chars):\n")
                        _fd = financial_data if isinstance(financial_data, str) else _json.dumps(financial_data, ensure_ascii=False)
                        _f.write(str(_fd)[:10000] + ("...\n" if len(str(_fd)) > 10000 else "\n"))
            except Exception as _e0:
                print(f"[debug] Failed to log Gemini input: {_e0}")

        model = genai.GenerativeModel(
            model_name='gemini-1.5-flash',
            system_instruction=prompt
        )
        response = model.generate_content(
            contents=f"Here are the {description} to analyze:\n\n---\n\n{content_to_send}",
            generation_config=genai.types.GenerationConfig(
                temperature=0.2,
                max_output_tokens=8192
            )
        )

        if _ENABLE_GEMINI_DEBUG_LOGS:
            try:
                _debug_path = os.path.join(os.path.dirname(__file__), "gemini_output_debug.txt")
                with open(_debug_path, "a", encoding="utf-8") as _f:
                    _f.write("\n\n=== GEMINI_CALL RAW OUTPUT @ " + time.strftime("%Y-%m-%d %H:%M:%S") + " ===\n")
                    _f.write(str(response.text) + "\n")
                    _f.write("=== END RAW OUTPUT ===\n")
            except Exception as _e1:
                print(f"[debug] Failed to log Gemini output: {_e1}")

        return response.text
    except Exception as e:
        print(f"An error occurred with the Gemini API: {e}")
        return f"Error: Could not generate content from AI. Details: {e}"


def call_deepseek(prompt, text_to_analyze, send_financials=False, financial_data=None, extra_context=None):
    """Calls the DeepSeek API with a specific prompt and text, optionally including financial data."""

    # --- Refined Prompt Construction ---
    if send_financials and financial_data:
        description = "transcripts and historical financial statements"
        content_to_send = f"{text_to_analyze}\n\n---\n\n{financial_data}"
    else:
        description = "transcripts"
        content_to_send = text_to_analyze

    if extra_context:
        content_to_send = f"{content_to_send}\n\n---\n\nPrior context from previous step:\n{extra_context}"

    user_message = f"Here are the {description} to analyze:\n\n---\n\n{content_to_send}"

    try:
        if _ENABLE_DEEPSEEK_DEBUG_LOGS:
            try:
                _debug_path = os.path.join(os.path.dirname(__file__), "deepseek_input_debug.txt")
                with open(_debug_path, "a", encoding="utf-8") as _f:
                    _f.write("\n\n=== DEEPSEEK_CALL INPUT @ " + time.strftime("%Y-%m-%d %H:%M:%S") + " ===\n")
                    _f.write("-- send_financials: " + str(bool(send_financials)) + "\n")
                    _f.write("-- extra_context: " + ("yes" if bool(extra_context) else "no") + "\n")
                    _f.write("-- prompt:\n" + (prompt if isinstance(prompt, str) else _json.dumps(prompt, ensure_ascii=False)) + "\n")
                    _f.write("-- content_to_analyze (possibly markdown):\n" + (text_to_analyze if isinstance(text_to_analyze, str) else _json.dumps(text_to_analyze, ensure_ascii=False)) + "\n")
                    if send_financials and financial_data is not None:
                        _f.write("-- financial_data (truncated to 10k chars):\n")
                        _fd = financial_data if isinstance(financial_data, str) else _json.dumps(financial_data, ensure_ascii=False)
                        _f.write(str(_fd)[:10000] + ("...\n" if len(str(_fd)) > 10000 else "\n"))
                    _f.write("=== END INPUT ===\n")
            except Exception as _e0:
                print(f"[debug] Failed to log DeepSeek input: {_e0}")

        response = client.chat.completions.create(
            model=deepseek_model_name,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.2,
            max_tokens=8192
        )
        response_text = response.choices[0].message.content

        if _ENABLE_DEEPSEEK_DEBUG_LOGS:
            try:
                _debug_path = os.path.join(os.path.dirname(__file__), "deepseek_input_debug.txt")
                with open(_debug_path, "a", encoding="utf-8") as _f:
                    _f.write("\n=== DEEPSEEK_CALL OUTPUT @ " + time.strftime("%Y-%m-%d %H:%M:%S") + " ===\n")
                    _f.write((response_text or "") + "\n")
                    _f.write("=== END OUTPUT ===\n")
            except Exception as _e1:
                print(f"[debug] Failed to log DeepSeek output: {_e1}")

        return response_text
    except Exception as e:
        print(f"An error occurred with the DeepSeek API: {e}")
        return f"Error: Could not generate content from AI. Details: {e}"


def call_deepseek_r1(prompt, text_to_analyze, send_financials=False, financial_data=None, extra_context=None):
    """
    Calls DeepSeek-R1 (deepseek-reasoner) — the deep reasoning model.
    Use this for all equity analysis agents where quality > speed.
    Note: R1 does NOT support temperature; it uses its own internal chain-of-thought.
    """
    # --- Prompt Construction ---
    if send_financials and financial_data:
        description = "transcripts and historical financial statements"
        content_to_send = f"{text_to_analyze}\n\n---\n\n{financial_data}"
    else:
        description = "transcripts"
        content_to_send = text_to_analyze

    if extra_context:
        content_to_send = f"{content_to_send}\n\n---\n\nPrior context from previous step:\n{extra_context}"

    user_message = f"Here are the {description} to analyze:\n\n---\n\n{content_to_send}"

    try:
        if _ENABLE_DEEPSEEK_DEBUG_LOGS:
            try:
                _debug_path = os.path.join(os.path.dirname(__file__), "deepseek_r1_input_debug.txt")
                with open(_debug_path, "a", encoding="utf-8") as _f:
                    _f.write("\n\n=== R1_CALL INPUT @ " + time.strftime("%Y-%m-%d %H:%M:%S") + " ===\n")
                    _f.write("-- prompt:\n" + (prompt if isinstance(prompt, str) else _json.dumps(prompt, ensure_ascii=False)) + "\n")
                    _f.write("=== END INPUT ===\n")
            except Exception as _e0:
                print(f"[debug] Failed to log R1 input: {_e0}")

        response = client.chat.completions.create(
            model=DEEPSEEK_R1,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_message}
            ],
            max_tokens=16000   # R1 benefits from generous token budget for chain-of-thought
        )
        response_text = response.choices[0].message.content

        if _ENABLE_DEEPSEEK_DEBUG_LOGS:
            try:
                _debug_path = os.path.join(os.path.dirname(__file__), "deepseek_r1_input_debug.txt")
                with open(_debug_path, "a", encoding="utf-8") as _f:
                    _f.write("\n=== R1_CALL OUTPUT @ " + time.strftime("%Y-%m-%d %H:%M:%S") + " ===\n")
                    _f.write((response_text or "") + "\n")
                    _f.write("=== END OUTPUT ===\n")
            except Exception as _e1:
                print(f"[debug] Failed to log R1 output: {_e1}")

        return response_text
    except Exception as e:
        print(f"An error occurred with DeepSeek R1: {e}")
        return f"Error: Could not generate content from DeepSeek R1. Details: {e}"


def call_deepseek_auto(prompt, text_to_analyze, use_r1=True, send_financials=False, financial_data=None, extra_context=None):
    """
    Smart router: use use_r1=True for analytical agents (default), use_r1=False for extraction.
    """
    if use_r1:
        return call_deepseek_r1(prompt, text_to_analyze, send_financials=send_financials,
                                financial_data=financial_data, extra_context=extra_context)
    else:
        return call_deepseek(prompt, text_to_analyze, send_financials=send_financials,
                             financial_data=financial_data, extra_context=extra_context)
