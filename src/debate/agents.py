from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from typing import Dict, List

from src.debate.prompts import (
    DOCTOR_A_SYSTEM,
    DOCTOR_B_SYSTEM,
    ROUND_1_USER_TEMPLATE,
    REBUTTAL_USER_TEMPLATE,
    format_evidence_for_prompt,
    parse_agent_response,
)

# ── Token Usage Tracking ─────────────────────────────────────────

_token_usage: Dict[str, int] = {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
    "num_calls": 0,
}


def reset_token_usage() -> None:
    """Reset the token usage counters (call before each experiment)."""
    _token_usage["prompt_tokens"] = 0
    _token_usage["completion_tokens"] = 0
    _token_usage["total_tokens"] = 0
    _token_usage["num_calls"] = 0


def get_token_usage() -> Dict[str, int]:
    """Return a copy of the current token usage stats."""
    return dict(_token_usage)


def _track_tokens(response) -> None:
    """Extract and accumulate token usage from an LLM response."""
    usage = getattr(response, "response_metadata", {}).get("token_usage", {})
    if not usage:
        usage = getattr(response, "usage_metadata", {}) or {}
    _token_usage["prompt_tokens"] += usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0)
    _token_usage["completion_tokens"] += usage.get("completion_tokens", 0) or usage.get("output_tokens", 0)
    _token_usage["total_tokens"] += usage.get("total_tokens", 0) or (
        usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
    )
    _token_usage["num_calls"] += 1


def create_llm(model: str = "openai/gpt-4o-mini", temperature: float = 0.7) -> ChatOpenAI:
    """Create the shared LLM instance. Supports OpenRouter via OPENROUTER_API_KEY."""
    import os
    api_key = os.getenv("OPENROUTER_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    base_url = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
    )


def build_agent_messages(system_prompt: str, history: List[dict]) -> List:
    """Convert system prompt + message history into LangChain message objects."""
    messages = [SystemMessage(content=system_prompt)]
    for msg in history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    return messages


def invoke_agent(
    llm: ChatOpenAI,
    system_prompt: str,
    history: List[dict],
    user_message: str,
) -> str:
    """Invoke a single agent turn: append user message, call LLM, return text."""
    history.append({"role": "user", "content": user_message})
    messages = build_agent_messages(system_prompt, history)
    response = llm.invoke(messages)
    _track_tokens(response)
    assistant_text = response.content
    history.append({"role": "assistant", "content": assistant_text})
    return assistant_text


def run_doctor_a_opening(
    llm: ChatOpenAI,
    question: str,
    evidence: List[dict],
    history: List[dict],
) -> dict:
    """Doctor A's opening argument (round 1)."""
    formatted_ev = format_evidence_for_prompt(evidence)
    user_msg = ROUND_1_USER_TEMPLATE.format(
        question=question, formatted_evidence=formatted_ev
    )
    response_text = invoke_agent(llm, DOCTOR_A_SYSTEM, history, user_msg)
    return parse_agent_response(response_text, "doctor_a", round_num=1)


def run_doctor_b_opening(
    llm: ChatOpenAI,
    question: str,
    evidence: List[dict],
    history: List[dict],
) -> dict:
    """Doctor B's opening argument (round 1)."""
    formatted_ev = format_evidence_for_prompt(evidence)
    user_msg = ROUND_1_USER_TEMPLATE.format(
        question=question, formatted_evidence=formatted_ev
    )
    response_text = invoke_agent(llm, DOCTOR_B_SYSTEM, history, user_msg)
    return parse_agent_response(response_text, "doctor_b", round_num=1)


def run_rebuttal(
    llm: ChatOpenAI,
    agent_name: str,
    system_prompt: str,
    opponent_message: dict,
    history: List[dict],
    round_num: int,
) -> dict:
    """Run a rebuttal turn for either agent."""
    evidence_cited = opponent_message.get("evidence_cited", [])
    evidence_str = ", ".join(evidence_cited) if evidence_cited else "none"
    user_msg = REBUTTAL_USER_TEMPLATE.format(
        opponent_position=opponent_message["position"],
        opponent_confidence=opponent_message["confidence"],
        opponent_evidence=evidence_str,
        opponent_reasoning=opponent_message["reasoning"],
    )
    response_text = invoke_agent(llm, system_prompt, history, user_msg)
    return parse_agent_response(response_text, agent_name, round_num)
