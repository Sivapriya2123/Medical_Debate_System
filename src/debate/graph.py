from langgraph.graph import StateGraph, END
from typing import List, Dict, Any

from src.debate.models import DebateState, DebateTranscript, AgentMessage, EvidenceItem
from src.debate.agents import (
    create_llm,
    run_doctor_a_opening,
    run_doctor_b_opening,
    run_rebuttal,
)
from src.debate.prompts import DOCTOR_A_SYSTEM, DOCTOR_B_SYSTEM


# ── Node functions ──────────────────────────────────────────────

def initialize_node(state: DebateState) -> dict:
    """Set up initial state."""
    return {
        "current_round": 1,
        "messages": [],
        "doctor_a_history": [],
        "doctor_b_history": [],
        "is_complete": False,
    }


def doctor_a_node(state: DebateState) -> dict:
    """Doctor A's turn: opening if round 1, rebuttal otherwise."""
    llm = create_llm()
    history = list(state["doctor_a_history"])
    round_num = state["current_round"]

    if round_num == 1:
        msg = run_doctor_a_opening(llm, state["question"], state["evidence"], history)
    else:
        opponent_msg = [m for m in state["messages"] if m["agent"] == "doctor_b"][-1]
        msg = run_rebuttal(llm, "doctor_a", DOCTOR_A_SYSTEM, opponent_msg, history, round_num)

    messages = list(state["messages"])
    messages.append(msg)
    return {"messages": messages, "doctor_a_history": history}


def doctor_b_node(state: DebateState) -> dict:
    """Doctor B's turn: opening if round 1, rebuttal otherwise."""
    llm = create_llm()
    history = list(state["doctor_b_history"])
    round_num = state["current_round"]

    if round_num == 1:
        msg = run_doctor_b_opening(llm, state["question"], state["evidence"], history)
    else:
        opponent_msg = [m for m in state["messages"] if m["agent"] == "doctor_a"][-1]
        msg = run_rebuttal(llm, "doctor_b", DOCTOR_B_SYSTEM, opponent_msg, history, round_num)

    messages = list(state["messages"])
    messages.append(msg)
    return {"messages": messages, "doctor_b_history": history}


def check_completion_node(state: DebateState) -> dict:
    """Increment round counter. Mark complete if max rounds reached."""
    next_round = state["current_round"] + 1
    is_complete = next_round > state["max_rounds"]
    return {"current_round": next_round, "is_complete": is_complete}


def should_continue(state: DebateState) -> str:
    """Routing function: continue debate or finalize."""
    if state["is_complete"]:
        return "finalize"
    return "doctor_a"


def finalize_node(state: DebateState) -> dict:
    """Build the final DebateTranscript stored in metadata."""
    messages = state["messages"]

    a_msgs = [m for m in messages if m["agent"] == "doctor_a"]
    b_msgs = [m for m in messages if m["agent"] == "doctor_b"]

    a_final = a_msgs[-1] if a_msgs else {"position": "maybe", "confidence": 0.5}
    b_final = b_msgs[-1] if b_msgs else {"position": "maybe", "confidence": 0.5}

    transcript = DebateTranscript(
        question=state["question"],
        evidence=[EvidenceItem(**e) if isinstance(e, dict) else e for e in state["evidence"]],
        messages=[AgentMessage(**m) if isinstance(m, dict) else m for m in messages],
        num_rounds=state["max_rounds"],
        doctor_a_final_position=a_final["position"],
        doctor_b_final_position=b_final["position"],
        doctor_a_final_confidence=a_final["confidence"],
        doctor_b_final_confidence=b_final["confidence"],
    )

    return {"metadata": {"transcript": transcript.model_dump()}}


# ── Graph construction ──────────────────────────────────────────

def build_debate_graph():
    """Construct and compile the LangGraph debate workflow."""
    graph = StateGraph(DebateState)

    graph.add_node("initialize", initialize_node)
    graph.add_node("doctor_a", doctor_a_node)
    graph.add_node("doctor_b", doctor_b_node)
    graph.add_node("check_completion", check_completion_node)
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("initialize")
    graph.add_edge("initialize", "doctor_a")
    graph.add_edge("doctor_a", "doctor_b")
    graph.add_edge("doctor_b", "check_completion")
    graph.add_conditional_edges("check_completion", should_continue, {
        "doctor_a": "doctor_a",
        "finalize": "finalize",
    })
    graph.add_edge("finalize", END)

    return graph.compile()


# ── Public API ──────────────────────────────────────────────────

def run_debate(
    question: str,
    evidence: List[dict],
    max_rounds: int = 2,
) -> DebateTranscript:
    """
    Run a full medical debate and return the transcript.

    This is the main entry point that Person 3's judge will call.

    Args:
        question: The biomedical question (yes/no/maybe).
        evidence: List of evidence dicts from the retrieval pipeline.
        max_rounds: Number of debate rounds (each round = A speaks + B speaks).

    Returns:
        DebateTranscript with all messages, positions, and confidences.
    """
    app = build_debate_graph()

    initial_state: DebateState = {
        "question": question,
        "evidence": evidence,
        "messages": [],
        "current_round": 1,
        "max_rounds": max_rounds,
        "doctor_a_history": [],
        "doctor_b_history": [],
        "is_complete": False,
        "metadata": {},
    }

    final_state = app.invoke(initial_state)

    # Reconstruct transcript from final state
    transcript_data = final_state.get("metadata", {}).get("transcript")
    if transcript_data:
        return DebateTranscript(**transcript_data)

    # Fallback: build from messages directly
    messages = final_state.get("messages", [])
    a_msgs = [m for m in messages if m["agent"] == "doctor_a"]
    b_msgs = [m for m in messages if m["agent"] == "doctor_b"]
    a_final = a_msgs[-1] if a_msgs else {"position": "maybe", "confidence": 0.5}
    b_final = b_msgs[-1] if b_msgs else {"position": "maybe", "confidence": 0.5}

    return DebateTranscript(
        question=question,
        evidence=[EvidenceItem(**e) if isinstance(e, dict) else e for e in evidence],
        messages=[AgentMessage(**m) if isinstance(m, dict) else m for m in messages],
        num_rounds=max_rounds,
        doctor_a_final_position=a_final["position"],
        doctor_b_final_position=b_final["position"],
        doctor_a_final_confidence=a_final["confidence"],
        doctor_b_final_confidence=b_final["confidence"],
    )
