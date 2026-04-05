"""LLM-based judge agent that synthesises a final answer from a debate transcript.

The judge reads the full debate (both doctors' positions, reasoning, evidence,
and confidence trajectories) and produces a justified yes/no/maybe verdict.
It is NOT a simple majority-vote — it weighs reasoning quality and evidence.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage

from src.debate.agents import create_llm
from src.debate.prompts import format_evidence_for_prompt
from src.Judge.models import JudgeVerdict, TrustScore

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI
    from src.debate.models import DebateTranscript

logger = logging.getLogger(__name__)

# ── System Prompt ───────────────────────────────────────────────

JUDGE_SYSTEM_PROMPT = """\
You are a senior medical expert serving as an impartial judge in a structured \
debate between two specialist doctors.

You will be given:
  1. The original clinical question.
  2. The retrieved evidence passages.
  3. The full debate transcript, including each doctor's position, confidence, \
and reasoning across all rounds.
  4. A trust score summarizing debate quality (agreement, reasoning consistency, \
confidence stability).

Your task is to synthesize the debate and produce a final, well-justified answer.

You MUST respond in EXACTLY this structured format:

FINAL_ANSWER: yes
EXPLANATION: <2-4 sentences citing specific evidence and doctor arguments that \
drove your decision>
DEBATE_SUMMARY: <2-3 sentence summary of how the debate unfolded>

CRITICAL RULES:
- When BOTH doctors agree on a position (yes or no), you MUST adopt their \
shared position. Two independent experts reaching the same conclusion is \
strong evidence. Do NOT override their consensus with "maybe".
- When the trust score is HIGH (>0.7) and doctors agree, commit to their answer.
- If they disagree, weigh the quality of cited evidence and the internal \
consistency of each doctor's reasoning. Pick the better-supported side.
- Return "maybe" ONLY as a last resort when doctors directly contradict each \
other AND neither side has convincing evidence.
- FINAL_ANSWER must be exactly one of: yes, no, maybe
"""

# ── Transcript Formatting ───────────────────────────────────────


def format_transcript_for_judge(transcript: DebateTranscript) -> str:
    """Build a human-readable prompt string from a DebateTranscript.

    Includes the question, numbered evidence, and each round's messages
    formatted with position, confidence, evidence cited, and reasoning.
    """
    parts = []

    # Question
    parts.append(f"MEDICAL QUESTION: {transcript.question}")
    parts.append("")

    # Evidence
    evidence_dicts = [
        e.model_dump() if hasattr(e, "model_dump") else e
        for e in transcript.evidence
    ]
    parts.append("RETRIEVED EVIDENCE:")
    parts.append(format_evidence_for_prompt(evidence_dicts))
    parts.append("")

    # Debate rounds
    parts.append("DEBATE TRANSCRIPT:")
    parts.append("-" * 40)

    for msg in transcript.messages:
        agent_label = "Doctor A" if msg.agent == "doctor_a" else "Doctor B"
        evidence_str = ", ".join(msg.evidence_cited) if msg.evidence_cited else "none"
        parts.append(
            f"[Round {msg.round} | {agent_label} | "
            f"Position: {msg.position} | Confidence: {msg.confidence:.2f}]"
        )
        parts.append(f"Evidence cited: {evidence_str}")
        parts.append(f"Reasoning: {msg.reasoning}")
        parts.append("-" * 40)

    # Final positions summary
    parts.append("")
    parts.append("FINAL POSITIONS:")
    parts.append(
        f"  Doctor A: {transcript.doctor_a_final_position} "
        f"(confidence: {transcript.doctor_a_final_confidence:.2f})"
    )
    parts.append(
        f"  Doctor B: {transcript.doctor_b_final_position} "
        f"(confidence: {transcript.doctor_b_final_confidence:.2f})"
    )

    return "\n".join(parts)


# ── Response Parsing ────────────────────────────────────────────


def parse_judge_response(response_text: str) -> dict:
    """Parse the structured judge response into components.

    Returns dict with keys: final_answer, explanation, debate_summary.
    Falls back to safe defaults if parsing fails.
    """
    final_answer = "maybe"
    explanation = response_text
    debate_summary = ""

    answer_match = re.search(
        r"FINAL_ANSWER:\s*(yes|no|maybe)", response_text, re.IGNORECASE
    )
    if answer_match:
        final_answer = answer_match.group(1).lower()

    explanation_match = re.search(
        r"EXPLANATION:\s*(.*?)(?=DEBATE_SUMMARY:|$)", response_text, re.DOTALL
    )
    if explanation_match:
        explanation = explanation_match.group(1).strip()

    summary_match = re.search(r"DEBATE_SUMMARY:\s*(.*)", response_text, re.DOTALL)
    if summary_match:
        debate_summary = summary_match.group(1).strip()

    return {
        "final_answer": final_answer,
        "explanation": explanation,
        "debate_summary": debate_summary,
    }


# ── Public API ──────────────────────────────────────────────────


def run_judge(
    transcript: DebateTranscript,
    trust: TrustScore,
    llm: ChatOpenAI | None = None,
) -> JudgeVerdict:
    """Run the judge agent on a completed debate transcript.

    Args:
        transcript: Full debate output from `run_debate()`.
        trust: Pre-computed trust score for this debate.
        llm: Optional LLM instance; creates a default one if not provided.

    Returns:
        A JudgeVerdict with the final answer, explanation, trust, and summary.
    """
    if llm is None:
        llm = create_llm()

    formatted = format_transcript_for_judge(transcript)

    # Append trust score so the judge is truly trust-aware
    trust_info = (
        f"\n\nTRUST SCORE: {trust.overall:.3f}\n"
        f"  Agent Agreement: {trust.agent_agreement:.2f}\n"
        f"  Reasoning Consistency: {trust.reasoning_consistency:.2f}\n"
        f"  Confidence Stability: {trust.confidence_stability:.2f}\n"
    )
    formatted += trust_info

    messages = [
        SystemMessage(content=JUDGE_SYSTEM_PROMPT),
        HumanMessage(content=formatted),
    ]

    logger.info("Invoking judge LLM...")
    response = llm.invoke(messages)
    parsed = parse_judge_response(response.content)

    return JudgeVerdict(
        question=transcript.question,
        final_answer=parsed["final_answer"],
        explanation=parsed["explanation"],
        trust=trust,
        debate_summary=parsed["debate_summary"],
    )
