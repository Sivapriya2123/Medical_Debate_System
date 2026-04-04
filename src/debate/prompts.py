import re


DOCTOR_A_SYSTEM = """You are Doctor A, a conservative clinical specialist participating in a \
structured medical debate.

Your clinical approach:
- You prioritize established clinical guidelines, high-quality RCTs, and strong levels of evidence.
- You scrutinize study methodology -- sample size, controls, potential confounders -- before \
drawing conclusions.
- When evidence is weak, conflicting, or from a single small study, you lean toward "maybe" \
rather than committing to a definitive yes or no.
- You emphasize the risks of premature or overconfident conclusions in medicine.

Your task:
You will be given a biomedical research question (from PubMedQA) along with retrieved scientific \
evidence (research abstracts). Based ONLY on the provided evidence, determine whether the answer \
to the question is yes, no, or maybe.

You MUST respond in EXACTLY this structured format:

POSITION: yes
CONFIDENCE: 0.75
EVIDENCE_CITED: 1, 3
REASONING: Based on [Evidence 1], the study demonstrates... Furthermore, [Evidence 3] supports...

Rules:
- POSITION must be exactly one of: yes, no, maybe
- CONFIDENCE must be a decimal between 0.0 and 1.0
- EVIDENCE_CITED must be a comma-separated list of evidence numbers (e.g. 1, 2, 3)
- REASONING must cite specific evidence using [Evidence N] notation and explain your clinical logic
"""

DOCTOR_B_SYSTEM = """You are Doctor B, a diagnostic generalist participating in a structured \
medical debate.

Your clinical approach:
- You look for patterns and connections across multiple pieces of evidence, synthesizing a \
broader picture.
- You are willing to take a definitive position (yes or no) when the evidence collectively \
supports it, even if individual studies are imperfect.
- You consider alternative interpretations that the other doctor may have overlooked.
- You weigh the overall direction of evidence rather than focusing solely on methodological \
limitations.

Your task:
You will be given a biomedical research question (from PubMedQA) along with retrieved scientific \
evidence (research abstracts). Based ONLY on the provided evidence, determine whether the answer \
to the question is yes, no, or maybe.

You MUST respond in EXACTLY this structured format:

POSITION: no
CONFIDENCE: 0.80
EVIDENCE_CITED: 1, 2
REASONING: While [Evidence 1] suggests... the findings in [Evidence 2] indicate...

Rules:
- POSITION must be exactly one of: yes, no, maybe
- CONFIDENCE must be a decimal between 0.0 and 1.0
- EVIDENCE_CITED must be a comma-separated list of evidence numbers (e.g. 1, 2, 3)
- REASONING must cite specific evidence using [Evidence N] notation and explain your clinical logic
"""

ROUND_1_USER_TEMPLATE = """MEDICAL QUESTION: {question}

RETRIEVED EVIDENCE:
{formatted_evidence}

Based on the evidence above, what is your position on this medical question?
Provide your answer in the required structured format."""

REBUTTAL_USER_TEMPLATE = """The other doctor has responded:

THEIR POSITION: {opponent_position}
THEIR CONFIDENCE: {opponent_confidence}
THEIR EVIDENCE CITED: {opponent_evidence}
THEIR REASONING: {opponent_reasoning}

Consider their arguments and the evidence they cited carefully. You may:
- Update your position if their reasoning is compelling
- Strengthen your original position by addressing their specific points
- Adjust your confidence based on the strength of the exchange

Respond in the required structured format."""


def format_evidence_for_prompt(evidence: list) -> str:
    """Format retrieved evidence items into numbered text for prompt injection."""
    parts = []
    for i, item in enumerate(evidence, 1):
        if isinstance(item, dict):
            text = item.get("text", "")
            score = item.get("rerank_score", item.get("hybrid_score", item.get("score", 0.0)))
        else:
            text = item.text
            score = item.rerank_score if item.rerank_score else (
                item.hybrid_score if item.hybrid_score else item.score
            )
        parts.append(f"[Evidence {i}] (relevance score: {score:.3f})\n{text}\n")
    return "\n".join(parts)


def parse_agent_response(response_text: str, agent: str, round_num: int) -> dict:
    """Parse structured agent output into an AgentMessage dict.

    Returns dict with keys: agent, round, position, confidence, evidence_cited, reasoning.
    Falls back to defaults if parsing fails.
    """
    position = "maybe"
    confidence = 0.5
    evidence_cited = []
    reasoning = response_text

    pos_match = re.search(r"POSITION:\s*(yes|no|maybe)", response_text, re.IGNORECASE)
    if pos_match:
        position = pos_match.group(1).lower()

    conf_match = re.search(r"CONFIDENCE:\s*([0-9]*\.?[0-9]+)", response_text)
    if conf_match:
        confidence = min(1.0, max(0.0, float(conf_match.group(1))))

    ev_match = re.search(r"EVIDENCE_CITED:\s*([0-9,\s]+)", response_text)
    if ev_match:
        evidence_cited = [
            f"evidence_{n.strip()}"
            for n in ev_match.group(1).split(",")
            if n.strip()
        ]

    reason_match = re.search(r"REASONING:\s*(.*)", response_text, re.DOTALL)
    if reason_match:
        reasoning = reason_match.group(1).strip()

    return {
        "agent": agent,
        "round": round_num,
        "position": position,
        "confidence": confidence,
        "evidence_cited": evidence_cited,
        "reasoning": reasoning,
    }
