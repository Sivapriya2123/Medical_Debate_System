from typing import List, Dict, Any, Literal
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class EvidenceItem(BaseModel):
    """A single piece of retrieved evidence passed to agents."""
    id: str
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    score: float = 0.0
    hybrid_score: float = 0.0
    rerank_score: float = 0.0
    source: str = ""


class AgentMessage(BaseModel):
    """A single message in the debate transcript."""
    agent: Literal["doctor_a", "doctor_b"]
    round: int
    position: Literal["yes", "no", "maybe"]
    reasoning: str
    evidence_cited: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class DebateTranscript(BaseModel):
    """Complete debate output -- the contract with Person 3's judge."""
    question: str
    evidence: List[EvidenceItem]
    messages: List[AgentMessage]
    num_rounds: int
    doctor_a_final_position: Literal["yes", "no", "maybe"]
    doctor_b_final_position: Literal["yes", "no", "maybe"]
    doctor_a_final_confidence: float
    doctor_b_final_confidence: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DebateState(TypedDict):
    """LangGraph state object threaded through the graph."""
    question: str
    evidence: List[dict]
    messages: List[dict]
    current_round: int
    max_rounds: int
    doctor_a_history: List[dict]
    doctor_b_history: List[dict]
    is_complete: bool
    metadata: Dict[str, Any]
