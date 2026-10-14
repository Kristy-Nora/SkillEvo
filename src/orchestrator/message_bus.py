"""
Message bus — inter-agent message routing and trajectory recording.

Routes messages by type between Doctor, Patient, and Exam agents.
Records the complete trajectory for evaluation and skill extraction.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum


class MessageType(Enum):
    """Message types for inter-agent communication."""
    DOCTOR_INQUIRY = "doctor_inquiry"       # Doctor → Patient
    PATIENT_RESPONSE = "patient_response"   # Patient → Doctor
    EXAM_REQUEST = "exam_request"           # Doctor → Exam
    EXAM_RESULT = "exam_result"             # Exam → Doctor
    DOCTOR_THINK = "doctor_think"           # Internal: doctor's reasoning (trajectory only)
    DOCTOR_ACTION = "doctor_action"         # Internal: doctor's action decision (trajectory only)


@dataclass
class AgentMessage:
    """A message in the agent communication system."""
    msg_id: str
    from_agent: str
    to_agent: str
    msg_type: MessageType
    content: Dict[str, Any] = field(default_factory=dict)
    turn_idx: int = 0


class MessageBus:
    """Central message bus for agent communication and trajectory recording.

    Routes messages by type between agents. Records the full trajectory
    for offline evaluation and skill extraction.
    """

    def __init__(self):
        self._agents: Dict[str, any] = {}
        self._trajectory: List[Dict] = []

    def register(self, agent_id: str, agent):
        """Register an agent on the bus."""
        self._agents[agent_id] = agent

    def send(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Send a message to its target agent and return the response."""
        if message.to_agent not in self._agents:
            print(f"   [Bus] Unknown target agent: {message.to_agent}")
            return None

        target = self._agents[message.to_agent]
        response = target.receive(message)

        # Record the exchange in trajectory
        self.log_think(message)
        if response:
            self.log_think(response)

        return response

    def make_message(self, from_agent: str, to_agent: str,
                     msg_type: MessageType, content: dict,
                     turn_idx: int = 0) -> AgentMessage:
        """Factory method to create a properly ID'd message."""
        return AgentMessage(
            msg_id=f"{msg_type.value}_{turn_idx}",
            from_agent=from_agent,
            to_agent=to_agent,
            msg_type=msg_type,
            content=content,
            turn_idx=turn_idx,
        )

    def log_think(self, message: AgentMessage):
        """Record a message in the trajectory log (for evaluation)."""
        self._trajectory.append({
            "msg_id": message.msg_id,
            "msg_type": message.msg_type.value,
            "from_agent": message.from_agent,
            "to_agent": message.to_agent,
            "content": message.content,
            "turn_idx": message.turn_idx,
        })

    def get_full_trajectory(self) -> List[Dict]:
        """Return the complete message trajectory."""
        return list(self._trajectory)
