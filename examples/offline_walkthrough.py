"""Exercise the released bus and consultation loop with scripted, network-free stubs."""
import json
from pathlib import Path

from src.agents.doctor_agent import DoctorAgent
from src.orchestrator.message_bus import AgentMessage, MessageBus, MessageType
from src.orchestrator.react_loop import DoctorReActLoop
from src.skill_system.skill_math import compute_initial_confidence, compute_retrieval_score

ROOT = Path(__file__).resolve().parents[1]


class ScriptedDoctor(DoctorAgent):
    """Fixed interface fixture, with no learned reasoning or skill injection."""

    def receive(self, message):
        # The loop delivers observations directly; no inbound doctor routing is used.
        return None

    def think(self, turn_idx):
        result = {"evidence_summary": "Scripted interface check; not clinical reasoning."}
        self.message_bus.log_think(AgentMessage(
            msg_id=f"think_{turn_idx}", from_agent=self.agent_id,
            to_agent=self.agent_id, msg_type=MessageType.DOCTOR_THINK,
            content={"think_result": result}, turn_idx=turn_idx,
        ))
        return result

    def act(self, think_result, turn_idx, max_turns):
        actions = [
            {"action_type": "inquiry", "content": {"question": "When did the symptoms start relative to treatment?"}},
            {"action_type": "exam_request", "content": {"exam_name": "Example report", "reason": "Inspect the report interface."}},
            {"action_type": "final_diagnosis", "content": {
                "primary_diagnoses": ["Synthetic label A"],
                "evidence_chain": ["Scripted patient response", "Synthetic report R1"],
                "uncertainty": "Artificial label for control-flow demonstration only.",
            }},
        ]
        return actions[turn_idx - 1]

    def observe(self, observation, turn_idx):
        self.collected_evidence.append({"turn": turn_idx, "observation": observation})


class FixtureResponder:
    """Minimal patient/examination interface, not an implementation of either agent."""

    def __init__(self, agent_id, visible_data):
        self.agent_id = agent_id
        self.visible_data = visible_data

    def receive(self, message):
        if self.agent_id == "patient":
            message_type = MessageType.PATIENT_RESPONSE
            content = {"answer": self.visible_data["present_illness"]}
        else:
            message_type = MessageType.EXAM_RESULT
            content = {"report": self.visible_data["lab_tests"][0]}
        return AgentMessage(
            msg_id=message.msg_id + "_reply", from_agent=self.agent_id,
            to_agent=message.from_agent, msg_type=message_type,
            content=content, turn_idx=message.turn_idx,
        )


class FixturePermission:
    """Record a permission notification; this is not a security boundary."""

    def __init__(self):
        self.granted = []

    def grant_exam_to_doctor(self, exam_name):
        self.granted.append(exam_name)


def run_demo():
    case = json.loads((ROOT / "data/example_case.json").read_text(encoding="utf-8"))
    bus = MessageBus()
    doctor = ScriptedDoctor("doctor", bus)
    doctor.init_session(case["patient_visible"]["chief_complaint"])
    patient = FixtureResponder("patient", case["patient_visible"])
    exam = FixtureResponder("exam", case["exam_visible"])
    for agent in [doctor, patient, exam]:
        bus.register(agent.agent_id, agent)
    permission = FixturePermission()
    # eval_hidden is deliberately not given to any online participant.
    result = DoctorReActLoop(doctor, patient, exam, bus, permission, max_turns=3).run()
    skill = json.loads((ROOT / "skills/diagnosis_conclusion.json").read_text(encoding="utf-8"))[0]
    return {
        "mode": "scripted_offline_interface_demonstration",
        "network_calls": 0, "learned_skill_retrieval": False,
        "consultation": result,
        "displayed_skill": skill,
        "illustrative_scores": {
            "initial_confidence_E_0_8_L2": compute_initial_confidence(0.8, "L2"),
            "retrieval_cosine_0_75_confidence_80": compute_retrieval_score(0.75, 80),
            "note": "Chosen numeric inputs, not experimental measurements.",
        },
    }


if __name__ == "__main__":
    print(json.dumps(run_demo(), indent=2))
