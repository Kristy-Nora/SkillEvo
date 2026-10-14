"""
ReAct loop engine — drives the Think → Act → Route → Observe cycle.

Each turn:
  1. Per-turn skill injection (if enabled)
  2. Doctor thinks → produces structured analysis
  3. Doctor acts → inquiry / exam_request / final_diagnosis
  4. Message routed to Patient or Exam agent
  5. Doctor observes → updates evidence and differential
  6. Check termination condition
"""

import json
from typing import Dict, Any

from src.agents.doctor_agent import DoctorAgent
from src.agents.patient_agent import PatientAgent
from src.orchestrator.message_bus import MessageBus, MessageType, AgentMessage


class DoctorReActLoop:
    """ReAct loop engine: drives Think → Act → Observe for each turn."""

    def __init__(
        self, doctor: DoctorAgent, patient: PatientAgent, exam,
        bus: MessageBus, permission, max_turns: int = 8,
    ):
        self.doctor = doctor
        self.patient = patient
        self.exam = exam
        self.bus = bus
        self.permission = permission
        self.max_turns = max_turns

    def run(self) -> Dict[str, Any]:
        """Execute the full ReAct consultation loop.

        Returns:
            {
                "final_diagnosis": {...},
                "total_turns": int,
                "termination_reason": str,
                "trajectory": [...]
            }
        """
        for turn in range(1, self.max_turns + 1):
            # Per-turn skill injection
            if getattr(self.doctor, 'skill_enabled', False):
                inject_skills_per_turn(self.doctor, turn)

            response = None
            think_result = self.doctor.think(turn)
            action = self.doctor.act(think_result, turn, self.max_turns)
            action_type = action.get("action_type", "")

            # Log action to trajectory
            action_msg = AgentMessage(
                msg_id=f"action_{turn}", from_agent=self.doctor.agent_id,
                to_agent=self.doctor.agent_id, msg_type=MessageType.DOCTOR_ACTION,
                content={
                    "action": action,
                    "injected_skill_ids": getattr(self.doctor, 'injected_skill_ids', []),
                },
                turn_idx=turn,
            )
            self.bus.log_think(action_msg)

            print(f"  [Turn {turn}/{self.max_turns}] {action_type}", flush=True)

            if action_type == "final_diagnosis":
                return self._finish(turn, action, "diagnosis_given")

            elif action_type == "inquiry":
                question = action.get("content", {}).get("question", "")
                if question.strip():
                    self.doctor.asked_questions.append(question)
                msg = self.bus.make_message(
                    from_agent=self.doctor.agent_id,
                    to_agent=self.patient.agent_id,
                    msg_type=MessageType.DOCTOR_INQUIRY,
                    content={"question": question}, turn_idx=turn,
                )
                response = self.bus.send(msg)
                if response:
                    observation = {
                        "type": "patient_response",
                        "question": question,
                        "answer": response.content.get("answer", ""),
                    }

            elif action_type == "exam_request":
                exam_name = action.get("content", {}).get("exam_name", "")
                reason = action.get("content", {}).get("reason", "")
                if exam_name.strip():
                    self.doctor.requested_exams.append(exam_name)
                msg = self.bus.make_message(
                    from_agent=self.doctor.agent_id,
                    to_agent=self.exam.agent_id,
                    msg_type=MessageType.EXAM_REQUEST,
                    content={"exam_name": exam_name, "reason": reason},
                    turn_idx=turn,
                )
                response = self.bus.send(msg)
                if response:
                    if "error" in response.content:
                        self.doctor.failed_exams.append(exam_name)
                        observation = {
                            "type": "exam_result", "exam_name": exam_name,
                            "error": response.content.get("error", ""),
                        }
                    else:
                        self.permission.grant_exam_to_doctor(exam_name)
                        report = response.content.get("report", {})
                        observation = {
                            "type": "exam_result",
                            "exam_name": exam_name,
                            "report": report,
                        }

            else:
                continue

            if response:
                self.doctor.observe(observation, turn)

        # Max turns reached — force diagnosis
        print(f"\n  WARNING: Max turns ({self.max_turns}) reached, forcing diagnosis...")
        return self._force_diagnosis(self.max_turns)

    def _force_diagnosis(self, max_turns: int) -> Dict[str, Any]:
        """Force a final diagnosis when max turns are exhausted."""
        think_result = self.doctor.think(max_turns)
        final_action = self.doctor.act(think_result, max_turns, max_turns)
        if final_action.get("action_type") == "final_diagnosis":
            return self._finish(max_turns, final_action, "max_turns_reached")

        # If LLM still won't give diagnosis, use a dedicated forced-diagnosis prompt
        from openai import OpenAI
        cfg = self.doctor.llm_config
        client = OpenAI(base_url=cfg.get("endpoint", ""), api_key=cfg.get("api_key", ""))
        prompt = MAKE_DIAGNOSIS_PROMPT.format(
            evidence=json.dumps(self.doctor.collected_evidence, ensure_ascii=False, indent=2)[:3000],
            diffs=json.dumps(self.doctor.differential_diagnoses, ensure_ascii=False, indent=2)[:1000],
        )
        resp = client.chat.completions.create(
            model=cfg.get("model", ""),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512, temperature=0.2,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        content = resp.choices[0].message.content or "{}"
        diag = self._extract_json_dict(content)
        final_action = {
            "action_type": "final_diagnosis",
            "content": {
                "primary_diagnoses": diag.get("primary_diagnoses", ["Unable to determine"]),
                "evidence_chain": diag.get("evidence_chain",
                    [str(e) for e in self.doctor.collected_evidence[-3:]]),
                "uncertainty": diag.get("uncertainty", "Max turns reached"),
            },
        }
        return self._finish(max_turns, final_action, "max_turns_reached")

    def _finish(self, turns: int, final_action: dict, reason: str) -> Dict[str, Any]:
        return {
            "final_diagnosis": final_action.get("content") if final_action else None,
            "total_turns": turns,
            "termination_reason": reason,
            "trajectory": self.bus.get_full_trajectory(),
        }

    @staticmethod
    def _extract_json_dict(text: str) -> dict:
        import re
        try:
            start = text.find('{')
            if start >= 0:
                depth = 0
                for i in range(start, len(text)):
                    if text[i] == '{':
                        depth += 1
                    elif text[i] == '}':
                        depth -= 1
                        if depth == 0:
                            return json.loads(text[start:i + 1])
        except Exception:
            pass
        return {}


def inject_skills_per_turn(doctor_agent, turn_idx: int):
    """Per-turn skill injection: retrieve relevant skills and inject into doctor's prompt.

    Steps:
      1. LLM selects which categories are needed this turn
      2. For each selected category, retrieve top-2 skills via bge-m3 embedding search
      3. Filter by cosine similarity (>= 0.50) and retrieval score (>= 0.25)
      4. Inject full skill texts into doctor's system prompt
    """
    from src.skill_system.skill_embedder import retrieve_skills_by_embedding

    # 1. LLM selects needed categories
    chosen_cats = _select_categories_by_state(doctor_agent, turn_idx)
    if not chosen_cats:
        return

    # 2. Build query from current evidence + differential diagnoses
    import json as _json
    evidence_text = _json.dumps(doctor_agent.collected_evidence[-5:], ensure_ascii=False)
    diff_text = _json.dumps(doctor_agent.differential_diagnoses[:3], ensure_ascii=False)
    query = f"Current evidence:{evidence_text[:300]}\nDifferential:{diff_text[:200]}"

    # 3. Semantic retrieval via bge-m3
    try:
        scored = retrieve_skills_by_embedding(query, top_k=30)
    except Exception:
        scored = []

    if not scored:
        return

    # 4. Filter & inject top-2 per category
    injected = []
    for cat in chosen_cats:
        cat_matched = 0
        for skill, score in scored:
            if skill.get("category") != cat:
                continue
            cos_sim = skill.get("_cosine_sim", 0)
            r_score = skill.get("_retrieval_score", 0)
            if cos_sim < 0.50 or r_score < 0.25:
                continue
            injected.append((skill, cat))
            cat_matched += 1
            if cat_matched >= 2:
                break

    if not injected:
        return

    # Format injection text
    skill_texts = []
    for skill, cat in injected:
        level = skill.get("abstraction_level", "L1")
        skill_texts.append(
            f"[{cat} | {level}] {skill.get('name', '?')}\n"
            f"  Trigger: {skill.get('trigger', '?')}\n"
            f"  Action: {skill.get('action', '?')}\n"
            f"  Effect: {skill.get('effect', '?')}\n"
            f"  Pitfall: {skill.get('pitfall', '?')}"
        )

    injection = "## Skill Library Guidance (matched this turn)\n\n" + "\n\n".join(skill_texts)
    doctor_agent.injected_skills = injection
    doctor_agent._turn_skill_cats = chosen_cats
    doctor_agent.injected_skill_ids = [s.get("skill_id", "") for s, _ in injected]
    doctor_agent._injected_skill_data = [s for s, _ in injected]


def _select_categories_by_state(doctor_agent, turn_idx: int = 1) -> list:
    """LLM selects which skill categories are needed for the current turn.

    4 categories:
      - patient_inquiry: need to question the patient
      - diagnostic_reasoning: need to decide next diagnostic direction
      - exam_execution: need to order / interpret exams
      - diagnosis_conclusion: ready to synthesize evidence into diagnosis

    Turn 7-8: diagnosis_conclusion is mandatory.
    """
    from openai import OpenAI
    import json as _json

    evidence_text = _json.dumps(doctor_agent.collected_evidence[-5:], ensure_ascii=False)
    diffs = doctor_agent.differential_diagnoses[:3]
    diff_text = _json.dumps(diffs, ensure_ascii=False)

    prompt = f"""You are a clinical decision assistant. Based on the doctor's current state,
determine which skill categories are needed this turn.

## Current State
- Turn: {turn_idx} / 8
- Evidence: {evidence_text[:400]}
- Differential: {diff_text[:300]}

## Categories
- patient_inquiry: need to question the patient
- diagnostic_reasoning: need to decide next diagnostic direction
- exam_execution: need to order or interpret exams
- diagnosis_conclusion: ready to synthesize and conclude

## Rules
1. Only select categories truly needed this turn
2. Turn 7-8: MUST include diagnosis_conclusion
3. Output as JSON array: ["patient_inquiry", "diagnostic_reasoning"]"""

    try:
        config = getattr(doctor_agent, 'llm_config', {})
        client = OpenAI(
            base_url=config.get("endpoint", ""),
            api_key=config.get("api_key", ""),
        )
        resp = client.chat.completions.create(
            model=config.get("model", ""),
            messages=[{"role": "system", "content": "Output ONLY a JSON array."},
                      {"role": "user", "content": prompt}],
            max_tokens=128, temperature=0.1,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        content = resp.choices[0].message.content or "[]"
        import re
        match = re.search(r'\[.*\]', content, re.DOTALL)
        if match:
            cats = _json.loads(match.group(0))
            valid = {"patient_inquiry", "diagnostic_reasoning",
                     "exam_execution", "diagnosis_conclusion"}
            chosen = [c for c in cats if c in valid]
            if chosen:
                return chosen
    except Exception:
        pass

    # Fallback: return all 4 categories
    return ["patient_inquiry", "diagnostic_reasoning",
            "exam_execution", "diagnosis_conclusion"]


MAKE_DIAGNOSIS_PROMPT = """You are an attending respiratory physician.
Consultation has reached the maximum turn limit. You MUST give a final diagnosis
based on existing evidence.

## Collected Evidence
{evidence}

## Current Differential Diagnoses
{diffs}

Output JSON:
{{
  "primary_diagnoses": ["Diagnosis 1", "Diagnosis 2"],
  "evidence_chain": ["Key evidence 1", "Key evidence 2"],
  "uncertainty": "Explanation of remaining uncertainty"
}}
Output ONLY JSON."""
