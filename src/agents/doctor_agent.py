"""
Doctor agent — the core clinical decision-making agent (ReAct mode).

Workflow (ReAct loop):
  1. Think: Analyze current evidence, update differential diagnoses, identify gaps
  2. Act: Question the patient, order exams, or give final diagnosis
  3. Observe: Receive patient answers or exam results, update evidence

Internal state maintained:
  - collected_evidence: All evidence gathered so far
  - differential_diagnoses: Current differential diagnosis list (with probabilities)
  - asked_questions: Record of questions asked
  - requested_exams: Record of exams ordered
  - think_log: Complete reasoning log (for trajectory and evaluation)
"""

import json
import re
from typing import Optional, List, Dict, Any

from src.agents.base_agent import BaseAgent


class DoctorAgent(BaseAgent):
    """Doctor agent with full diagnostic decision-making capability via ReAct loop."""

    def __init__(self, agent_id: str, message_bus):
        super().__init__(agent_id, message_bus)
        # Internal state
        self.collected_evidence: List[Dict] = []
        self.differential_diagnoses: List[Dict] = []
        self.asked_questions: List[str] = []
        self.requested_exams: List[str] = []
        self.failed_exams: List[str] = []
        self.think_log: List[Dict] = []
        self.last_think: Optional[Dict] = None
        # Per-turn skill injection
        self.injected_skills: str = ""
        self.injected_skill_ids: list = []
        self._injected_skill_data: list = []
        self.skill_enabled: bool = False
        self._turn_skill_cats: list = []

    def init_session(self, chief_complaint: str = None, vital_signs: dict = None):
        """Initialize a consultation session. Doctor initially only knows vital signs."""
        self.collected_evidence: List[Dict] = []
        self.differential_diagnoses: List[Dict] = []
        self.asked_questions: List[str] = []
        self.requested_exams: List[str] = []
        self.failed_exams: List[str] = []
        self.think_log: List[Dict] = []
        self.last_think: Optional[Dict] = None
        self.injected_skills: str = ""
        self.injected_skill_ids: list = []
        self._injected_skill_data: list = []
        self._turn_skill_cats: list = []

        if chief_complaint:
            self.collected_evidence.append({
                "source": "chief_complaint",
                "content": f"Chief complaint: {chief_complaint}",
            })
        if vital_signs and isinstance(vital_signs, dict):
            vs_text = ", ".join(f"{k}: {v}" for k, v in vital_signs.items())
            self.collected_evidence.append({
                "source": "vital_signs",
                "content": f"Vital signs: {vs_text}",
            })

    # ═══════════════════════════════════════════════════════════
    # Think Phase
    # ═══════════════════════════════════════════════════════════

    def think(self, turn_idx: int) -> Dict[str, Any]:
        """Reasoning phase: output structured analysis.

        Returns:
            {
                "evidence_summary": "Summary of current evidence",
                "differential_diagnoses": [{"disease": "...", "probability": 0.X, ...}],
                "information_gaps": ["gap1", "gap2"],
                "next_plan": {"action_type": "inquiry|exam_request", "detail": "..."}
            }
        """
        user_prompt = DOCTOR_THINK_PROMPT.format(
            collected_evidence=json.dumps(self.collected_evidence, ensure_ascii=False, indent=2),
            differential_diagnoses=json.dumps(self.differential_diagnoses, ensure_ascii=False, indent=2),
            asked_questions=json.dumps(self.asked_questions, ensure_ascii=False),
            requested_exams=json.dumps(self.requested_exams, ensure_ascii=False),
        )
        response = self._call_llm(
            system_prompt=self._build_system_prompt(),
            user_prompt=user_prompt,
            max_tokens=4096,
            temperature=0.6,
        )

        think_result = self._parse_think_response(response)
        self.last_think = think_result
        self.think_log.append({"turn": turn_idx, "think": think_result})

        # Record to trajectory
        think_msg = AgentMessage(
            msg_id=f"think_{turn_idx}",
            from_agent=self.agent_id,
            to_agent=self.agent_id,
            msg_type=MessageType.DOCTOR_THINK,
            content={"think_result": think_result, "raw_response": response},
            turn_idx=turn_idx,
        )
        self.message_bus.log_think(think_msg)

        return think_result

    # ═══════════════════════════════════════════════════════════
    # Act Phase
    # ═══════════════════════════════════════════════════════════

    def act(self, think_result: Dict[str, Any], turn_idx: int, max_turns: int) -> Dict[str, Any]:
        """Action phase: decide what to do based on the reasoning result.

        Adds progress-awareness pressure in later turns to push toward diagnosis.
        """
        # Progress awareness: let the agent know where it is in the consultation
        if turn_idx >= max_turns - 1:
            pressure = (
                f"\n*** FINAL TURN (turn {turn_idx}/{max_turns})! You MUST output final_diagnosis."
            )
        elif turn_idx >= max_turns - 2:
            pressure = (
                f"\n*** Only 2 turns left (turn {turn_idx}/{max_turns}). "
                f"If evidence is sufficient, give diagnosis now; do NOT order new exams."
            )
        else:
            pressure = ""

        user_prompt = DOCTOR_ACT_PROMPT.format(
            think_result=json.dumps(think_result, ensure_ascii=False, indent=2),
            turn_idx=turn_idx,
            max_turns=max_turns,
            asked_questions=json.dumps(self.asked_questions, ensure_ascii=False),
            requested_exams=json.dumps(self.requested_exams, ensure_ascii=False),
            failed_exams=json.dumps(self.failed_exams, ensure_ascii=False),
            turn_pressure=pressure,
        )
        response = self._call_llm(
            system_prompt=self._build_system_prompt(),
            user_prompt=user_prompt,
            max_tokens=2048,
            temperature=0.3,
        )

        action = self._parse_action_response(response)
        return action

    # ═══════════════════════════════════════════════════════════
    # Observe Phase
    # ═══════════════════════════════════════════════════════════

    def observe(self, observation: Dict[str, Any], turn_idx: int):
        """Observation phase: incorporate patient answers or exam results."""
        user_prompt = DOCTOR_OBSERVE_PROMPT.format(
            new_observation=json.dumps(observation, ensure_ascii=False, indent=2),
            current_evidence=json.dumps(self.collected_evidence, ensure_ascii=False, indent=2),
            current_differential=json.dumps(self.differential_diagnoses, ensure_ascii=False, indent=2),
        )
        response = self._call_llm(
            system_prompt=self._build_system_prompt(),
            user_prompt=user_prompt,
            max_tokens=1024,
            temperature=0.3,
        )

        update = self._parse_observe_response(response)

        # Update internal state
        new_ev = update.get("new_evidence")
        if isinstance(new_ev, list):
            self.collected_evidence.extend(new_ev)

        new_diff = update.get("updated_differential")
        if isinstance(new_diff, list):
            self.differential_diagnoses = new_diff

    # ═══════════════════════════════════════════════════════════
    # Internal Helpers
    # ═══════════════════════════════════════════════════════════

    def _build_system_prompt(self) -> str:
        """Build system prompt with current state and injected skills."""
        base = DOCTOR_SYSTEM_PROMPT.format(
            collected_evidence=json.dumps(self.collected_evidence, ensure_ascii=False, indent=2),
            differential_diagnoses=json.dumps(self.differential_diagnoses, ensure_ascii=False, indent=2),
        )
        # Inject per-turn matched skills
        if self.skill_enabled and self.injected_skills:
            base += "\n\n" + self.injected_skills
            base += "\n\nPlease refer to the above skill guidance for this turn's reasoning and action."
        return base

    def _parse_think_response(self, response: str) -> Dict[str, Any]:
        """Extract structured JSON from LLM output (handles mixed reasoning + JSON)."""
        json_str = self._extract_json(response)
        if json_str:
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
        return {"raw_think": response, "parse_error": True}

    def _parse_action_response(self, response: str) -> Dict[str, Any]:
        json_str = self._extract_json(response)
        if json_str:
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
        return {"action_type": "unknown", "raw_response": response}

    def _parse_observe_response(self, response: str) -> Dict[str, Any]:
        json_str = self._extract_json(response)
        if json_str:
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
        return {"new_evidence": [], "raw_response": response}

    @staticmethod
    def _extract_json(response: str) -> Optional[str]:
        """Extract valid JSON from mixed reasoning text using brace counting."""
        if not response:
            return None
        # Try ```json code block first
        json_block = re.search(r'```json\s*\n(.*?)\n```', response, re.DOTALL)
        if json_block:
            return json_block.group(1).strip()
        # Fallback: find the last complete JSON object via brace counting
        last_brace = response.rfind('}')
        if last_brace > 0:
            depth = 0
            start = -1
            for i in range(last_brace, -1, -1):
                if response[i] == '}':
                    depth += 1
                elif response[i] == '{':
                    depth -= 1
                    if depth == 0:
                        start = i
                        break
            if start >= 0:
                candidate = response[start:last_brace + 1]
                try:
                    json.loads(candidate)
                    return candidate
                except json.JSONDecodeError:
                    pass
        return None


# ═══════════════════════════════════════════════════════════
# Doctor Prompts
# ═══════════════════════════════════════════════════════════

DOCTOR_SYSTEM_PROMPT = """You are an attending physician in the respiratory department.
You are conducting a consultation with a patient. Your goal is to arrive at the correct
diagnosis through systematic history-taking, physical examination, and appropriate
diagnostic testing.

## Current Evidence
{collected_evidence}

## Current Differential Diagnoses
{differential_diagnoses}

## Instructions
1. Think step by step about what you know and what you still need to find out
2. Ask targeted questions to fill information gaps
3. Order appropriate tests when indicated
4. Synthesize all evidence before giving a final diagnosis
5. Always consider life-threatening conditions first
6. Maintain diagnostic openness — don't anchor on your first impression"""

DOCTOR_THINK_PROMPT = """Based on the current state, analyze the diagnostic situation.

## Collected Evidence
{collected_evidence}

## Current Differential Diagnoses
{differential_diagnoses}

## Questions Asked So Far
{asked_questions}

## Exams Ordered So Far
{requested_exams}

## Task
Output a JSON object with your analysis:
{{
  "evidence_summary": "Concise summary of key findings so far",
  "differential_diagnoses": [
    {{"disease": "Disease name", "probability": 0.X, "support": ["evidence"], "against": ["evidence"]}}
  ],
  "information_gaps": ["gap1", "gap2"],
  "next_plan": {{"action_type": "inquiry|exam_request", "detail": "Specific plan"}}
}}

Output ONLY the JSON object."""

DOCTOR_ACT_PROMPT = """Based on your analysis, decide what action to take.

## Think Result
{think_result}

## State
- Turn: {turn_idx}/{max_turns}
- Questions asked: {asked_questions}
- Exams ordered: {requested_exams}
- Failed exams (confirmed unavailable): {failed_exams}
{turn_pressure}

## Task
Output a JSON object with your action:

For inquiry:
{{"action_type": "inquiry", "content": {{"question": "Your question to the patient"}}}}

For exam request:
{{"action_type": "exam_request", "content": {{"exam_name": "Name of exam", "reason": "Why needed"}}}}

For final diagnosis:
{{"action_type": "final_diagnosis", "content": {{
  "primary_diagnoses": ["Diagnosis 1", "Diagnosis 2"],
  "evidence_chain": ["Key evidence 1", "Key evidence 2"],
  "uncertainty": "Any remaining uncertainty"
}}}}

Output ONLY the JSON object."""

DOCTOR_OBSERVE_PROMPT = """Incorporate the new observation into your understanding.

## New Observation
{new_observation}

## Current Evidence
{current_evidence}

## Current Differential Diagnoses
{current_differential}

## Task
Output a JSON object:
{{
  "new_evidence": [{{"source": "...", "content": "..."}}],
  "updated_differential": [{{"disease": "...", "probability": 0.X, "support": [...], "against": [...]}}]
}}

Output ONLY the JSON object."""


# Lightweight imports at bottom for circular dependency avoidance
from src.orchestrator.message_bus import AgentMessage, MessageType
