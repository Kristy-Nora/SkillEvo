"""
Patient agent — standardized simulated patient.

Core rules:
  1. Only holds patient_visible data (chief complaint, history, symptoms, vitals)
  2. No external knowledge access, no medical reasoning
  3. Only answers based on the doctor's specific question, looking up from known data
  4. Does not volunteer information not asked about
  5. If asked about information not in scope, responds "I don't know" or "No"
"""

import json
from typing import Optional

from src.agents.base_agent import BaseAgent
from src.orchestrator.message_bus import AgentMessage, MessageType


class PatientAgent(BaseAgent):
    """Patient agent that simulates a real patient — knows only personal health data,
    does not analyze, reason about, or suggest medical conclusions."""

    def __init__(self, agent_id: str, message_bus):
        super().__init__(agent_id, message_bus)
        self._patient_data: dict = {}
        self._disclosed_history: list = []  # Track what has been asked (prevent re-disclosure)

    def load_patient_data(self, patient_data: dict):
        """Inject patient data at session initialization."""
        self._patient_data = patient_data

    def generate_opening(self) -> str:
        """Generate the patient's opening statement — describes their situation naturally."""
        system_prompt = PATIENT_SYSTEM_PROMPT.format(
            patient_data=json.dumps(self._patient_data, ensure_ascii=False, indent=2)
        )
        user_prompt = PATIENT_OPENING_PROMPT.format(
            patient_data=json.dumps(self._patient_data, ensure_ascii=False, indent=2)
        )
        opening = self._call_llm(system_prompt, user_prompt, max_tokens=512, temperature=0.5)
        self._disclosed_history.append({"question": "[opening statement]", "answer": opening})
        return opening

    def receive(self, message: AgentMessage) -> Optional[AgentMessage]:
        if message.msg_type != MessageType.DOCTOR_INQUIRY:
            return None

        doctor_question = message.content.get("question", "")
        answer = self._generate_answer(doctor_question)

        return AgentMessage(
            msg_id=message.msg_id + "_r",
            from_agent=self.agent_id,
            to_agent=message.from_agent,
            msg_type=MessageType.PATIENT_RESPONSE,
            content={"answer": answer, "in_response_to": doctor_question},
            turn_idx=message.turn_idx,
        )

    def _generate_answer(self, question: str) -> str:
        """Use LLM to match the question against patient data and generate a natural answer."""
        system_prompt = PATIENT_SYSTEM_PROMPT.format(
            patient_data=json.dumps(self._patient_data, ensure_ascii=False, indent=2)
        )
        user_prompt = PATIENT_ANSWER_PROMPT.format(
            doctor_question=question,
            patient_data=json.dumps(self._patient_data, ensure_ascii=False, indent=2),
            disclosed_so_far=json.dumps(self._disclosed_history, ensure_ascii=False),
        )
        answer = self._call_llm(system_prompt, user_prompt, max_tokens=2048, temperature=0.3)
        self._disclosed_history.append({"question": question, "answer": answer})
        return answer


# ═══════════════════════════════════════════════════════════
# Patient Prompts
# ═══════════════════════════════════════════════════════════

PATIENT_SYSTEM_PROMPT = """You are a standardized patient. Your role is to simulate a real patient.

## Your Patient Data
{patient_data}

## Critical Rules
1. You ONLY know the information listed in your patient data above
2. Do NOT make up any information not in your data
3. Do NOT use medical knowledge to interpret or analyze your symptoms
4. When the doctor asks a question, find the relevant information in your data and answer naturally
5. If the doctor asks about something not in your data, say "I don't know" or "I don't remember"
6. You are here to be examined, not to diagnose yourself
7. Answer in the first person, as a real patient would"""

PATIENT_OPENING_PROMPT = """You are meeting a doctor for the first time. Based on your patient data,
describe what brought you here today. Speak naturally, as a real patient would.

## Your Patient Data
{patient_data}

## Instructions
- Describe your main complaint in your own words
- Include when it started and what it feels like
- Mention any obvious relevant history if you think it's important
- Keep it to 2-4 sentences
- Do NOT use medical terminology — speak like a regular person"""

PATIENT_ANSWER_PROMPT = """The doctor asked you: "{doctor_question}"

## Your Patient Data (everything you know about yourself)
{patient_data}

## What You've Already Told the Doctor
{disclosed_so_far}

## Instructions
1. Find information in your patient data that answers the doctor's question
2. Answer naturally, in the first person
3. If the information is NOT in your patient data, say "I don't know" or "No"
4. Do NOT repeat information you've already disclosed unless the doctor asks for clarification
5. Keep your answer concise and relevant"""
