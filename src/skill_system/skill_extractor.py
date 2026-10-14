"""
Skill extraction from clinical consultation trajectories.

Extracts reusable diagnostic skills from multi-turn ReAct consultation
trajectories. Skills are extracted at two granularities:
  1. Full-trajectory extraction: 2-4 skills from the complete 8-turn trajectory
  2. Per-turn extraction: fine-grained decision skills from high-scoring turns

Skills are categorized into 4 functional types with 3 abstraction levels:
  - L1 Disease-specific: single-disease experience
  - L2 Scenario-general: cross-disease patterns
  - L3 Universal principles: cross-department clinical reasoning heuristics
"""

import json
import os
import re
from typing import Optional, List, Dict


# ═══════════════════════════════════════════════════════════
# Skill Extraction Prompt
# ═══════════════════════════════════════════════════════════

SKILL_EXTRACTION_PROMPT = """You are a clinical medical educator. Extract reusable,
transferable diagnostic skills from the following formatted consultation trajectory.

## Formatted Trajectory
{trajectory_text}

## Gold Standard
{gold_standard}

## Skill Categories (4 core categories, at least 1 per category)

### 1. patient_inquiry
How to question patients given diagnostic information gaps, what to ask
and in what order, how to follow up based on patient answers.

Template: "When [information gap exists] → Use [specific questioning approach] → Based on response [follow up / confirm / adjust direction]"

### 2. diagnostic_reasoning
How to form diagnostic hypotheses, prioritize differentials, and dynamically
adjust direction based on new evidence.

Template: "When [clinical state / evidence pattern] → Use [hypothesis generation / evidence weighting strategy] → Based on new evidence [raise / lower / retain / exclude diagnoses]"

### 3. exam_execution
When to order which exams (labs, imaging, monitoring), how to sequence them,
and how to act on results.

Template: "When needing to confirm/distinguish [diagnostic question] → Select and order [exams by urgency / diagnostic value / invasiveness] → Based on results [confirm / exclude / add exams / adjust direction]"

### 4. diagnosis_conclusion
How to synthesize history, symptoms, signs and exam results into a diagnosis,
expressing uncertainty and explaining why alternatives are excluded or retained.

Template: "When [enough evidence obtained] → Synthesize [key positive + key negative evidence + consistency] → Output [most likely diagnosis / alternatives / uncertainty] → Explain [why others excluded or retained]"

## Abstraction Levels

- L1: Disease-specific skills (specific disease, exam, or indicator level)
- L2: Syndrome/scenario-level skills (applicable to a class of diseases/scenarios)
- L3: Universal principles (cross-department, applicable to almost all cases)

## Extraction Principles

- Prioritize extracting from high-scoring behaviors that contributed to diagnostic accuracy
- For low-scoring behaviors, only extract if the error cause is clear, as correction skills
- L2 + L3 should be >= 60% of total extracted skills
- Extract at least 1-2 L3 skills per trajectory
- At least 1 per category (but do not fabricate if the trajectory has no relevant behavior)
- Each skill must specify abstraction_level
- Skills should describe diagnostic decision rules, not just re-state case facts
- Do not include specific patient names, disease names, or single exam results as universal skills
- Merge similar skills to avoid duplicates

## Skill Format
{{
  "skill_id": "SKILL_XXX",
  "category": "patient_inquiry|diagnostic_reasoning|exam_execution|diagnosis_conclusion",
  "name": "Skill name (general, concise)",
  "abstraction_level": "L1|L2|L3",
  "trigger": "What clinical state, information gap, or evidence pattern triggers this",
  "action": "What specific action to take",
  "decision_rule": "How to choose the next step based on conditions, answers, or results",
  "effect": "How this improves diagnostic accuracy, efficiency, or safety",
  "pitfall": "Risk if not done or done incorrectly",
  "generalizable_to": ["Scenario A", "Scenario B"],
  "source_turns": ["TURN_XX"],
  "score_basis": "Relationship between this skill and the evaluation score"
}}

{extraction_guidance}
Output ONLY a valid JSON array. No other text."""


def extract_skills(
    trajectory_path: str,
    case_path: str = None,
    data_root: str = "data",
    llm_model: str = None,
    llm_endpoint: str = None,
    llm_api_key: str = None,
    skill_mode: str = "mixed",
    weak_dimensions: list = None,
    eval_score: float = None,
    trajectory_data: dict = None,
) -> List[dict]:
    """Extract reusable diagnostic skills from a consultation trajectory.

    After extraction, automatically computes initial confidence via Formula 1.

    Args:
        trajectory_path: Path to trajectory JSON file
        case_path: Path to case JSON (for gold standard)
        data_root: Data root directory
        llm_model: LLM model identifier
        llm_endpoint: OpenAI-compatible API endpoint
        llm_api_key: API key
        skill_mode: "success" / "lesson" / "mixed"
        weak_dimensions: Weak dimensions from evaluation (for targeted extraction)
        eval_score: Weighted evaluation score E(tau), for Formula 1
        trajectory_data: Pre-loaded trajectory dict (avoids re-reading file)

    Returns:
        List of extracted skill dicts
    """
    from openai import OpenAI

    # Load trajectory
    if trajectory_data is not None:
        trajectory = trajectory_data
    else:
        with open(trajectory_path, "r", encoding="utf-8") as f:
            trajectory = json.load(f)

    formatted = _format_for_skill_extraction(trajectory)

    # Load gold standard
    gold_standard = "Not available"
    if case_path:
        try:
            full_path = os.path.join(data_root, case_path)
            if not full_path.endswith(".json"):
                full_path += ".json"
            with open(full_path, "r", encoding="utf-8") as f:
                case_data = json.load(f)
            eval_hidden = case_data.get("eval_hidden", {})
            gold_standard = json.dumps(eval_hidden, ensure_ascii=False, indent=2)
        except Exception:
            pass

    # Build extraction guidance
    guidance = _build_extraction_guidance(skill_mode, weak_dimensions)

    # Call LLM
    client = OpenAI(base_url=llm_endpoint, api_key=llm_api_key)
    prompt = SKILL_EXTRACTION_PROMPT.format(
        trajectory_text=formatted,
        gold_standard=gold_standard,
        extraction_guidance=guidance,
    )
    resp = client.chat.completions.create(
        model=llm_model,
        messages=[
            {"role": "system", "content": "You are a clinical medical educator. Output ONLY a JSON array."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=4096,
        temperature=0.3,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    content = resp.choices[0].message.content or "[]"

    # Parse JSON
    try:
        match = re.search(r'\[.*\]', content, re.DOTALL)
        if match:
            skills = json.loads(match.group(0))

            # Validate abstraction level distribution
            l2_l3 = sum(1 for s in skills if s.get("abstraction_level") in ("L2", "L3"))
            total = len(skills)
            ratio = l2_l3 / total if total > 0 else 0
            level_counts = {
                "L1": sum(1 for s in skills if s.get("abstraction_level") == "L1"),
                "L2": sum(1 for s in skills if s.get("abstraction_level") == "L2"),
                "L3": sum(1 for s in skills if s.get("abstraction_level") == "L3"),
            }
            print(f"Extracted {len(skills)} skills (L3={level_counts['L3']} "
                  f"L2={level_counts['L2']} L1={level_counts['L1']})")
            if ratio < 0.5:
                print(f"WARNING: L2+L3 ratio only {ratio:.0%} (target >= 60%)")

            # Fill missing fields
            for s in skills:
                if "generalizable_to" not in s:
                    s["generalizable_to"] = []

            # Compute initial confidence via Formula 1
            if eval_score is not None:
                from src.skill_system.skill_math import annotate_skills_with_confidence
                skills = annotate_skills_with_confidence(skills, eval_score, skill_mode)
                avg_conf = sum(s.get("_score", 50) for s in skills) / len(skills)
                print(f"Formula 1: initial confidence annotated, avg = {avg_conf:.1f}")

            return skills
    except json.JSONDecodeError:
        pass

    print("WARNING: Skill extraction failed, returning empty list")
    return []


# ═══════════════════════════════════════════════════════════
# Internal Helpers
# ═══════════════════════════════════════════════════════════

def _format_for_skill_extraction(trajectory: dict) -> str:
    """Format a trajectory JSON into a readable clinical narrative for skill extraction."""
    lines = []
    msgs = trajectory.get("trajectory", [])

    for msg in msgs:
        tp = msg.get("msg_type", "")
        content = msg.get("content", {})
        turn = msg.get("turn_idx", "?")

        if tp == "patient_response":
            answer = content.get("answer", "")
            lines.append(f"[T{turn}] Patient: {answer[:300]}")

        elif tp == "doctor_inquiry":
            question = content.get("question", "")
            lines.append(f"[T{turn}] Doctor asks: {question[:200]}")

        elif tp == "doctor_think":
            think = content.get("think_result", {})
            if isinstance(think, dict):
                evidence = think.get("evidence_summary", "")
                diffs = think.get("differential_diagnoses", [])
                gaps = think.get("information_gaps", [])
                plan = think.get("next_plan", {})

                diff_text = ""
                if diffs:
                    items = []
                    for d in diffs[:3]:
                        prob = d.get("probability", "?")
                        disease = d.get("disease", "?")
                        items.append(f"{disease}({prob})")
                    diff_text = f" | Differential: {', '.join(items)}"

                gap_text = ""
                if gaps:
                    gap_text = f" | Gaps: {'; '.join(gaps[:2])}"

                plan_text = ""
                if plan:
                    plan_text = f" | Plan: {plan.get('action_type','?')}"

                lines.append(f"[T{turn}] Doctor thinks: evidence={evidence[:80]}{diff_text}{gap_text}{plan_text}")

        elif tp == "doctor_action":
            action = content.get("action", {})
            atype = action.get("action_type", "")
            if atype == "final_diagnosis":
                diag_list = action.get("content", {}).get("primary_diagnoses",
                            [action.get("content", {}).get("primary_diagnosis", "")])
                diag = "; ".join(str(d) for d in diag_list if d) or "Not given"
                lines.append(f"[T{turn}] FINAL DIAGNOSIS: {diag}")
            elif atype == "inquiry":
                q = action.get("content", {}).get("question", "")
                lines.append(f"[T{turn}] Decided to ask: {q[:150]}")
            elif atype == "exam_request":
                exam = action.get("content", {}).get("exam_name", "")
                reason = action.get("content", {}).get("reason", "")
                lines.append(f"[T{turn}] Decided to order: {exam} (reason: {reason[:100]})")

        elif tp == "exam_request":
            exam_name = content.get("exam_name", "")
            lines.append(f"[T{turn}] Exam requested: {exam_name}")

        elif tp == "exam_result":
            report = content.get("report", {})
            error = content.get("error", "")
            if error:
                lines.append(f"[T{turn}] Exam result: ERROR - {error[:150]}")
            else:
                summary = (report.get("structured_result") or report.get("original_report") or "")[:300]
                lines.append(f"[T{turn}] Exam result: {summary}")

    return "\n".join(lines)


def _build_extraction_guidance(mode: str, weak_dims: list) -> str:
    """Build extraction guidance text based on training mode and weak dimensions."""
    base = ("IMPORTANT: Each skill MUST specify abstraction_level (L1/L2/L3). "
            "L2+L3 combined must be >= 60%. "
            "Prioritize cross-disease reusable patterns. "
            "Output ONLY a JSON array, no other text.")

    if mode == "success":
        guidance = (
            "This consultation performed well (correct diagnosis). "
            "Focus on extracting what the doctor did RIGHT and WHY — "
            "which key decision steps ensured diagnostic accuracy. "
            "Promote these as successful experience patterns. "
        ) + base
    elif mode == "lesson":
        guidance = (
            "This consultation had deficiencies (incorrect or incomplete diagnosis). "
            "Focus on extracting LESSONS FROM FAILURE — "
            "where did the doctor go wrong, what error led to diagnostic deviation, "
            "and what the CORRECT diagnostic pathway should be. "
            "Failure lessons are often more valuable than success patterns. "
        )
        if weak_dims:
            dim_names = {
                "consultation_quality": "history-taking",
                "exam_selection_quality": "exam selection",
                "result_interpretation": "result interpretation",
                "differential_diagnosis": "differential diagnosis",
                "diagnosis_quality": "diagnostic accuracy",
            }
            names = [dim_names.get(d, d) for d in weak_dims]
            guidance += f" Focus especially on these weak dimensions: {', '.join(names)}."
        guidance += " " + base
    else:
        guidance = "Extract from both successes and failures. Failure experience is often more valuable. " + base

    return guidance
