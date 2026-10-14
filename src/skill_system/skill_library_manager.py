"""
Skill library manager — CRUD, merge, dedup, and refinement of the diagnostic skill library.

Core functionality:
  - Multi-file skill storage (one JSON file per category)
  - Buffer-based incremental ingestion (refinement triggers at BUFFER_SIZE=5)
  - LLM-driven incremental merge (top-2 similar skills, cosine > 0.55)
  - L1→L2→L3 hierarchical upgrade during merge
  - Safety valve: per-level capacity caps with forced compression
  - Skill deprecation (low-confidence or superseded skills)
  - Markdown export for human review

Storage layout:
  memory_bank/skills/
    patient_inquiry.json
    diagnostic_reasoning.json
    exam_execution.json
    diagnosis_conclusion.json
    buffer/
      patient_inquiry.json  (pending, not yet merged)
      ...
"""

import json
import os
import re
import time
from typing import List, Dict, Optional, Tuple

import numpy as np


# ═══════════════════════════════════════════════════════════
# Skill Categories & Constants
# ═══════════════════════════════════════════════════════════

SKILL_CATEGORIES = {
    "patient_inquiry":        "Patient Inquiry Skills",
    "diagnostic_reasoning":   "Diagnostic Reasoning Skills",
    "exam_execution":         "Exam Execution Skills",
    "diagnosis_conclusion":   "Diagnosis Conclusion Skills",
}

SKILL_CATEGORY_ORDER = [
    "patient_inquiry", "diagnostic_reasoning",
    "exam_execution", "diagnosis_conclusion",
]

# Category alias mapping (for LLM output normalization)
CATEGORY_ALIASES = {
    "consultation":           "patient_inquiry",
    "exam_selection":         "exam_execution",
    "result_interpretation":  "exam_execution",
    "differential":           "diagnostic_reasoning",
    "evidence_integration":   "diagnosis_conclusion",
    "safety":                 "diagnostic_reasoning",
    "communication":          "patient_inquiry",
    "inquiry":                "patient_inquiry",
    "diagnostic":             "diagnostic_reasoning",
    "reasoning":              "diagnostic_reasoning",
    "exam":                   "exam_execution",
    "tool_use":               "exam_execution",
    "conclusion":             "diagnosis_conclusion",
    "final_diagnosis":        "diagnosis_conclusion",
}

# Buffer & merge parameters
DEFAULT_SKILLS_DIR = "memory_bank/skills"
DEFAULT_BUFFER_DIR = "memory_bank/skills/buffer"
BUFFER_SIZE = 5             # Trigger refinement when buffer reaches this
MERGE_TOP_K = 2             # Find top-2 most similar skills per new skill
MERGE_COS_THRESHOLD = 0.55  # Minimum cosine similarity for merge consideration

# Per-category capacity caps (safety valve)
L1_MAX_PER_CATEGORY = 200
L2_MAX_PER_CATEGORY = 100
SAFETY_VALVE_RATIO = 1.2   # Trigger when > max * ratio


# ═══════════════════════════════════════════════════════════
# Skill Merge Prompt
# ═══════════════════════════════════════════════════════════

SKILL_MERGE_INCREMENTAL_PROMPT = """You are a medical knowledge refinement expert.
Decide whether the following skills should be merged.

## New Skill (to be added)
{new_skill}

## Most Similar Existing Skills (cosine similarity > {threshold})
{similar_skills}

## Decision Rules

Choose one of the following:

### A) Merge into 1
If the new skill and an existing skill share the same core logic (same diagnostic
thinking / exam strategy):
- Merge into 1 entry, accumulate source_count, keep the more complete description
- If the merged pattern becomes more general, upgrade abstraction_level (L1→L2 or L2→L3)

### B) Keep independent (2-3 entries)
If skills have meaningful differences that cannot be forced together:
- The new skill has independent value → output all originals + new skill
- The new pattern can augment an old skill → output 2 (enhanced old + discard new)

### C) Output 1-3 skills. Never output an empty list.

## Key Principles
- When uncertain, keep independent — better to keep more than less
- Only merge when core logic is truly identical
- The category field must remain unchanged

## Output Format
Output ONLY a JSON array (1-3 skills), no other text."""


# ═══════════════════════════════════════════════════════════
# File I/O Helpers
# ═══════════════════════════════════════════════════════════

def _normalize_category(cat: str) -> str:
    """Map non-standard category names to canonical names."""
    if cat in SKILL_CATEGORIES:
        return cat
    if cat in CATEGORY_ALIASES:
        return CATEGORY_ALIASES[cat]
    for alias, canonical in CATEGORY_ALIASES.items():
        if alias in cat or cat in alias:
            return canonical
    return cat


def _ensure_skill_fields(skill: dict) -> dict:
    """Fill missing default fields for backward compatibility."""
    if "abstraction_level" not in skill:
        skill["abstraction_level"] = "L1"
    if "generalizable_to" not in skill:
        skill["generalizable_to"] = []
    if "_score" not in skill:
        skill["_score"] = 50.0
    return skill


def _load_buffer(category: str) -> List[dict]:
    """Load the buffer for a category (pending new skills not yet merged)."""
    path = os.path.join(DEFAULT_BUFFER_DIR, f"{category}.json")
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, IOError):
        return []


def _save_buffer(category: str, skills: List[dict]):
    """Save the buffer for a category."""
    os.makedirs(DEFAULT_BUFFER_DIR, exist_ok=True)
    path = os.path.join(DEFAULT_BUFFER_DIR, f"{category}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(skills, f, ensure_ascii=False, indent=2)


def _load_skills(path: str = None) -> List[dict]:
    """Load all skills from the multi-file skill directory.

    Also loads buffer skills (marked with _in_buffer=True).
    """
    skills_dir = path or DEFAULT_SKILLS_DIR
    if not os.path.exists(skills_dir):
        return []

    all_skills = []
    for cat in SKILL_CATEGORIES:
        filepath = os.path.join(skills_dir, f"{cat}.json")
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                cat_skills = json.load(f)
                for s in cat_skills:
                    _ensure_skill_fields(s)
                all_skills.extend(cat_skills)

    # Load buffer skills (for retrieval visibility)
    existing_ids = {s.get("skill_id") for s in all_skills if s.get("skill_id")}
    buffer_dir = DEFAULT_BUFFER_DIR
    if os.path.exists(buffer_dir):
        for fname in os.listdir(buffer_dir):
            if fname.endswith(".json"):
                buf_skills = _load_buffer(fname[:-5])
                for bs in buf_skills:
                    bid = bs.get("skill_id", "")
                    if bid and bid not in existing_ids:
                        bs["_in_buffer"] = True
                        all_skills.append(bs)
                        existing_ids.add(bid)

    return all_skills


def _save_skills(skills: List[dict], path=None):
    """Save skills by category to individual JSON files."""
    skills_dir = path or DEFAULT_SKILLS_DIR
    os.makedirs(skills_dir, exist_ok=True)

    # Group by category
    by_category = {}
    for s in skills:
        cat = _normalize_category(s.get("category", "other"))
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(s)

    for cat in SKILL_CATEGORY_ORDER:
        items = by_category.get(cat, [])
        filepath = os.path.join(skills_dir, f"{cat}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)


# ═══════════════════════════════════════════════════════════
# Incremental Merge
# ═══════════════════════════════════════════════════════════

def _incremental_merge_skills(
    skills_list: List[dict],
    target_level: str,
    existing_skills: List[dict],
    llm_model: str,
    llm_endpoint: str,
    llm_api_key: str,
    top_k: int = MERGE_TOP_K,
    cos_threshold: float = MERGE_COS_THRESHOLD,
) -> List[dict]:
    """Incremental merge: each new skill is compared against top-k similar skills.

    LLM autonomously decides: merge into 1, keep all, or upgrade abstraction level.

    Args:
        skills_list: New skills to process (from buffer)
        target_level: Current abstraction level ("L1" or "L2")
        existing_skills: All existing skills of the same level
        top_k: Number of most similar skills to compare against
        cos_threshold: Cosine similarity threshold for merge consideration

    Returns:
        Merged skill list
    """
    from openai import OpenAI
    from src.skill_system.skill_math import resolve_confidence, merge_scores
    from src.skill_system.skill_embedder import embed_texts, _skill_to_text

    if not skills_list:
        return []

    client = OpenAI(base_url=llm_endpoint, api_key=llm_api_key)

    # Build embedding index for existing skills
    existing_texts = [_skill_to_text(s) for s in existing_skills]
    try:
        existing_embs = embed_texts(existing_texts)
    except Exception:
        return list(skills_list)  # bge-m3 unavailable, keep all

    result = list(existing_skills)
    processed = set()

    for new_s in skills_list:
        new_text = _skill_to_text(new_s)
        try:
            new_emb = embed_texts([new_text])[0]
            sims = np.dot(existing_embs, new_emb)
            # Find top-k with cos > threshold
            candidates = []
            for i, s in enumerate(existing_skills):
                if s.get("skill_id") in processed:
                    continue
                cos = float(sims[i])
                if cos >= cos_threshold:
                    candidates.append((cos, i, s))
            candidates.sort(key=lambda x: x[0], reverse=True)
            similar = [s for _, _, s in candidates[:top_k]]
        except Exception:
            similar = []

        if not similar:
            # No similar skills found, add as new
            new_s_copy = dict(new_s)
            new_s_copy["abstraction_level"] = target_level
            _ensure_skill_fields(new_s_copy)
            if "source_count" not in new_s_copy:
                new_s_copy["source_count"] = 1
            result.append(new_s_copy)
            continue

        # LLM merge decision
        prompt = SKILL_MERGE_INCREMENTAL_PROMPT.format(
            new_skill=json.dumps(new_s, ensure_ascii=False, indent=2),
            similar_skills=json.dumps(similar, ensure_ascii=False, indent=2),
            threshold=cos_threshold,
        )

        try:
            resp = client.chat.completions.create(
                model=llm_model,
                messages=[{"role": "system", "content": "Output ONLY a JSON array."},
                          {"role": "user", "content": prompt}],
                max_tokens=4096, temperature=0.3,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            content = resp.choices[0].message.content or "[]"
            match = re.search(r'\[.*\]', content, re.DOTALL)
            if match:
                merged = json.loads(match.group(0))
                if not merged:
                    result.append(new_s)
                    continue
                # Compute merged score: avg of source scores + small bonus
                source_scores = [resolve_confidence(s) for s in similar] + [resolve_confidence(new_s)]
                merged_score = merge_scores(source_scores)
                for m in merged:
                    if "abstraction_level" not in m:
                        m["abstraction_level"] = target_level
                    m["_score"] = round(merged_score, 1)
                    if merged_score >= 70.0:
                        m["confidence"] = "high"
                    elif merged_score >= 40.0:
                        m["confidence"] = "medium"
                    else:
                        m["confidence"] = "low"
                    if "source_count" not in m:
                        m["source_count"] = 1
                # Replace merged old skills with results
                similar_ids = {s.get("skill_id") for s in similar}
                result = [s for s in result if s.get("skill_id") not in similar_ids]
                result.extend(merged)
                for s in similar:
                    processed.add(s.get("skill_id"))
            else:
                result.append(new_s)
        except Exception as e:
            print(f"   LLM incremental merge failed: {e}, keeping as new")
            result.append(new_s)

    return result


# ═══════════════════════════════════════════════════════════
# Category Refinement Pipeline
# ═══════════════════════════════════════════════════════════

def refine_category(
    cat: str,
    library_path: str = None,
    llm_model: str = None,
    llm_endpoint: str = None,
    llm_api_key: str = None,
    eval_score: float = None,
):
    """Core refinement pipeline for a single category.

    Pipeline:
      1. Load existing skills + buffer
      2. L1 incremental merge → refined L1 + L2 candidates
      3. L2 incremental merge → refined L2 + L3 candidates
      4. Safety valve check (exceeds capacity → force compress)
      5. L3 dedup + write back to category file
      6. Clear buffer
    """
    skills_dir = library_path or DEFAULT_SKILLS_DIR
    cat_file = os.path.join(skills_dir, f"{cat}.json")

    # Load existing skills
    existing_cat = []
    if os.path.exists(cat_file):
        with open(cat_file, "r", encoding="utf-8") as f:
            existing_cat = json.load(f)
            if not isinstance(existing_cat, list):
                existing_cat = []

    # Load buffer
    buffer = _load_buffer(cat)
    if not buffer:
        return

    print(f"\n[{cat}] Buffer {len(buffer)} + existing {len(existing_cat)} → refining...")

    # Separate by level
    existing_l1 = [s for s in existing_cat if s.get("abstraction_level", "L1") == "L1"]
    existing_l2 = [s for s in existing_cat if s.get("abstraction_level") == "L2"]
    existing_l3 = [s for s in existing_cat if s.get("abstraction_level") == "L3"]

    buffer_l1 = [s for s in buffer if s.get("abstraction_level", "L1") == "L1"]
    buffer_l2 = [s for s in buffer if s.get("abstraction_level") == "L2"]

    # Step 1: L1 incremental merge
    refined_l1, l2_from_l1 = [], []
    if buffer_l1:
        merged_l1 = _incremental_merge_skills(
            buffer_l1, "L1", existing_l1,
            llm_model, llm_endpoint, llm_api_key,
            MERGE_TOP_K, MERGE_COS_THRESHOLD,
        )
        l2_from_l1 = [s for s in merged_l1 if s.get("abstraction_level") == "L2"]
        refined_l1 = [s for s in merged_l1 if s.get("abstraction_level") != "L2"]
        print(f"   L1: → {len(refined_l1)} kept + {len(l2_from_l1)} upgraded to L2")
    else:
        refined_l1 = list(existing_l1)

    # Step 2: L2 incremental merge
    all_l2 = existing_l2 + l2_from_l1
    refined_l2, l3_candidates = [], []
    if buffer_l2:
        merged_l2 = _incremental_merge_skills(
            buffer_l2, "L2", all_l2,
            llm_model, llm_endpoint, llm_api_key,
            MERGE_TOP_K, MERGE_COS_THRESHOLD,
        )
        l3_candidates = [s for s in merged_l2 if s.get("abstraction_level") == "L3"]
        refined_l2 = [s for s in merged_l2 if s.get("abstraction_level") != "L3"]
        print(f"   L2: → {len(refined_l2)} kept + {len(l3_candidates)} upgraded to L3")
    else:
        refined_l2 = list(all_l2)

    # Step 3: Safety valve check
    if len(refined_l1) > L1_MAX_PER_CATEGORY * SAFETY_VALVE_RATIO:
        refined_l1 = _force_compress(refined_l1, L1_MAX_PER_CATEGORY, "L1",
                                     llm_model, llm_endpoint, llm_api_key)
    if len(refined_l2) > L2_MAX_PER_CATEGORY * SAFETY_VALVE_RATIO:
        refined_l2 = _force_compress(refined_l2, L2_MAX_PER_CATEGORY, "L2",
                                     llm_model, llm_endpoint, llm_api_key)

    # Step 4: L3 dedup
    for s in l3_candidates:
        s["abstraction_level"] = "L3"
    all_l3 = existing_l3 + l3_candidates

    l3_dedup = {}
    for s in all_l3:
        key = s.get("name", s.get("skill_id", ""))
        if key in l3_dedup:
            l3_dedup[key]["source_count"] = l3_dedup[key].get("source_count", 1) + s.get("source_count", 1)
        else:
            l3_dedup[key] = dict(s)
    unique_l3 = list(l3_dedup.values())

    # Step 5: Deprecation + write back
    from src.skill_system.skill_math import should_deprecate
    all_final = refined_l1 + refined_l2 + unique_l3
    deprecated_count = 0
    for s in all_final:
        dep, _ = should_deprecate(s)
        if dep and s.get("status") != "deprecated":
            s["status"] = "deprecated"
            deprecated_count += 1
    if deprecated_count > 0:
        print(f"   Deprecated {deprecated_count} low-scoring skills")

    os.makedirs(skills_dir, exist_ok=True)
    with open(cat_file, "w", encoding="utf-8") as f:
        json.dump(all_final, f, ensure_ascii=False, indent=2)

    # Clear buffer
    _save_buffer(cat, [])

    # Rebuild embedding cache
    from src.skill_system.skill_embedder import invalidate_cache, build_embedding_cache
    invalidate_cache()
    build_embedding_cache(skills_dir=skills_dir, force=True)

    # Annotate initial confidence for new skills
    if eval_score is not None:
        from src.skill_system.skill_math import compute_initial_confidence, resolve_confidence
        for s in all_final:
            if s.get("_score") is None:
                level = s.get("abstraction_level", "L1")
                mode = "success" if eval_score >= 0.7 else "lesson"
                c0 = compute_initial_confidence(eval_score, level, mode)
                s["_score"] = round(c0, 1)
            score_val = resolve_confidence(s)
            if score_val >= 70:
                s["confidence"] = "high"
            elif score_val >= 40:
                s["confidence"] = "medium"
            else:
                s["confidence"] = "low"

    print(f"   [OK] [{cat}] Refinement complete: "
          f"buffer {len(buffer)} + existing {len(existing_cat)} → {len(all_final)}")
    print(f"      L1={len(refined_l1)} L2={len(refined_l2)} L3={len(unique_l3)}")


# ═══════════════════════════════════════════════════════════
# Public API: Skill Ingestion
# ═══════════════════════════════════════════════════════════

def ingest_skills(
    new_skills: List[dict],
    library_path: str = None,
    llm_model: str = None,
    llm_endpoint: str = None,
    llm_api_key: str = None,
    eval_score: float = None,
) -> List[dict]:
    """Buffer-based skill ingestion. Skills accumulate per category.

    When a category buffer reaches BUFFER_SIZE, refinement is triggered.
    Returns the full skill library after ingestion.
    """
    import time as _time

    # SKIP SOP skills for simplicity in this snapshot
    non_sop = [s for s in new_skills if s.get("category") != "sop"]
    if not non_sop:
        return _load_skills(library_path)

    # Group by category
    by_cat = {}
    for s in non_sop:
        cat = _normalize_category(s.get("category", "other"))
        if cat not in by_cat:
            by_cat[cat] = []
        by_cat[cat].append(s)

    # Dedup skill IDs
    existing = _load_skills(library_path)
    all_ids = {s.get("skill_id", "") for s in existing}
    for cat in SKILL_CATEGORIES:
        for s in _load_buffer(cat):
            all_ids.add(s.get("skill_id", ""))
    all_ids.discard("")

    for skills in by_cat.values():
        for s in skills:
            sid = s.get("skill_id", "")
            if not sid or sid in all_ids:
                s["skill_id"] = f"SKILL_{int(_time.time() * 1000) % 100000}_{id(s) % 10000}"
            all_ids.add(s["skill_id"])

    refined_any = False
    for cat, skills in by_cat.items():
        if not skills:
            continue
        buffer = _load_buffer(cat)
        buffer.extend(skills)
        _save_buffer(cat, buffer)
        print(f"   [+] [{cat}] buffer +{len(skills)} → {len(buffer)} total")

        # Incrementally update embedding cache
        try:
            from src.skill_system.skill_embedder import update_embedding_cache
            update_embedding_cache(skills)
        except Exception:
            pass

        # Trigger refinement if buffer is full
        if len(buffer) >= BUFFER_SIZE:
            refine_category(cat, library_path, llm_model, llm_endpoint, llm_api_key, eval_score)
            refined_any = True

    return _load_skills(library_path)


def _force_compress(
    skills: List[dict],
    target_count: int,
    level: str,
    llm_model: str, llm_endpoint: str, llm_api_key: str,
) -> List[dict]:
    """Safety valve: force-compress skills when exceeding capacity.

    Strategy: split into batches of 10, LLM-merge each batch.
    """
    from openai import OpenAI

    if len(skills) <= target_count:
        return skills

    print(f"   SAFETY VALVE: {level} has {len(skills)}, compressing to {target_count}")

    batch_size = 10
    batches = [skills[i:i + batch_size] for i in range(0, len(skills), batch_size)]

    compressed = []
    client = OpenAI(base_url=llm_endpoint, api_key=llm_api_key)
    for batch in batches:
        prompt = SKILL_MERGE_INCREMENTAL_PROMPT.format(
            new_skill=json.dumps(batch[0], ensure_ascii=False, indent=2),
            similar_skills=json.dumps(batch[1:], ensure_ascii=False, indent=2),
            threshold=0.5,
        )
        try:
            resp = client.chat.completions.create(
                model=llm_model,
                messages=[{"role": "system", "content": "Output ONLY a JSON array."},
                          {"role": "user", "content": prompt}],
                max_tokens=4096, temperature=0.3,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            content = resp.choices[0].message.content or "[]"
            match = re.search(r'\[.*\]', content, re.DOTALL)
            if match:
                merged = json.loads(match.group(0))
                for s in merged:
                    s["abstraction_level"] = level
                compressed.extend(merged)
            else:
                compressed.extend(batch)
        except Exception:
            compressed.extend(batch)

    if len(compressed) > target_count:
        compressed = _force_compress(compressed, target_count, level,
                                     llm_model, llm_endpoint, llm_api_key)

    return compressed


def get_skill_count(library_path: str = None) -> int:
    """Get total number of skills in the library."""
    return len(_load_skills(library_path))
