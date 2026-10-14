"""
Skill mathematics engine — core formulas for the skill scoring system.

Provides the mathematical foundation for all key skill operations:
  - Formula 1: Initial skill confidence scoring (0-100 scale)
  - Formula 2: Feedback-driven score evolution (impact delta)
  - Formula 3: Merge scoring for combined skills
  - Formula 4: Retrieval relevance scoring
  - Formula 5: Skill deprecation criteria
  - Formula 6: Training mode determination

All formulas are pure functions — independently computable, composable, and trackable.
"""

import math
from typing import List, Dict, Optional, Tuple


# ═══════════════════════════════════════════════════════════════
# Hyperparameters
# ═══════════════════════════════════════════════════════════════

# Scoring system (0-100 scale)
SCORE_DEFAULT = 50          # Default score for new skills
SCORE_DELTA_MAX = 5.0       # Maximum score change per feedback event
SCORE_SENSITIVITY_K = 2.0   # tanh sensitivity coefficient

# Formula 1: Initial confidence
BETA_EVAL = 0.7             # Evaluation score weight
LEVEL_BONUS = {             # Abstraction level prior bonus (0-100)
    "L3": 100.0,
    "L2": 70.0,
    "L1": 50.0,
}
LESSON_PENALTY_FLOOR = 0.4
LESSON_PENALTY_SCALE = 0.6

# Formula 5: Skill deprecation
THETA_MIN_CONF = 15.0       # Minimum confidence threshold (0-100)
N_MIN_SOURCE = 1            # Minimum source count
THETA_SUBSUME = 0.85        # Supersedence similarity threshold

# Formula 6: Training mode thresholds
THRESHOLD_SUCCESS = 0.70    # E >= 0.70 → success mode
THRESHOLD_SKIP = 0.30       # E < 0.30 → skip training


# ═══════════════════════════════════════════════════════════════
# Formula 1: Skill Initial Confidence
# ═══════════════════════════════════════════════════════════════

def compute_initial_confidence(
    eval_score: float,
    abstraction_level: str,
    trajectory_mode: str = "success",
) -> float:
    r"""Compute the initial confidence score C_0(s) for a newly extracted skill.

    Formula:
      C_0(s) = beta * E(tau) * 100 + (1 - beta) * g(l(s))

    Where:
      - E(tau) in [0, 1]: weighted evaluation score of the source trajectory
      - g(L3)=100, g(L2)=70, g(L1)=50: abstraction level prior bonus
      - beta = 0.7: balance between evaluation score and prior

    For lesson mode (E < 0.7):
      C_0'(s) = C_0(s) * (0.4 + 0.6 * E(tau))
    """
    level_bonus = LEVEL_BONUS.get(abstraction_level, LEVEL_BONUS["L1"])
    c0 = BETA_EVAL * eval_score * 100.0 + (1.0 - BETA_EVAL) * level_bonus
    if trajectory_mode == "lesson":
        c0 = c0 * (LESSON_PENALTY_FLOOR + LESSON_PENALTY_SCALE * eval_score)
    return min(100.0, max(0.0, c0))


# ═══════════════════════════════════════════════════════════════
# Formula 2: Feedback-Driven Score Update (Impact Delta)
# ═══════════════════════════════════════════════════════════════

def compute_impact_delta(impact_score: float) -> float:
    r"""Compute score adjustment delta from a single evaluation's impact_score.

    Formula:
      delta = delta_max * tanh(k * impact_score)

    Where:
      - impact_score in [-1, 1]: LLM-evaluated skill impact (direction + magnitude)
      - delta_max = 5.0: max single-event adjustment
      - k = 2.0: tanh sensitivity

    Design rationale:
      - tanh is approximately linear for small inputs, saturates for large ones
      - k=2 gives: impact=0.5 → delta≈3.8, impact=1.0 → delta≈4.8
      - Symmetric: positive and negative impacts are handled equally
    """
    return SCORE_DELTA_MAX * math.tanh(SCORE_SENSITIVITY_K * impact_score)


# ═══════════════════════════════════════════════════════════════
# Formula 3: Merge Scoring
# ═══════════════════════════════════════════════════════════════

def merge_scores(source_scores: list) -> float:
    """Compute the initial score for a merged skill.

    score_merged = avg(source_scores) + bonus
    bonus = min(3.0, len(source_scores) * 0.5)

    The small bonus encourages multi-source merges but caps at +3.
    The skill must still prove itself through subsequent feedback.
    """
    if not source_scores:
        return float(SCORE_DEFAULT)
    avg = sum(source_scores) / len(source_scores)
    bonus = min(3.0, len(source_scores) * 0.5)
    return min(100.0, avg + bonus)


# ═══════════════════════════════════════════════════════════════
# Formula 4: Retrieval Relevance Score (v3.2 simplified)
# ═══════════════════════════════════════════════════════════════

def compute_retrieval_score(
    cosine_similarity: float,
    confidence: float = 50.0,
) -> float:
    r"""Compute retrieval score R(s|c) — semantic + confidence (50/50 split).

    Formula:
      R(s|c) = 0.5 * cos(e_s, e_c) + 0.5 * (C(s)/100)

    Where:
      - cos in [0, 1]: bge-m3 cosine similarity
      - C(s) in [0, 100]: skill confidence (0-100 scale)

    Injection thresholds: cos >= 0.50, R >= 0.25
    """
    cos = max(0.0, min(1.0, cosine_similarity))
    conf = max(0.0, min(100.0, confidence))
    return 0.5 * cos + 0.5 * (conf / 100.0)


# ═══════════════════════════════════════════════════════════════
# Formula 5: Skill Deprecation Criterion
# ═══════════════════════════════════════════════════════════════

def should_deprecate(
    skill: dict,
    newer_skills: List[dict] = None,
) -> Tuple[bool, str]:
    r"""Determine whether a skill should be deprecated.

    Condition 1 (quality too low):
      C(s) < theta_min_conf AND n_s < n_min

    Condition 2 (superseded):
      exists s': Sim(s, s') > theta_subsume AND C(s') >= C(s) AND n_{s'} >= n_s

    Deprecated skills are marked, not deleted — preserving the audit trail.
    """
    conf = resolve_confidence(skill)
    n = skill.get("source_count", 1)

    # Condition 1: quality too low
    if conf < THETA_MIN_CONF and n < N_MIN_SOURCE + 1:
        return True, f"low_quality(C={conf:.1f}<{THETA_MIN_CONF}, n={n})"

    # Condition 2: superseded
    if newer_skills:
        for new_s in newer_skills:
            new_conf = resolve_confidence(new_s)
            new_n = new_s.get("source_count", 1)
            same_cat = new_s.get("category") == skill.get("category")
            same_level = new_s.get("abstraction_level") == skill.get("abstraction_level")
            if same_cat and same_level and new_conf >= conf and new_n >= n:
                name_sim = _simple_name_similarity(
                    skill.get("name", ""), new_s.get("name", ""))
                if name_sim > THETA_SUBSUME:
                    return True, f"superseded_by({new_s.get('skill_id','?')}, sim={name_sim:.2f})"

    return False, ""


def _simple_name_similarity(name1: str, name2: str) -> float:
    """Simple name similarity based on character overlap."""
    if not name1 or not name2:
        return 0.0
    set1 = set(name1)
    set2 = set(name2)
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


# ═══════════════════════════════════════════════════════════════
# Formula 6: Training Mode Determination
# ═══════════════════════════════════════════════════════════════

def determine_training_mode(eval_score: Optional[float]) -> Tuple[str, bool]:
    r"""Determine training mode based on evaluation score E(tau).

      success  (E >= 0.70): learn from success — skills directly generalizable
      lesson   (0.30 <= E < 0.70): learn from mistakes — extract what went wrong
      skip     (E < 0.30): quality too low, noise dominates signal
    """
    if eval_score is None:
        return "mixed", False

    if eval_score >= THRESHOLD_SUCCESS:
        return "success", False
    elif eval_score >= THRESHOLD_SKIP:
        return "lesson", False
    else:
        return "skip", True


# ═══════════════════════════════════════════════════════════════
# Utility Functions
# ═══════════════════════════════════════════════════════════════

def resolve_confidence(skill: dict) -> float:
    """Extract numerical confidence (0-100) from a skill dict.

    Compatible with old 0-1 format (auto-converts to 0-100).
    Handles _score field, confidence field (numeric or label).
    """
    score = skill.get("_score")
    if score is not None:
        s = float(score)
        if s < 2.0:  # Old 0-1 format → convert to 0-100
            s = s * 100.0
        return s
    conf = skill.get("confidence")
    if conf is not None:
        if isinstance(conf, (int, float)):
            s = float(conf)
            if s < 2.0:
                s = s * 100.0
            return s
        label_map = {"high": 85.0, "medium": 55.0, "low": 30.0}
        return float(label_map.get(str(conf).lower(), 50.0))
    return float(SCORE_DEFAULT)


def annotate_skills_with_confidence(
    skills: List[dict],
    eval_score: float,
    trajectory_mode: str,
) -> List[dict]:
    """Annotate a batch of newly extracted skills with initial confidence scores."""
    for s in skills:
        level = s.get("abstraction_level", "L1")
        c0 = compute_initial_confidence(eval_score, level, trajectory_mode)
        s["_score"] = round(c0, 1)
        if c0 >= 70.0:
            s["confidence"] = "high"
        elif c0 >= 40.0:
            s["confidence"] = "medium"
        else:
            s["confidence"] = "low"
    return skills
