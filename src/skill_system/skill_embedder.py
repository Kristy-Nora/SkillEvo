"""
Skill embedder — bge-m3 embedding-based semantic skill retrieval.

Core functionality:
  - Convert skill texts to bge-m3 (1024-dim) embedding vectors
  - Cache embeddings to disk for fast retrieval
  - Semantic search: query embedding → cosine similarity → multi-factor ranking
  - Incremental update when new skills are added
  - Full rebuild after skill merge/refinement
"""

import json
import os
import time
from typing import List, Dict, Optional, Tuple

import numpy as np

# bge-m3 embedding service endpoint
# NOTE: Configure this via environment variable before running:
#   export BGE_M3_ENDPOINT="http://localhost:8006/v1"
BGE_M3_ENDPOINT = os.environ.get("BGE_M3_ENDPOINT", "http://localhost:8006/v1")

# Default paths
DEFAULT_SKILLS_DIR = "memory_bank/skills"
EMBEDDING_CACHE_FILE = "skill_embeddings.npy"
SKILL_ID_FILE = "skill_ids.json"

# Embedding dimension (bge-m3: 1024)
EMBEDDING_DIM = 1024

# Cosine similarity threshold for injection
SIM_THRESHOLD = 0.50
R_SCORE_MIN = 0.25


def _skill_to_text(skill: dict) -> str:
    """Convert a skill dict to embedding text.

    Uses the four core fields: name, trigger, action, effect.
    Truncated to 512 chars to respect bge-m3's efficient encoding range.
    """
    parts = [
        skill.get("name", ""),
        skill.get("trigger", ""),
        skill.get("action", ""),
        skill.get("effect", ""),
    ]
    text = " | ".join(p for p in parts if p)
    return text[:512]


def embed_texts(texts: List[str], endpoint: str = None) -> np.ndarray:
    """Embed a list of texts using bge-m3.

    Args:
        texts: List of text strings to embed
        endpoint: bge-m3 service endpoint (default: localhost:8006)

    Returns:
        numpy array of shape (len(texts), 1024) with L2-normalized embeddings
    """
    import requests

    ep = endpoint or BGE_M3_ENDPOINT

    resp = requests.post(
        f"{ep}/embeddings",
        json={"input": texts, "model": "bge-m3"},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    embeddings = np.array([item["embedding"] for item in data["data"]], dtype=np.float32)

    # L2-normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    embeddings = embeddings / norms

    return embeddings


def build_embedding_cache(skills_dir: str = None, force: bool = False):
    """Build (or rebuild) the embedding cache for all skills.

    Args:
        skills_dir: Directory containing category JSON files
        force: If True, rebuild even if cache exists
    """
    skills_dir = skills_dir or DEFAULT_SKILLS_DIR

    if not os.path.exists(skills_dir):
        print(f"   [Embedder] Skills directory not found: {skills_dir}")
        return

    # Load all skills
    all_skills = _load_all_skills(skills_dir)
    if not all_skills:
        print("   [Embedder] No skills found, skipping cache build")
        return

    # Build embeddings
    texts = [_skill_to_text(s) for s in all_skills]
    print(f"   [Embedder] Building embedding cache for {len(all_skills)} skills...")
    embeddings = embed_texts(texts)

    # Save cache
    cache_dir = os.path.join(skills_dir, "..")
    emb_path = os.path.join(cache_dir, EMBEDDING_CACHE_FILE)
    id_path = os.path.join(cache_dir, SKILL_ID_FILE)

    np.save(emb_path, embeddings)
    with open(id_path, "w", encoding="utf-8") as f:
        json.dump([s.get("skill_id", "") for s in all_skills], f, ensure_ascii=False)

    print(f"   [Embedder] Cache built: {emb_path} ({embeddings.shape})")


def update_embedding_cache(new_skills: List[dict], skills_dir: str = None):
    """Incrementally update the embedding cache with new skills.

    Only computes embeddings for new skills (not already in cache).
    If the cache doesn't exist, builds it from scratch.
    """
    skills_dir = skills_dir or DEFAULT_SKILLS_DIR
    cache_dir = os.path.join(skills_dir, "..")
    emb_path = os.path.join(cache_dir, EMBEDDING_CACHE_FILE)
    id_path = os.path.join(cache_dir, SKILL_ID_FILE)

    # If no cache exists, build from scratch
    if not os.path.exists(emb_path) or not os.path.exists(id_path):
        build_embedding_cache(skills_dir, force=True)
        return

    # Filter out skills already in cache
    with open(id_path, "r", encoding="utf-8") as f:
        existing_ids = set(json.load(f))

    truly_new = [s for s in new_skills
                 if s.get("skill_id", "") not in existing_ids]

    if not truly_new:
        return

    # Compute embeddings for new skills only
    texts = [_skill_to_text(s) for s in truly_new]
    new_embs = embed_texts(texts)

    # Append to cache
    old_embs = np.load(emb_path)
    updated_embs = np.vstack([old_embs, new_embs])
    updated_ids = list(existing_ids) + [s.get("skill_id", "") for s in truly_new]

    np.save(emb_path, updated_embs)
    with open(id_path, "w", encoding="utf-8") as f:
        json.dump(updated_ids, f, ensure_ascii=False)


def invalidate_cache(skills_dir: str = None):
    """Remove the embedding cache (forces rebuild on next retrieval)."""
    skills_dir = skills_dir or DEFAULT_SKILLS_DIR
    cache_dir = os.path.join(skills_dir, "..")
    for fname in [EMBEDDING_CACHE_FILE, SKILL_ID_FILE]:
        path = os.path.join(cache_dir, fname)
        if os.path.exists(path):
            os.remove(path)


def retrieve_skills_by_embedding(
    query: str,
    top_k: int = 10,
    skills_dir: str = None,
    sim_threshold: float = SIM_THRESHOLD,
    r_min: float = R_SCORE_MIN,
) -> List[Tuple[dict, float]]:
    """Retrieve the top-k most relevant skills via bge-m3 embedding search.

    Args:
        query: Query text (current evidence + differential diagnoses)
        top_k: Number of skills to return
        skills_dir: Skills directory path
        sim_threshold: Minimum cosine similarity to include
        r_min: Minimum retrieval score R(s|c) to include

    Returns:
        List of (skill_dict, retrieval_score) tuples, sorted by score descending
    """
    from src.skill_system.skill_math import compute_retrieval_score, resolve_confidence

    skills_dir = skills_dir or DEFAULT_SKILLS_DIR
    cache_dir = os.path.join(skills_dir, "..")
    emb_path = os.path.join(cache_dir, EMBEDDING_CACHE_FILE)
    id_path = os.path.join(cache_dir, SKILL_ID_FILE)

    if not os.path.exists(emb_path):
        raise RuntimeError("Embedding cache not found. Run build_embedding_cache() first.")

    # Load cache
    embeddings = np.load(emb_path)
    with open(id_path, "r", encoding="utf-8") as f:
        skill_ids = json.load(f)

    # Load skills
    all_skills = _load_all_skills(skills_dir)

    # Embed query
    query_emb = embed_texts([query])[0]

    # Compute cosine similarities (embeddings are already L2-normalized)
    cos_sims = np.dot(embeddings, query_emb)

    # Compute retrieval scores and filter
    results = []
    for i, (skill_id, cos) in enumerate(zip(skill_ids, cos_sims)):
        if cos < sim_threshold:
            continue
        # Find the skill dict
        skill = next((s for s in all_skills if s.get("skill_id") == skill_id), None)
        if skill is None:
            continue
        conf = resolve_confidence(skill)
        r_score = compute_retrieval_score(float(cos), conf)
        if r_score < r_min:
            continue
        skill["_cosine_sim"] = float(cos)
        skill["_retrieval_score"] = round(r_score, 4)
        results.append((skill, r_score))

    # Sort by retrieval score descending
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


def _load_all_skills(skills_dir: str) -> List[dict]:
    """Load all skills from the multi-file skill directory.

    Handles categories: patient_inquiry, diagnostic_reasoning,
    exam_execution, diagnosis_conclusion.
    """
    from src.skill_system.skill_library_manager import _load_skills
    return _load_skills(skills_dir)
