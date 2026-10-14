# Architecture and interfaces

## A consultation, step by step

1. The caller loads a case and keeps `eval_hidden` separate from online inputs. The patient receives only `patient_visible`; the examination component receives only `exam_visible`.
2. The caller initializes the doctor with allowed opening information and registers the agents on a `MessageBus`.
3. Before each turn, `inject_skills_per_turn` selects needed categories and retrieves candidates using current evidence and differential hypotheses. Selected skills become prompt guidance.
4. `DoctorAgent.think` produces an evidence summary, hypotheses, information gaps, and a plan. `act` selects `inquiry`, `exam_request`, or `final_diagnosis`.
5. The loop routes the request to the registered component. The doctor observes its response and updates the evidence. The bus records messages.
6. Final diagnosis or the turn limit ends the consultation. A fallback may require extra model calls, so eight turns is not an eight-request budget.
7. Outside the online loop, `extract_skills` can read the trajectory and training reference. Candidate scoring and category refinement prepare the next library state. The omitted runner coordinates evaluation, feedback, and training/inference modes.

## External services

The live components use an OpenAI-compatible chat interface and an embedding endpoint. Secrets are passed by the caller; `.env.example` is documentation and is not automatically loaded.

```python
import os
from src.agents.doctor_agent import DoctorAgent
from src.orchestrator.message_bus import MessageBus

class LoopDrivenDoctor(DoctorAgent):
    # The partial snapshot leaves receive abstract. The loop calls observe directly.
    def receive(self, message):
        return None

doctor = LoopDrivenDoctor("doctor", MessageBus())
doctor.set_llm_config(
    model=os.environ["LLM_MODEL"],
    endpoint=os.environ["LLM_BASE_URL"],
    api_key=os.environ["LLM_API_KEY"],
)
```

This configures one component; it does not start the omitted session manager. Backend calls include Qwen-style `extra_body={"chat_template_kwargs": {"enable_thinking": False}}`. Verify support before live use or adapt the call for your provider.

The embedding client reads `BGE_M3_ENDPOINT` and calls `/embeddings` with `model="bge-m3"`. It normalizes returned vectors before cosine comparison. Its default cache is under `memory_bank/`; initialize a working copy rather than treating example files as a trained bank:

```bash
python -c "import shutil; shutil.copytree('skills', 'memory_bank/skills', dirs_exist_ok=True)"
python -c "from src.skill_system.skill_embedder import build_embedding_cache; build_embedding_cache()"
```

The second command contacts your configured embedding service. Retrieval defaults to that working directory, while `retrieve_skills_by_embedding(..., skills_dir=...)` accepts an explicit location. No prebuilt cache or learned embeddings are supplied.

## Snapshot-specific behavior and limitations

- Retrieval selects up to two skills **per chosen category**, potentially eight across four categories. It does not select two per abstraction level.
- `DoctorAgent` inherits an abstract `receive` contract; supply a loop/session adapter when instantiating it. The adapter above and offline fixture make this boundary explicit.
- The category selector defaults to all four categories if model output is unusable. Embedding-retrieval errors produce an empty candidate list.
- The historical injection function returns early when no new match is found. It does not clear previously assigned guidance on every early-return path; callers inspecting a live system must account for this behavior.
- Embedding text is limited to 512 characters. Retrieval queries and several prompt/history views use character limits. This snapshot does not implement later context handling and cannot guarantee complete-context reasoning.
- The patient prompt contains legacy ambiguity between unknown information and a negative answer. This snapshot is retained for inspection rather than presented as a resolved clinical information-preservation implementation.
- Image-call failures can fall back to text-only generation. A live evaluation must track which modality was actually consumed.
- Merge candidates use similarity, category/level processing, and model decisions. A larger bank or successful merge does not by itself establish improved accuracy.
- Confidence/impact values are heuristic model-derived scores, not causal attribution or calibrated probabilities.
- The snapshot does not provide a frozen-evaluation command or reconstruct complete paper results. The offline example must not be used as an accuracy benchmark.

These boundaries make the released implementation inspectable; they do not imply that omitted components have been validated here.
