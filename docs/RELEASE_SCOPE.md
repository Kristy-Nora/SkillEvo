# Release scope

This is a selected source snapshot of the paper's multi-agent system, prepared for code inspection during peer review. It is not a turnkey release of the full evaluation pipeline.

## Included

| Module | Included functionality |
|---|---|
| `src/agents/base_agent.py` | Abstract agent interface, LLM calls, image serialization |
| `src/agents/doctor_agent.py` | Structured Think/Act/Observe prompts and doctor state |
| `src/agents/patient_agent.py` | Responses based on patient-visible data |
| `src/orchestrator/message_bus.py` | Synchronous message routing and trajectory recording |
| `src/orchestrator/react_loop.py` | Consultation control, per-turn retrieval and injection |
| `src/skill_system/skill_math.py` | Initial confidence, retrieval score, feedback and merge formulas |
| `src/skill_system/skill_embedder.py` | Embedding requests, cache, and semantic retrieval |
| `src/skill_system/skill_extractor.py` | Full-trajectory extraction prompt and formatter |
| `src/skill_system/skill_library_manager.py` | Category buffers, incremental refinement and persistence |

## Not included

- Examination and evaluation agents, guideline retriever, segmentation models, and the permission/session manager.
- The top-level training, feedback-application, and benchmark-evaluation runner.
- The separate per-turn extraction/evaluation pipeline. Per-turn evaluation concepts in prompts do not make it an included runner.
- Benchmark splits, patient images, book indexes, checkpoints, complete learned libraries, and experimental logs.
- Later single-agent diagnosis-only experiments, web-search adapters, and pair-specific skill formats.

`DoctorReActLoop` accepts examination and permission objects from its caller. The offline example supplies clearly labeled stubs to show these interfaces; it does not recreate the omitted agents or enforce the full experimental access policy.

## Examples and provenance

The four English skills are adapted, simplified illustrations of categories and abstraction levels from the supplied review snapshot. Their identifiers begin with `ILLUSTRATION_`. Experimental confidence values and source counts are deliberately absent because this release does not include the supporting training records. They must not be presented as verbatim successful experimental outputs.

The case in `data/example_case.json` is a synthetic text-only fixture with an artificial target, not a dataset case or a clinical diagnostic example. The offline script uses scripted actions, not model-generated reasoning. Its apparent success is a software demonstration, not a result.

## Preparation changes

The selected Python algorithms are preserved. Preparation changes are limited to documentation, a generic model identifier in a docstring, illustrative data, and new offline examples/tests. No claim is made that this snapshot incorporates later experimental fixes.

The old review README contained a multiplicative retrieval description and a skip threshold of 0.45. The release README instead documents the included code: additive similarity/confidence ranking and a skip threshold of 0.30. Documentation is aligned to this snapshot; it does not resolve any mismatch with a manuscript or another experimental code revision.

## Validation boundary

The included tests cover local imports, scoring behavior, message routing, illustrative data integrity, and a scripted consultation. No external service is contacted. Passing these checks does not validate live provider compatibility, medical content, learned merging quality, or reported experimental accuracy. A license has not been selected in this snapshot.
