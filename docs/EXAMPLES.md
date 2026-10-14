# English skill examples

These four adapted illustrations show the schema and one example per functional category. They are not raw experimental exports, successful test cases, or medically validated guidelines. No measured score or source count is assigned. See the machine-readable files in [`skills/`](../skills/).

## 1. Patient inquiry · L2

**Clarify the symptom and treatment timeline**

| Field | Illustration |
|---|---|
| Trigger | Symptoms change during or after an intervention. |
| Action | Ask when symptoms began, when the intervention started, and what changed afterward. Record unknown dates explicitly. |
| Decision rule | Use chronology to guide further questions; sequence alone does not prove causation. |
| Intended effect | Establish an explicit timeline before updating hypotheses. |
| Pitfall | Treating an unreported event as absent or sequence as causation. |

## 2. Diagnostic reasoning · L2

**Update hypotheses when new evidence arrives**

Compare each new response or report with active hypotheses. Record support, contradiction, and unknowns. Revise the leading hypothesis when needed rather than searching only for confirmation. The desired behavior is an observable update, not simply repeating a reasoning principle.

## 3. Examination execution · L3

**Choose examinations that distinguish active hypotheses**

Before requesting an available examination, state the information gap and how alternative results would change the differential. An unavailable report must remain unknown; it cannot be treated as normal. This demonstrates a reusable action-selection rule rather than a disease-specific answer.

## 4. Diagnosis conclusion · L3

**Match diagnostic certainty to available evidence**

Separate observations, interpretations, and assumptions. Link the conclusion to available evidence and name unresolved discriminators. The `effect` field describes intended behavior, not proof that the skill improves correctness.

## Walking through the interfaces

```bash
python -m examples.offline_walkthrough
```

The example reads a synthetic case, routes one inquiry and one examination request, and returns an artificial final label. It reuses the released loop and bus with deterministic stubs. It then prints one skill and computes illustrative confidence and retrieval values. The skill is chosen explicitly for display; no embedding search, extraction, evolution, or learned diagnosis takes place.

Online inputs and offline target are separate in the fixture. No actual patient image or benchmark explanation is included.
