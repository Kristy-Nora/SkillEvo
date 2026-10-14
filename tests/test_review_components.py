"""Offline checks for released interfaces; no medical accuracy claims."""
import importlib
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from examples.offline_walkthrough import run_demo
from src.orchestrator.message_bus import AgentMessage, MessageBus, MessageType
from src.skill_system.skill_math import (
    compute_impact_delta,
    compute_initial_confidence,
    compute_retrieval_score,
    determine_training_mode,
    merge_scores,
)

ROOT = Path(__file__).resolve().parents[1]


class ReviewComponentsTest(unittest.TestCase):
    def test_all_released_modules_import(self):
        for path in (ROOT / "src").rglob("*.py"):
            name = ".".join(path.relative_to(ROOT).with_suffix("").parts)
            if name.endswith(".__init__"):
                name = name.removesuffix(".__init__")
            with self.subTest(module=name):
                importlib.import_module(name)

    def test_initial_confidence_and_lesson_adjustment(self):
        self.assertAlmostEqual(compute_initial_confidence(0.8, "L2"), 77.0)
        self.assertAlmostEqual(compute_initial_confidence(0.5, "L1", "lesson"), 35.0)

    def test_feedback_is_signed_and_bounded(self):
        self.assertEqual(compute_impact_delta(0), 0)
        self.assertAlmostEqual(compute_impact_delta(-0.5), -compute_impact_delta(0.5))
        self.assertLess(compute_impact_delta(1), 5)

    def test_ranking_matches_documented_formula(self):
        self.assertAlmostEqual(compute_retrieval_score(0.75, 80), 0.775)
        self.assertEqual(compute_retrieval_score(2, 200), 1)

    def test_training_mode_boundaries(self):
        self.assertEqual(determine_training_mode(0.70), ("success", False))
        self.assertEqual(determine_training_mode(0.30), ("lesson", False))
        self.assertEqual(determine_training_mode(0.29), ("skip", True))

    def test_merge_score_is_capped(self):
        self.assertAlmostEqual(merge_scores([60, 80]), 71.0)
        self.assertEqual(merge_scores([100, 100]), 100)

    def test_bus_routes_and_records_exchange(self):
        class Responder:
            def receive(self, message):
                return AgentMessage("reply", "patient", "doctor", MessageType.PATIENT_RESPONSE, {"answer": "Unknown"}, 1)
        bus = MessageBus()
        bus.register("patient", Responder())
        result = bus.send(bus.make_message("doctor", "patient", MessageType.DOCTOR_INQUIRY, {"question": "Example?"}, 1))
        self.assertEqual(result.content["answer"], "Unknown")
        self.assertEqual(len(bus.get_full_trajectory()), 2)

    def test_four_english_skill_illustrations(self):
        categories = set()
        for path in (ROOT / "skills").glob("*.json"):
            for skill in json.loads(path.read_text(encoding="utf-8")):
                categories.add(skill["category"])
                self.assertTrue(skill["skill_id"].startswith("ILLUSTRATION_"))
                self.assertNotIn("_score", skill)
                self.assertNotIn("source_count", skill)
                for key in ["trigger", "action", "decision_rule", "effect", "pitfall"]:
                    self.assertTrue(skill[key])
        self.assertEqual(categories, {"patient_inquiry", "diagnostic_reasoning", "exam_execution", "diagnosis_conclusion"})

    def test_scripted_loop_without_network_or_target_leak(self):
        with patch("socket.socket.connect", side_effect=AssertionError("Network is forbidden in this demo")):
            demo = run_demo()
        result = demo["consultation"]
        self.assertEqual(result["total_turns"], 3)
        self.assertEqual(result["termination_reason"], "diagnosis_given")
        self.assertEqual(demo["network_calls"], 0)
        events = result["trajectory"]
        self.assertIn("patient_response", [e["msg_type"] for e in events])
        self.assertIn("exam_result", [e["msg_type"] for e in events])
        self.assertNotIn("eval_hidden", json.dumps(events))


if __name__ == "__main__":
    unittest.main()
