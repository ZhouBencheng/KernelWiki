#!/usr/bin/env python3
"""CPU-only truth table for the Orin evidence registry + shared validator seam.

Deterministic, no hardware. Exercises the shared structural seam
(validate_evidence_registry) through the Orin and Thor wrappers with synthetic
records, and checks that the refactor preserves Thor behavior. Run with:
    python3 -I -m unittest discover -s tests -p 'test_*.py'
"""

import importlib.util
import sys
import unittest
from pathlib import Path

SKILL_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = SKILL_ROOT / "scripts"
DATA = SKILL_ROOT / "data"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec.loader.exec_module(mod)
    return mod


validate = _load_module("orin_validate_mod", SCRIPTS / "validate.py")

import yaml  # noqa: E402

TAGS = yaml.safe_load((DATA / "tags.yaml").read_text(encoding="utf-8"))
RESOLVABLE = {"doc-nvidia-ampere-sm87-features", "doc-nvidia-jetson-orin"}
H = "a" * 64  # a structurally valid sha256 digest

LADDER_FIELDS = ("evidence_status", "runtime_check", "missing_check",
                 "artifacts", "ncu_records")


def _corpus_source_ids():
    ids = set()
    for md in (SKILL_ROOT / "sources").rglob("*.md"):
        fm = validate.extract_frontmatter(md)
        if isinstance(fm, dict) and "id" in fm:
            ids.add(fm["id"])
    return ids


def _wrap(rec):
    return {"probe_inventory": {"device": {"cc": "8.7", "name": "Orin"},
                                "tool": {"nvcc": "12.6.85"}},
            "features": [rec]}


def _good_unsupported(**over):
    rec = {
        "id": "u", "feature": "tma", "targets": ["sm_87"],
        "applicability": "unsupported",
        "scope": "TMA bulk copy (cp.async.bulk tensor-map engine)",
        "reason": "TMA is a Hopper sm_90 feature; unavailable on sm_87.",
        "source_ids": ["doc-nvidia-ampere-sm87-features"],
    }
    rec.update(over)
    return rec


def check_orin(data):
    return validate.validate_orin_evidence_registry(RESOLVABLE, TAGS, data=data)


class OrinUnsupportedTruthTableTests(unittest.TestCase):
    """Unsupported applicability accepts documentation, rejects any ladder field."""

    def test_good_unsupported_record_passes(self):
        self.assertEqual(check_orin(_wrap(_good_unsupported())), [])

    def test_unsupported_requires_scope(self):
        for bad in ("", "   "):
            rec = _good_unsupported(scope=bad)
            self.assertTrue(check_orin(_wrap(rec)), f"empty scope {bad!r} must reject")

    def test_unsupported_requires_reason(self):
        for bad in ("", "   "):
            rec = _good_unsupported(reason=bad)
            self.assertTrue(check_orin(_wrap(rec)), f"empty reason {bad!r} must reject")

    def test_unsupported_requires_resolvable_source_ids(self):
        self.assertTrue(check_orin(_wrap(_good_unsupported(source_ids=[]))))
        self.assertTrue(check_orin(_wrap(_good_unsupported(source_ids=["pr-does-not-exist"]))))

    def test_unsupported_rejects_any_ladder_field_value_empty_or_null(self):
        # The subtle case: an empty list [] and an explicit null still count as
        # PRESENT (membership test, not truthiness) and must be rejected.
        sample = {
            "evidence_status": "pending",
            "runtime_check": "it ran",
            "missing_check": "x",
            "artifacts": [{"kind": "binary", "sha256": H}],
            "ncu_records": [{"report": "r.ncu-rep", "sha256": H}],
        }
        for field in LADDER_FIELDS:
            for present_value in (sample[field], [], None):
                with self.subTest(field=field, value=present_value):
                    rec = _good_unsupported(**{field: present_value})
                    errs = check_orin(_wrap(rec))
                    self.assertTrue(
                        any(field in e and "documentation-only" in e for e in errs),
                        f"unsupported record carrying {field}={present_value!r} "
                        f"must be rejected as documentation-only")

    def test_invalid_applicability_rejected(self):
        rec = _good_unsupported(applicability="maybe")
        self.assertTrue(any("invalid applicability" in e for e in check_orin(_wrap(rec))))


class OrinSupportedLadderTests(unittest.TestCase):
    """Supported records still obey the evidence ladder exactly."""

    def _executed(self, **over):
        rec = {
            "id": "s", "feature": "cp-async", "targets": ["sm_87"],
            "applicability": "supported", "evidence_status": "executed",
            "source_ids": ["doc-nvidia-ampere-sm87-features"],
            "runtime_check": "exact FP32 copy/reduce equality",
            "artifacts": [{"kind": "execution-log", "name": "c.stdout", "sha256": H}],
        }
        rec.update(over)
        return rec

    def test_good_executed_record_passes(self):
        self.assertEqual(check_orin(_wrap(self._executed())), [])

    def test_executed_without_runtime_rejected(self):
        self.assertTrue(check_orin(_wrap(self._executed(runtime_check=""))))

    def test_executed_with_ncu_records_rejected(self):
        rec = self._executed(ncu_records=[{"report": "r.ncu-rep", "sha256": H}])
        self.assertTrue(check_orin(_wrap(rec)))

    def test_fabricated_pending_claiming_runtime_rejected(self):
        rec = {"id": "p", "feature": "tma", "targets": ["sm_87"],
               "evidence_status": "pending", "missing_check": "x",
               "source_ids": ["doc-nvidia-ampere-sm87-features"],
               "runtime_check": "it ran"}
        self.assertTrue(check_orin(_wrap(rec)))

    def test_pending_without_missing_check_rejected(self):
        rec = {"id": "p", "feature": "tma", "targets": ["sm_87"],
               "evidence_status": "pending",
               "source_ids": ["doc-nvidia-ampere-sm87-features"]}
        self.assertTrue(check_orin(_wrap(rec)))

    def test_profiled_without_ncu_rejected(self):
        rec = {"id": "p", "feature": "mma", "targets": ["sm_87"],
               "evidence_status": "profiled",
               "source_ids": ["doc-nvidia-ampere-sm87-features"],
               "runtime_check": "ran",
               "artifacts": [{"kind": "binary", "sha256": H}]}
        self.assertTrue(check_orin(_wrap(rec)))


class RealRegistryFilesTests(unittest.TestCase):
    def test_real_orin_registry_is_valid(self):
        self.assertEqual(
            validate.validate_orin_evidence_registry(_corpus_source_ids(), TAGS), [])

    def test_real_thor_registry_is_valid(self):
        self.assertEqual(
            validate.validate_thor_evidence_registry(_corpus_source_ids(), TAGS), [])


class ThorParityTests(unittest.TestCase):
    """The shared seam preserves Thor behavior byte-for-byte."""

    def _thor_wrap(self, rec):
        return {"probe_inventory": {"device": {"cc": "11.0"},
                                    "tool": {"nvcc": "13.2.86", "ncu": "2026.1.1"}},
                "features": [rec]}

    def test_thor_profiled_without_ncu_same_error_string(self):
        # A crafted bad Thor record produces the unchanged Thor-labelled error.
        rec = {"id": "r", "feature": "tma", "targets": ["sm_110"],
               "evidence_status": "profiled",
               "source_ids": ["doc-nvidia-thor-targets"],
               "runtime_check": "ran",
               "artifacts": [{"kind": "binary", "name": "b", "sha256": H}]}
        errs = validate.validate_thor_evidence_registry(
            {"doc-nvidia-thor-targets"}, TAGS, data=self._thor_wrap(rec))
        self.assertIn(
            "data/thor-evidence.yaml::r: profiled evidence requires ncu_records, "
            "each a mapping with a report name and a sha256 digest", errs)

    def test_thor_applicability_absent_follows_ladder(self):
        # A Thor record without applicability still passes the ladder as before.
        rec = {"id": "r", "feature": "tcgen05", "targets": ["sm_110a"],
               "evidence_status": "compiled",
               "source_ids": ["doc-nvidia-thor-targets"],
               "artifacts": [{"kind": "source", "name": "g.cu", "sha256": H}]}
        self.assertEqual(
            validate.validate_thor_evidence_registry(
                {"doc-nvidia-thor-targets"}, TAGS, data=self._thor_wrap(rec)), [])


if __name__ == "__main__":
    unittest.main()
