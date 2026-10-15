#!/usr/bin/env python3
"""CPU-only routing coverage for Orin (sm_87) aliases and feature scope.

These are deterministic CPU routing/vocabulary checks against the public
query.py CLI and the data tables. They are NOT hardware validation: nothing here
executes CUDA, NCU, or reads a device. The synthetic query strings exercise only
the alias/rerank machinery. Run with:
    python3 -I -m unittest discover -s tests -p 'test_*.py'
"""

import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path

SKILL_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = SKILL_ROOT / "scripts"
DATA = SKILL_ROOT / "data"
QUERY = SCRIPTS / "query.py"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec.loader.exec_module(mod)
    return mod


query = _load_module("orin_query_mod", QUERY)

import yaml  # noqa: E402

TAGS = yaml.safe_load((DATA / "tags.yaml").read_text(encoding="utf-8"))


def run_query(*args):
    out = subprocess.run(
        [sys.executable, "-I", str(QUERY), *args],
        capture_output=True, text=True, check=True,
    )
    return out.stdout


def paths(*args):
    return [ln for ln in run_query(*args, "--paths-only").splitlines() if ln.strip()]


def first_result_line(text):
    for ln in text.splitlines():
        if ln.strip().startswith("["):
            return ln.strip()
    return ""


class OrinAliasResolutionTests(unittest.TestCase):
    """Orin/sm87/sm_87 resolve to the Orin device page and stay scoped to it."""

    def test_orin_keywords_rank_orin_page_first(self):
        for q in ("Orin", "Jetson Orin", "AGX Orin", "sm87", "sm_87"):
            line = first_result_line(run_query(q, "--compact"))
            self.assertIn("hw-orin", line,
                          f"{q!r} must surface the Orin device page first")

    def test_sm87_equiv_sm_87_equiv_orin(self):
        a = set(paths("--architecture", "Orin"))
        b = set(paths("--architecture", "sm87"))
        c = set(paths("--architecture", "sm_87"))
        self.assertTrue(a, "the Orin architecture set must be non-empty")
        self.assertEqual(a, b, "Orin must equal sm87")
        self.assertEqual(b, c, "sm87 must equal sm_87")
        self.assertIn("wiki/hardware/orin.md", a)
        self.assertIn("wiki/hardware/ampere-mma-async-copy.md", a)

    def test_orin_pages_do_not_leak_into_other_archs(self):
        for other in ("sm100", "sm90", "sm110"):
            s = set(paths("--architecture", other))
            self.assertNotIn("wiki/hardware/orin.md", s,
                             f"Orin page must not appear in the {other} set")
            self.assertNotIn("wiki/hardware/ampere-mma-async-copy.md", s,
                             f"Ampere sm_87 page must not appear in the {other} set")


class OrinNegativeRoutingTests(unittest.TestCase):
    """Unqualified Ampere/sm80/sm86 never resolve to Orin, and no sm87a/sm87f
    target or architecture exists."""

    def test_detect_query_archs_orin_tokens(self):
        for tok in ("Orin", "sm87", "sm_87"):
            self.assertEqual(query.detect_query_archs([tok]), {"sm87"},
                             f"{tok!r} must detect the sm87 architecture")

    def test_unqualified_ampere_sm80_sm86_do_not_detect_orin(self):
        for tok in ("Ampere", "sm80", "sm86"):
            self.assertEqual(query.detect_query_archs([tok]), set(),
                             f"{tok!r} must not silently resolve to an architecture")

    def test_orin_page_not_first_for_ampere_sm80_sm86(self):
        for q in ("Ampere", "sm80", "sm86"):
            line = first_result_line(run_query(q, "--compact"))
            self.assertNotIn("hw-orin", line,
                             f"the Orin device page must not rank first for {q!r}")

    def test_no_sm87a_sm87f_in_vocab(self):
        archs = set(TAGS.get("architectures", []))
        targets = set(TAGS.get("targets", []))
        for forbidden in ("sm87a", "sm87f", "sm_87a", "sm_87f"):
            self.assertNotIn(forbidden, archs)
            self.assertNotIn(forbidden, targets)

    def test_sm_87_is_device_identity_not_a_target_family(self):
        # sm_87's bare spelling is a device architecture, so it must route
        # through device identity, not the compiler-target-family path.
        families = query.load_target_families()
        self.assertNotIn("sm87", families)
        self.assertEqual(families.get("sm110f"), "sm_110f",
                         "the existing sm110f target family must be unchanged")


class PreservedHistoricalRerankTests(unittest.TestCase):
    """Existing keyword reranks are unchanged by the Orin addition."""

    def test_thor_and_b200_reranks_unchanged(self):
        self.assertEqual(query.detect_query_archs(["Thor"]), {"sm110"})
        self.assertEqual(query.detect_query_archs(["B200"]), {"sm100"})

    def test_b200_alias_still_equals_sm100(self):
        self.assertEqual(set(paths("--architecture", "B200")),
                         set(paths("--architecture", "sm100")))


if __name__ == "__main__":
    unittest.main()
