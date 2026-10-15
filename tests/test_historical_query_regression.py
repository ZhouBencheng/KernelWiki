#!/usr/bin/env python3
"""Regression: replay the frozen historical query baseline byte-for-byte.

`inputs/historical-query-baseline.json` freezes the complete ordered result
arrays of 21 architecture/keyword queries against the accepted immutable Thor
final delivery. This test replays each query against the candidate's own
query.py and asserts the ordered path list is identical — so the Orin additions
cannot reorder or drop any pre-existing SM90/SM100/SM110/`a`/`f` result.

The baseline lives at the controller root (outside the delivered skill tree). If
it is absent (e.g. a standalone skill checkout), the test skips rather than
failing. Run with:
    python3 -I -m unittest discover -s tests -p 'test_*.py'
"""

import json
import subprocess
import sys
import unittest
from pathlib import Path

SKILL_ROOT = Path(__file__).resolve().parents[1]
QUERY = SKILL_ROOT / "scripts" / "query.py"


def _find_baseline():
    name = "historical-query-baseline.json"
    for base in SKILL_ROOT.parents:
        cand = base / "inputs" / name
        if cand.is_file():
            return cand
    return None


BASELINE = _find_baseline()


@unittest.skipIf(BASELINE is None,
                 "historical-query-baseline.json not present (controller-root artifact)")
class HistoricalQueryRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads(BASELINE.read_text(encoding="utf-8"))

    def _replay(self, cmd):
        argv = [sys.executable, "-I", str(QUERY)]
        if cmd.get("architecture"):
            argv += ["--architecture", cmd["architecture"]]
        argv += list(cmd.get("keywords") or [])
        argv += ["--paths-only", "--limit", "100000"]
        out = subprocess.run(argv, capture_output=True, text=True, check=True)
        return [ln for ln in out.stdout.splitlines() if ln.strip()]

    def test_all_frozen_queries_match_ordered_paths(self):
        commands = self.data["commands"]
        self.assertEqual(len(commands), 21, "expected 21 frozen baseline queries")
        for cmd in commands:
            with self.subTest(case=cmd["case"]):
                got = self._replay(cmd)
                self.assertEqual(
                    got, cmd["ordered_paths"],
                    f"ordered result drift for {cmd['case']!r}")


if __name__ == "__main__":
    unittest.main()
