#!/usr/bin/env python3
"""CPU-only regression tests for Thor (SM110) routing and the evidence registry.

Deterministic, no hardware. Query behavior is exercised through the public
query.py CLI; the rerank and the evidence-registry validator are also exercised
directly with synthetic inputs so the invariants are pinned independently of the
live corpus. Run with: python3 -I -m unittest discover -s tests -p 'test_*.py'
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


query = _load_module("thor_query_mod", QUERY)
validate = _load_module("thor_validate_mod", SCRIPTS / "validate.py")

import yaml  # noqa: E402

TAGS = yaml.safe_load((DATA / "tags.yaml").read_text(encoding="utf-8"))


def run_query(*args):
    """Invoke query.py in an isolated interpreter and return stdout."""
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


class ThorRoutingCLITests(unittest.TestCase):
    """AC-1 / AC-1.1: routing, distinguishable targets, preserved behavior."""

    def test_thor_bf16_ranks_thor_above_b200(self):
        out = run_query("Thor M=1 BF16 FFN linear optimization", "--compact")
        self.assertIn("hw-thor", first_result_line(out),
                      "Thor page must rank first for a Thor BF16 query")

    def test_thor_fp32_ranks_thor_above_b200(self):
        out = run_query("Thor M=1 FP32 linear keep precision", "--compact")
        self.assertIn("hw-thor", first_result_line(out),
                      "Thor page must rank first for a Thor FP32 query")

    def test_thor_page_outranks_any_sm100_only_page(self):
        # The top result must be a Thor (sm110) page, never an unqualified
        # B200/SM100-only page.
        for q in ("Thor M=1 BF16 FFN linear optimization",
                  "Thor M=1 FP32 linear keep precision"):
            line = first_result_line(run_query(q, "--compact"))
            self.assertIn("arch=sm110", line, f"top result for {q!r} must be a Thor page")

    def test_sm110_sm110a_sm110f_distinguishable(self):
        a = set(paths("--architecture", "sm110"))
        b = set(paths("--architecture", "sm110a"))
        c = set(paths("--architecture", "sm110f"))
        for s in (a, b, c):
            self.assertTrue(s, "each architecture/target set must be non-empty")
        self.assertNotEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertNotEqual(b, c)

    def test_sm110f_resolves_through_targets_not_device_identity(self):
        # sm110f resolves via compiler-target metadata: its set is a subset of
        # pages carrying the sm_110f target, and excludes sm110-only PR pages.
        c = set(paths("--architecture", "sm110f"))
        self.assertIn("wiki/hardware/thor.md", c)
        self.assertNotIn("sources/prs/flashinfer/PR-2543.md", c,
                         "an sm110 device page without the f-target must not match sm110f")

    def test_compact_shows_architecture_and_evidence(self):
        line = first_result_line(run_query("Thor target selection", "--compact"))
        self.assertIn("arch=", line)
        self.assertIn("evidence=", line)

    def test_normal_output_shows_evidence_status(self):
        out = run_query("Thor target selection", "--limit", "1")
        self.assertIn("evidence_status", out)

    def test_b200_alias_preserves_sm100_semantics(self):
        sm100 = set(paths("--architecture", "sm100"))
        b200 = set(paths("--architecture", "B200"))
        self.assertTrue(sm100)
        self.assertEqual(sm100, b200, "B200 must remain an alias of sm100")
        self.assertNotIn("wiki/hardware/thor.md", sm100,
                         "a Thor-only page must not leak into the sm100 set")

    def test_sm90_still_resolves(self):
        self.assertTrue(set(paths("--architecture", "sm90")))

    def test_bf16_fp32_not_answered_with_quantized_substitute(self):
        # The requested precision must not be answered with a quantized/TF32
        # page: the top result is the Thor page, not an NVFP4/FP4/FP8/TF32 one.
        banned = ("nvfp4", "fp4", "fp8", "tf32")
        for q in ("Thor M=1 BF16 FFN linear optimization",
                  "Thor M=1 FP32 linear keep precision"):
            line = first_result_line(run_query(q, "--compact")).lower()
            self.assertIn("hw-thor", line)
            self.assertFalse(any(b in line.split("arch=")[0] for b in banned),
                             f"top result for {q!r} must not be a quantized substitute")


class RerankUnitTests(unittest.TestCase):
    """Pin the deterministic rerank/transfer-limit logic on synthetic pages."""

    def _page(self, pid, archs):
        return {"path": pid, "fm": {"id": pid, "architectures": list(archs)}, "_score": 5}

    def test_detect_query_archs(self):
        self.assertEqual(query.detect_query_archs(["Thor"]), {"sm110"})
        self.assertEqual(query.detect_query_archs(["B200"]), {"sm100"})
        self.assertEqual(query.detect_query_archs(["FP32", "linear"]), set())
        # sm110f is a target family, not a device architecture.
        self.assertEqual(query.detect_query_archs(["sm110f"]), set())

    def test_arch_match_boosted_cross_arch_demoted(self):
        thor = self._page("thor", ["sm110", "sm110a"])
        b200 = self._page("b200", ["sm100"])
        generic = {"path": "g", "fm": {"id": "g"}, "_score": 9}  # no architectures
        pages = [thor, b200, generic]
        query.apply_architecture_rerank(pages, ["Thor"])
        # The matched page sorts first via a higher architecture tier, without
        # any mutation of the keyword score.
        self.assertGreater(thor["_arch_tier"], b200["_arch_tier"])
        self.assertEqual(thor["_score"], 5)
        self.assertTrue(b200["_transfer_limited"])
        self.assertFalse(thor["_transfer_limited"])
        self.assertFalse(generic["_transfer_limited"],
                         "a page with no architecture must not be tagged transfer-limited")

    def test_no_query_arch_leaves_scores_unchanged(self):
        thor = self._page("thor", ["sm110"])
        before = thor["_score"]
        query.apply_architecture_rerank([thor], ["linear", "optimization"])
        self.assertEqual(thor["_score"], before)
        self.assertFalse(thor["_transfer_limited"])

    def test_target_families_are_data_driven(self):
        families = query.load_target_families()
        # sm110f is a target-only family; sm110/sm110a are device archs, not here.
        self.assertEqual(families.get("sm110f"), "sm_110f")
        self.assertNotIn("sm110", families)
        self.assertNotIn("sm110a", families)


class EvidenceRegistryTests(unittest.TestCase):
    """AC-2 / AC-2.1: the registry validator accepts the real file and rejects
    malformed / dishonest records."""

    RESOLVABLE = {"doc-nvidia-thor-targets", "pr-flashinfer-1608"}

    def check(self, data):
        return validate.validate_thor_evidence_registry(self.RESOLVABLE, TAGS, data=data)

    H = "a" * 64  # a structurally valid sha256 digest

    def _good_record(self, **over):
        rec = {
            "id": "r", "feature": "tma", "targets": ["sm_110"],
            "evidence_status": "profiled",
            "source_ids": ["doc-nvidia-thor-targets"],
            "device": {"cc": "11.0"}, "tool": {"nvcc": "13.2.86", "ncu": "2026.1.1"},
            "runtime_check": "exact FP32 equality",
            "artifacts": [{"kind": "binary", "name": "probe", "sha256": self.H}],
            "ncu_records": [{"report": "probe.ncu-rep", "sha256": self.H,
                             "kernels": ["thor_tma_copy_reduce"], "sections": ["PmSampling"]}],
        }
        rec.update(over)
        return rec

    def _wrap(self, rec):
        return {"probe_inventory": {"device": {"cc": "11.0"},
                                    "tool": {"nvcc": "13.2.86", "ncu": "2026.1.1"}},
                "features": [rec]}

    def test_real_registry_is_valid(self):
        src_ids = set()
        for md in (SKILL_ROOT / "sources").rglob("*.md"):
            fm = validate.extract_frontmatter(md)
            if isinstance(fm, dict) and "id" in fm:
                src_ids.add(fm["id"])
        self.assertEqual(validate.validate_thor_evidence_registry(src_ids, TAGS), [])

    def test_good_record_passes(self):
        self.assertEqual(self.check(self._wrap(self._good_record())), [])

    def test_profiled_without_ncu_is_rejected(self):
        rec = self._good_record(ncu_records=[])
        self.assertTrue(self.check(self._wrap(rec)))

    def test_executed_without_runtime_check_is_rejected(self):
        rec = self._good_record(evidence_status="executed", runtime_check="",
                                ncu_records=None)
        self.assertTrue(self.check(self._wrap(rec)))

    def test_pending_claiming_runtime_is_rejected(self):
        rec = self._good_record(evidence_status="pending", missing_check="x")
        self.assertTrue(self.check(self._wrap(rec)))

    def test_pending_without_missing_check_is_rejected(self):
        rec = {"id": "r", "feature": "tcgen05", "targets": ["sm_110a"],
               "evidence_status": "pending",
               "source_ids": ["doc-nvidia-thor-targets"]}
        self.assertTrue(self.check(self._wrap(rec)))

    def test_missing_source_id_is_rejected(self):
        rec = self._good_record(source_ids=["pr-does-not-exist"])
        self.assertTrue(self.check(self._wrap(rec)))

    def test_device_mismatch_is_rejected(self):
        rec = self._good_record(device={"cc": "10.0"})
        self.assertTrue(self.check(self._wrap(rec)))

    def test_tool_mismatch_is_rejected(self):
        rec = self._good_record(tool={"nvcc": "12.0.0", "ncu": "2026.1.1"})
        self.assertTrue(self.check(self._wrap(rec)))

    def test_invented_feature_is_rejected(self):
        rec = self._good_record(feature="warp-teleportation")
        self.assertTrue(self.check(self._wrap(rec)))

    def test_unknown_target_is_rejected(self):
        rec = self._good_record(targets=["sm110"])  # device token, not a target
        self.assertTrue(self.check(self._wrap(rec)))

    def test_root_counterexample_compile_only_promotion_rejected(self):
        # Exact mutation from docs/root-verification-round0.md: a pending
        # compile-only MMA record promoted to profiled with a self-contradicting
        # runtime_check and a bare, non-existent ncu_records string must FAIL.
        rec = {
            "id": "thor-tcgen05-mma", "feature": "tcgen05", "targets": ["sm_110a"],
            "evidence_status": "profiled",
            "source_ids": ["doc-nvidia-thor-targets"],
            "runtime_check": "Only tcgen05.fence compiled; no MMA execution was performed.",
            "ncu_records": ["nonexistent-mma-report.ncu-rep"],
        }
        self.assertTrue(self.check(self._wrap(rec)),
                        "a compile-only claim promoted to profiled must be rejected")

    def test_bare_string_ncu_record_rejected(self):
        rec = self._good_record(ncu_records=["probe.ncu-rep"])  # no sha256 mapping
        self.assertTrue(self.check(self._wrap(rec)))

    def test_hashless_artifact_rejected(self):
        rec = self._good_record(artifacts=[{"kind": "binary", "name": "probe"}])
        self.assertTrue(self.check(self._wrap(rec)))

    def test_profiled_without_artifacts_rejected(self):
        rec = self._good_record(artifacts=None)
        self.assertTrue(self.check(self._wrap(rec)))

    def test_compiled_requires_artifacts(self):
        rec = {"id": "r", "feature": "tcgen05", "targets": ["sm_110a"],
               "evidence_status": "compiled",
               "source_ids": ["doc-nvidia-thor-targets"]}
        self.assertTrue(self.check(self._wrap(rec)))

    def test_compiled_must_not_claim_runtime(self):
        rec = {"id": "r", "feature": "tcgen05", "targets": ["sm_110a"],
               "evidence_status": "compiled",
               "source_ids": ["doc-nvidia-thor-targets"],
               "artifacts": [{"kind": "source", "name": "gate.cu", "sha256": self.H}],
               "runtime_check": "it ran"}
        self.assertTrue(self.check(self._wrap(rec)))

    def test_good_compiled_record_passes(self):
        rec = {"id": "r", "feature": "tcgen05", "targets": ["sm_110a"],
               "evidence_status": "compiled",
               "source_ids": ["doc-nvidia-thor-targets"],
               "artifacts": [{"kind": "source", "name": "gate.cu", "sha256": self.H}]}
        self.assertEqual(self.check(self._wrap(rec)), [])

    def test_profiled_ncu_without_kernels_or_sections_rejected(self):
        rec = self._good_record(
            ncu_records=[{"report": "probe.ncu-rep", "sha256": self.H}])  # no kernels/sections
        self.assertTrue(self.check(self._wrap(rec)))

    def test_profiled_ncu_string_kernels_returns_structural_error_not_crash(self):
        # BSI-3: a malformed receipt whose kernels/sections is a string (not a
        # list) must yield a clean structural error, never an uncaught TypeError
        # from str + list concatenation that crashes validate.py main().
        rec = self._good_record(
            ncu_records=[{"report": "probe.ncu-rep", "sha256": self.H,
                          "kernels": "thor_tma_copy_reduce",  # a string, not a list
                          "sections": ["PmSampling"]}])
        try:
            errs = self.check(self._wrap(rec))
        except TypeError as exc:  # pragma: no cover - the bug this guards against
            self.fail(f"malformed kernels shape raised TypeError instead of a "
                      f"structural error: {exc}")
        self.assertTrue(any("must be a list" in e for e in errs),
                        "string kernels must produce a shape structural error")

    def test_profiled_receipt_missing_declaration_rejected(self):
        # BSI-3: per-receipt declaration — a second cited receipt with no
        # kernels/sections cannot ride on the first receipt's declaration.
        rec = self._good_record(ncu_records=[
            {"report": "a.ncu-rep", "sha256": self.H,
             "kernels": ["thor_tma_copy_reduce"], "sections": ["PmSampling"]},
            {"report": "b.ncu-rep", "sha256": self.H}])  # undeclared
        errs = self.check(self._wrap(rec))
        self.assertTrue(any("must declare its covered" in e for e in errs),
                        "an undeclared second receipt must be rejected")

    def test_tma_receipt_relabelled_as_tcgen05_rejected(self):
        # Case-2 shape: real-looking TMA receipt (kernels name TMA) but the
        # record claims the tcgen05 feature — the feature is not reflected.
        rec = self._good_record(feature="tcgen05", targets=["sm_110a"])
        self.assertTrue(self.check(self._wrap(rec)),
                        "a TMA receipt relabelled as tcgen05 must be rejected")

    def test_root_counterexample_v2_cases_rejected(self):
        # The exact two cases from analysis/registry-round0-counterexample-v2.json:
        # fabricated zero-hash receipts, and real TMA receipts relabelled as MMA.
        # Neither cites tcgen05 kernels/sections, so the feature binding rejects
        # both at the CPU gate (byte existence remains the root-run gate).
        zero = "0" * 64
        real_report = "4e013885c997fc30752e72795f12570339d16c456ea7087e25982a1a07c08953"
        real_exec = "19b8960f75ecb739b53c1ff0d09fce204df76aee56aa9479ef17f46a8dc06067"
        fabricated = {
            "id": "thor-tcgen05-mma", "feature": "tcgen05", "targets": ["sm_110a"],
            "evidence_status": "profiled", "source_ids": ["doc-nvidia-thor-targets"],
            "runtime_check": "Only tcgen05.fence compiled; no MMA execution was performed.",
            "artifacts": [{"kind": "execution", "path": "nonexistent.json", "sha256": zero}],
            "ncu_records": [{"report": "nonexistent-mma-report.ncu-rep", "sha256": zero}],
        }
        relabelled = {
            "id": "thor-tcgen05-mma", "feature": "tcgen05", "targets": ["sm_110a"],
            "evidence_status": "profiled", "source_ids": ["doc-nvidia-thor-targets"],
            "runtime_check": "Only tcgen05.fence compiled; no MMA execution was performed.",
            "artifacts": [{"kind": "execution", "path": "validation-check/manifest.json",
                           "sha256": real_exec}],
            "ncu_records": [{"report": "validation-profile/probe-lineinfo.ncu-rep",
                             "sha256": real_report}],
        }
        self.assertTrue(self.check(self._wrap(fabricated)))
        self.assertTrue(self.check(self._wrap(relabelled)))


class ProbeStagingTests(unittest.TestCase):
    """Staged probe sources exist and match the evidence-registry digests."""

    def test_probe_sources_match_registry_digests(self):
        import hashlib
        reg = yaml.safe_load((DATA / "thor-evidence.yaml").read_text())
        probes = SKILL_ROOT / "probes"
        self.assertTrue((probes / "probe-manifest.yaml").is_file())
        # Collect declared source digests from the registry artifacts.
        declared = {}
        for feat in reg["features"]:
            for art in feat.get("artifacts", []) or []:
                name = (art.get("name") or "").split("/")[-1]
                if name.endswith(".cu"):
                    declared[name] = art["sha256"]
        self.assertIn("thor_probe.cu", declared)
        for name, sha in declared.items():
            staged = probes / name
            self.assertTrue(staged.is_file(), f"probe source {name} not staged")
            got = hashlib.sha256(staged.read_bytes()).hexdigest()
            self.assertEqual(got, sha, f"staged {name} digest must match the registry")


if __name__ == "__main__":
    unittest.main()
