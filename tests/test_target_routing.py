import hashlib
import json
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import query
import get_page
import target_routing
import validate
from _yaml_compat import yaml


def options(target=None, architecture=None):
    return SimpleNamespace(target=target, architecture=architecture, type=None,
                           tag=None, repo=None, language=None, symptom=None,
                           confidence=None, has_code=False)


class TargetRoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pages = query.load_all_pages()

    def results(self, target):
        return {p["fm"]["id"]: p for p in query.filter_pages(self.pages, options(target))}

    def test_product_aliases_are_explicit_and_suffixes_not_device_identities(self):
        for name, expected in (("Jetson AGX Orin", "sm87"), ("agx-orin", "sm87"),
                               ("RTX 4090", "sm89"), ("sm_89", "sm89"),
                               ("Thor", "sm110"), ("11.0", "sm110"),
                               ("B200", "sm100"), ("B300", "sm103")):
            self.assertEqual(expected, target_routing.resolve_target(name))
        for unknown in ("sm110a", "sm_100f", "Blackwell", "sm120", "new GPU"):
            self.assertIsNone(target_routing.resolve_target(unknown))
            self.assertTrue(self.results(unknown))
            self.assertTrue(all(p["fm"]["applicable_to"] == ["portable"]
                                for p in self.results(unknown).values()))

    def test_orin_and_ada_get_shared_principles_without_sm100_implementations(self):
        for target in ("orin", "4090"):
            pages = self.results(target)
            self.assertIn("hw-async-copy-warp-mma", pages)
            self.assertIn("technique-pipeline-stages", pages)
            self.assertNotIn("technique-sm100-pipeline-stages", pages)
            self.assertNotIn("hw-tmem", pages)
            self.assertNotIn("hw-tma", pages)
            self.assertTrue(all(p["fm"].get("type") for p in pages.values()))

    def test_thor_reuses_reviewed_mechanisms_without_b200_specific_examples(self):
        pages = self.results("thor")
        for page_id in ("hw-target-sm110", "hw-tcgen05-mma", "hw-tmem", "hw-tma"):
            self.assertIn(page_id, pages)
        self.assertNotIn("hw-sm100-tmem", pages)
        self.assertNotIn("technique-sm100-persistent-kernels", pages)
        self.assertNotIn("lang-ptx", pages)

    def test_b200_keeps_original_case_studies_and_specific_implementations(self):
        pages = self.results("b200")
        for page_id in ("kernel-flash-attention-4", "kernel-nvfp4-gemm", "lang-ptx",
                        "hw-sm100-tmem", "technique-sm100-pipeline-stages",
                        "technique-sm100-persistent-kernels", "kernel-sm100-deepgemm",
                        "technique-sm100-warp-specialization", "lang-sm100-triton"):
            self.assertIn(page_id, pages)
        self.assertEqual("wiki/architectures/sm100/ptx.md", pages["lang-ptx"]["path"])

    def test_hopper_gets_valid_shared_knowledge_but_not_sm100_recipes(self):
        pages = self.results("h100")
        for page_id in ("kernel-deepgemm", "kernel-fp8-block-scale-gemm",
                        "kernel-grouped-gemm", "kernel-flashmla",
                        "technique-warp-specialization", "lang-triton"):
            self.assertIn(page_id, pages)
        for page_id in ("kernel-sm100-deepgemm", "kernel-sm100-fp8-block-scale-gemm",
                        "technique-sm100-warp-specialization", "lang-sm100-triton"):
            self.assertNotIn(page_id, pages)

        allowed = subprocess.run(
            [sys.executable, str(ROOT / "scripts/get_page.py"),
             "kernel-deepgemm", "--target", "h100", "--frontmatter-only"],
            capture_output=True, text=True,
        )
        blocked = subprocess.run(
            [sys.executable, str(ROOT / "scripts/get_page.py"),
             "kernel-sm100-deepgemm", "--target", "h100"],
            capture_output=True, text=True,
        )
        self.assertEqual(0, allowed.returncode, allowed.stderr)
        self.assertEqual(2, blocked.returncode)
        self.assertEqual("", blocked.stdout)

    def test_shared_deepgemm_dispatch_matches_retained_source_semantics(self):
        retained = (
            ROOT / "artifacts/prs/DeepGEMM/PR-304/key-files/csrc/apis/gemm.hpp"
        ).read_text(encoding="utf-8")
        dispatch = retained.split("    // Dispatch into different implements\n", 1)[1]
        dispatch = textwrap.dedent(
            dispatch.split("\n}\n\nstatic void fp8_fp4_gemm_nn", 1)[0]
        ).rstrip()

        self.assertIn("const int gran_n", dispatch)
        self.assertIn("sm90_fp8_gemm_1d1d", dispatch)
        self.assertIn("sm90_fp8_gemm_1d2d", dispatch)
        self.assertIn("DG_HOST_UNREACHABLE", dispatch)
        for relative in (
            "wiki/kernels/deepgemm.md",
            "wiki/kernels/fp8-block-scale-gemm.md",
        ):
            page = (ROOT / relative).read_text(encoding="utf-8")
            self.assertIn(dispatch, page, relative)

    def test_target_selected_artifact_expansion_is_labeled_unfiltered_evidence(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts/get_page.py"),
             "kernel-deepgemm", "--target", "h100", "--include-code"],
            capture_output=True, text=True,
        )
        self.assertEqual(0, result.returncode, result.stderr)
        label = "Unfiltered source/comparison evidence"
        self.assertIn(label, result.stdout)
        self.assertIn("sm100_fp8_gemm_1d1d.cuh", result.stdout)
        self.assertLess(result.stdout.index(label), result.stdout.index("sm100_fp8_gemm_1d1d.cuh"))
        self.assertIn("other architectures may be present", result.stdout)

    def test_source_architecture_evidence_does_not_grant_guidance_applicability(self):
        source = {"path": "sources/prs/example/PR-1.md", "body": "",
                  "fm": {"id": "source", "architectures": ["sm87"],
                         "architecture_disposition": "exact"}}
        self.assertEqual([source], query.filter_pages([source], options(architecture="sm87")))
        self.assertEqual([], query.filter_pages([source], options(target="sm87")))
        self.assertFalse(target_routing.applicable_to({"type": "hardware", "architectures": ["sm110"]}, "thor"))

    def test_target_fetch_blocks_cross_architecture_implementation(self):
        result = subprocess.run([sys.executable, str(ROOT / "scripts/get_page.py"),
                                 "technique-sm100-pipeline-stages", "--target", "orin"],
                                capture_output=True, text=True)
        self.assertEqual(2, result.returncode)
        self.assertEqual("", result.stdout)

    def test_target_and_architecture_filters_are_separate_modes(self):
        result = subprocess.run([sys.executable, str(ROOT / "scripts/query.py"),
                                 "--target", "thor", "--architecture", "sm110"],
                                capture_output=True, text=True)
        self.assertEqual(2, result.returncode)
        self.assertIn("not allowed with argument", result.stderr)

    def test_unknown_target_fetch_discloses_portable_only_mode(self):
        result = subprocess.run([sys.executable, str(ROOT / "scripts/get_page.py"),
                                 "technique-pipeline-stages", "--target", "sm_110a"],
                                capture_output=True, text=True)
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn("Unknown device target", result.stderr)

    def test_selected_fetch_preserves_cited_evidence(self):
        result = subprocess.run([sys.executable, str(ROOT / "scripts/get_page.py"),
                                 "technique-pipeline-stages", "--target", "orin", "--follow-sources"],
                                capture_output=True, text=True)
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn("doc-cuda-portable-practices", result.stdout)
        self.assertNotIn("695.43", result.stdout)
        result = subprocess.run([sys.executable, str(ROOT / "scripts/get_page.py"),
                                 "doc-ptx-isa-sm100", "--target", "orin", "--frontmatter-only"],
                                capture_output=True, text=True)
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn("Source evidence only", result.stderr)

    def test_declared_targets_and_entry_pages_are_valid(self):
        for sm, profile in target_routing.targets().items():
            path = get_page.find_page(profile["page"])
            self.assertIsNotNone(path)
            fm, _ = get_page.split_frontmatter(path.read_text())
            self.assertEqual([sm], fm["applicable_to"])
        for page in self.pages:
            self.assertEqual([], target_routing.applicability_errors(page["fm"]), page["path"])
            scope = page["fm"].get("applicable_to")
            if isinstance(scope, list) and len(scope) == 1 and scope[0] in target_routing.targets():
                self.assertTrue(
                    page["path"].startswith(f"wiki/architectures/{scope[0]}/"),
                    page["path"],
                )
        for bad in (["sm110a"], ["portable", "sm100"], ["sm87", "sm87"], []):
            self.assertTrue(target_routing.applicability_errors({"type": "technique", "applicable_to": bad}))
        self.assertTrue(target_routing.applicability_errors({
            "type": "technique", "architectures": ["sm100"],
            "applicable_to": ["portable"],
        }))
        self.assertEqual([], target_routing.target_registry_errors())

    def test_portable_concept_exception_does_not_weaken_implementation_gate(self):
        schemas = yaml.safe_load((ROOT / "data/schemas.yaml").read_text())
        tags = yaml.safe_load((ROOT / "data/tags.yaml").read_text())
        sources = {p["fm"]["id"] for p in self.pages if p["path"].startswith("sources/")}
        common = ROOT / "wiki/techniques/pipeline-stages.md"
        self.assertEqual([], validate.validate_file(common, schemas, tags, sources, {"cpp"}))
        portable_kernel = ROOT / "wiki/kernels/fused-moe.md"
        self.assertEqual([], validate.validate_file(
            portable_kernel, schemas, tags, sources, {"cpp"}
        ))
        with tempfile.TemporaryDirectory(dir=ROOT / "wiki") as tmp:
            test_page = Path(tmp) / "concept.md"
            test_page.write_text(common.read_text().replace("applicable_to: [portable]", "applicable_to: [sm100]"))
            errors = validate.validate_file(test_page, schemas, tags, sources, {"cpp"})
            self.assertTrue(any("below minimum" in error for error in errors), errors)
        with tempfile.TemporaryDirectory(dir=ROOT / "wiki/architectures/sm100") as tmp:
            test_page = Path(tmp) / "kernel-concept.md"
            test_page.write_text(
                portable_kernel.read_text()
                .replace("architectures: []", "architectures: [sm100]")
                .replace("applicable_to: [portable]", "applicable_to: [sm100]")
            )
            errors = validate.validate_file(test_page, schemas, tags, sources, {"cpp"})
            self.assertTrue(any("below minimum" in error for error in errors), errors)

    def test_tma_routes_keep_their_required_shared_barrier_mechanism(self):
        for target in ("h100", "b200", "b300", "thor"):
            pages = self.results(target)
            self.assertIn("hw-tma", pages)
            self.assertIn("hw-mbarrier", pages)


class DeviceReceiptTests(unittest.TestCase):
    def test_retained_probe_sources_match_all_native_execution_receipts(self):
        for sm in ("sm87", "sm89", "sm110"):
            receipt = json.loads((ROOT / f"evidence/multiarch-2026-10-27/{sm}.json").read_text())
            device = receipt["runtime_device"]
            self.assertEqual(sm, f"sm{device['cc_major']}{device['cc_minor']}")
            self.assertEqual(sm.replace("sm", "sm_"), receipt["compiled_target"])
            for source in receipt["sources"]:
                path = ROOT / "probes/multiarch" / source["file"]
                self.assertEqual(source["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
                self.assertEqual("passed", source["build_result"])
                self.assertEqual("passed", source["run_result"])
            self.assertTrue(receipt["generic_correctness"]["vector_copy"]["correct"])
            self.assertTrue(receipt["generic_correctness"]["block_reduce"]["correct"])
            self.assertTrue(receipt["generic_correctness"]["matrix_multiply"]["correct"])
            self.assertTrue(receipt["instruction_correctness"]["cp_async_4byte_copy"]["correct"])
            self.assertTrue(receipt["instruction_correctness"]["wmma_bf16_fp32"]["correct"])
            self.assertIsNone(receipt["scope"]["performance_claim"])
            self.assertFalse(receipt["scope"]["clock_or_power_changes"])
            self.assertIn("tcgen05", receipt["scope"]["not_proven"])
            self.assertIn("optimization speedup", receipt["scope"]["not_proven"])
            for profile in receipt["profiling"]:
                self.assertFalse((ROOT / "evidence/multiarch-2026-10-27" / profile["file"]).exists())

    def test_no_new_b200_live_receipt_is_implied(self):
        self.assertFalse((ROOT / "evidence/multiarch-2026-10-27/sm100.json").exists())


if __name__ == "__main__":
    unittest.main()
