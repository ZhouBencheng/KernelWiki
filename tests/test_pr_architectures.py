#!/usr/bin/env python3
"""Offline page-generation regression checks; no hardware validation."""

import importlib.util
from pathlib import Path
import unittest

import yaml


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "generate-pr-pages.py"
SPEC = importlib.util.spec_from_file_location("generate_pr_pages", SCRIPT)
GENERATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GENERATOR)


class ArchitectureTests(unittest.TestCase):
    def architectures(self, title, body="", files=()):
        page = GENERATOR.generate_page(
            "example/kernels",
            {
                "number": 1,
                "title": title,
                "body": body,
                "user": {"login": "fixture"},
                "created_at": "2026-01-01T00:00:00Z",
                "html_url": "https://example.invalid/pull/1",
            },
            files,
            "regression fixture",
            "2026-01-02",
        )
        return yaml.safe_load(page.split("---", 2)[1])["architectures"]

    def test_thor_title_and_body(self):
        self.assertEqual(self.architectures("Thor support"), ["sm110"])
        self.assertEqual(self.architectures("Device support", "NVIDIA Thor"), ["sm110"])

    def test_targets_stay_distinct(self):
        for target, expected in (("SM110", "sm110"), ("sm_110", "sm110"),
                                 ("SM110a", "sm110a"), ("sm_110a", "sm110a")):
            with self.subTest(target=target):
                self.assertEqual(self.architectures(target), [expected])
        self.assertEqual(self.architectures("Thor", "Build for sm_110a"),
                         ["sm110", "sm110a"])

    def test_virtual_targets_in_title_body_and_paths(self):
        for target, expected in (("compute_110", "sm110"),
                                 ("compute_110a", "sm110a")):
            with self.subTest(target=target):
                self.assertEqual(self.architectures(f"Enable {target} support"),
                                 [expected])
                self.assertEqual(self.architectures("Enable target", f"Build for {target}"),
                                 [expected])
                self.assertEqual(self.architectures("Add kernel", files=[f"csrc/gemm_{target}.cu"]),
                                 [expected])
        self.assertEqual(self.architectures("COMPUTE_110A support"), ["sm110a"])

    def test_virtual_target_boundaries(self):
        for target in ("compute_1100", "compute_110a0", "compute_110ab",
                       "compute_110b", "xcompute_110", "xcompute_110a", "compute110"):
            with self.subTest(target=target):
                self.assertEqual(self.architectures(target), ["sm100"])

    def test_mixed_virtual_target_evidence(self):
        self.assertEqual(self.architectures("compute_110 and compute_110a"),
                         ["sm110", "sm110a"])
        self.assertEqual(self.architectures("B200 and compute_110a", "Hopper support"),
                         ["sm100", "sm110a", "sm90"])
        self.assertEqual(self.architectures("Thor", "Build for compute_110a"),
                         ["sm110", "sm110a"])

    def test_changed_paths_supply_architecture_evidence(self):
        self.assertEqual(self.architectures("Add kernel", files=["csrc/gemm_sm110a.cu"]),
                         ["sm110a"])
        self.assertEqual(self.architectures("Add kernel", files=["csrc/thor/gemm.cu"]),
                         ["sm110"])

    def test_existing_sm100_family_and_default(self):
        for title in ("SM100", "sm_100a", "Blackwell", "B100", "B200", "GB200",
                      "GB300", "GEMM optimization", "Author changes", "sm1100"):
            with self.subTest(title=title):
                self.assertEqual(self.architectures(title), ["sm100"])

    def test_mixed_evidence_survives(self):
        self.assertEqual(self.architectures("SM100 and Thor support", "Hopper support"),
                         ["sm100", "sm110", "sm90"])
        self.assertEqual(self.architectures("Thor support", files=["csrc/gemm_sm100.cu"]),
                         ["sm100", "sm110"])
        self.assertEqual(self.architectures("B200 and sm_110a support"),
                         ["sm100", "sm110a"])
        self.assertEqual(self.architectures("Blackwell Thor support"),
                         ["sm100", "sm110"])

    def test_device_capability_does_not_imply_architecture_specific_target(self):
        self.assertEqual(self.architectures("Thor", "Device CC 11.0"), ["sm110"])
        self.assertNotIn("sm110a", self.architectures("Device CC 11.0"))


if __name__ == "__main__":
    unittest.main()
