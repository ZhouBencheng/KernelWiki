---
name: KernelWiki
description: Find evidence-backed NVIDIA GPU kernel optimization knowledge, select guidance for the actual device, and trace techniques to upstream code and measurements. Use for kernel design, bottleneck remedies, architecture-specific CUDA/PTX/CuTe/Triton questions, or implementation references. Host-side serving and distributed-system design are outside this knowledge base.
argument-hint: "[question] | [--target device] | [page-id]"
allowed-tools: "Bash Read Grep Glob"
---

# KernelWiki — device-aware kernel optimization knowledge

Use shared principles and the selected architecture's implementation knowledge.
The retained B200/Hopper corpus, sources and pinned code remain searchable;
new target guidance does not turn their benchmark results into portable claims.

## Select the device before selecting an implementation

Establish the actual GPU name and runtime compute capability from the user's
selected machine or report. Keep compiled `a`/`f` targets separate from runtime
CC. The explicit routes are Orin/SM87, RTX 4090/SM89, H100/SM90, B200/SM100,
B300/SM103 and Thor/SM110. **Thor is not SM120.**

```bash
python3 scripts/query.py --target orin --compact
python3 scripts/query.py --target 4090 "pipeline" --compact
python3 scripts/get_page.py hw-target-sm110 --target thor
```

Read the target entry reported by the query, then only the relevant common
principles and target-specific pages. Keep `--target` on subsequent wiki
fetches. Unknown targets get portable principles only: do not substitute B200.
If no target is known, start with `pattern-target-triage` and establish identity
before recommending a specialized instruction.

The [routing and validation contract](references/target-routing.md) explains
aliases, applicability, compilation targets and device evidence. Do not load
all architecture directories or the full source corpus into context.

## Retrieve knowledge and evidence

Commands run from this skill directory. Scripts resolve their own root and
use host PyYAML or the bundled fallback; no install or network is required.

```bash
python3 scripts/query.py --target b200 --tag nvfp4 --type kernel --compact
python3 scripts/get_page.py technique-pipeline-stages --target orin
python3 scripts/get_page.py technique-sm100-pipeline-stages --target b200 --follow-sources
```

Filters include `--type`, `--tag`, `--language`, `--symptom`, `--confidence`,
`--limit`, `--compact`, `--paths-only` and `--has-code`.

For source research or explicit comparisons, `--architecture` retains its
separate exact/family/unknown evidence semantics:

```bash
python3 scripts/query.py --architecture sm110 --type pr --compact
python3 scripts/query.py --architecture blackwell --repo cutlass --limit 5
python3 scripts/get_page.py pr-cutlass-2472
python3 scripts/grep_wiki.py "tcgen05" --only wiki
```

Do not combine `--target` and `--architecture`; guidance selection and evidence
search are separate modes.

A PR's architecture mention is not a runnable compatibility claim.
`--follow-sources` labels evidence separately; it does not certify that source
code runs on the selected target. Use [the schema](references/schema.md) when
interpreting metadata. [The legacy topic map](references/primer.md) and
[generated indices](queries/by-architecture.md) cover the wider corpus and
must be filtered for the selected target before using an implementation.

## Answer with bounded evidence

- Cite page IDs and paths, and trace material claims through `sources`.
- Keep source confidence (`verified`, `source-reported`, `inferred`,
  `experimental`) separate from compilation, correctness, collection and
  performance evidence. A validator's code-fence check does not execute code.
- Report performance with GPU, dtype, shape, metric, value and source ID;
  preserve software, power/clock and timing conditions when available.
- Preserve the user's precision and numerical contract. Do not substitute
  FP4/TF32 for a BF16 or strict-FP32 workload without an explicit change in scope.
- Missing device evidence, unsupported instructions and absent profiler
  measurements are gaps, not zeros or implied support.

Source and artifact snapshots are shared once. Run `scripts/repo_status.py` for
counts; maintenance uses `scripts/generate-indices.py` and `scripts/validate.py`.
