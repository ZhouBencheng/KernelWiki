# Target routing and validation contract

The root skill loads common principles plus the selected device route. Target
routing is curated applicability; it is separate from the existing source
architecture evidence index.

```bash
python3 scripts/query.py --target 4090 --compact
python3 scripts/query.py --target orin "pipeline" --compact
python3 scripts/get_page.py hw-target-sm110 --target thor
python3 scripts/get_page.py technique-pipeline-stages --target orin
```

- `--architecture` retains exact/family/unknown source-evidence semantics.
- `--target` and `--architecture` are intentionally mutually exclusive so a
  guidance route cannot silently narrow an evidence search.
- `--target` selects wiki pages explicitly marked `applicable_to: [portable]`
  or a list containing the selected base SM. No numeric or marketing-family
  inheritance is used. Pages without reviewed applicability stay searchable
  without `--target` but do not enter the selected implementation route.
- `data/targets.yaml` contains explicit aliases and entry IDs. Unknown names,
  unknown SMs and `a`/`f` compiler suffixes yield only portable guidance; first
  establish runtime identity rather than defaulting to B200.
- `get_page.py --target` checks a wiki page before loading it. Follow related
  wiki pages through the same target check. Plain Markdown links remain useful
  for explicit comparisons; a link alone does not change applicability.
- `--follow-sources` deliberately keeps cited raw evidence readable and labels
  it as evidence. A source can substantiate a principle without its kernel
  being runnable on the selected device.

`applicable_to` describes the scope of the whole page. Split mixed pages when
some instructions or performance claims only apply to one target. Shared
principle pages use exactly `applicable_to: [portable]` with `architectures: []`
and can honestly declare `reproducibility: concept`; implementation pages retain
their own snippet/runnable/benchmarked status. The validator's code-fence check
is not a compiler or hardware execution test.

## Identify the actual environment

Record selected GPU name, runtime CC and device ordinal, then the compiler,
assembler and framework actually building the kernel. Do not infer an `a`/`f`
compilation target from CC. Query device resource limits instead of copying
another SKU's SM count, occupancy or memory parameters.

Keep each experiment's dtype, shape, layout, math/accumulation mode, output
contract, build target, driver/runtime versions and power/clock conditions.
Performance is comparable only within a defined experimental contract. A
small-M BF16 or strict-FP32 question does not authorize an NVFP4 replacement.

## Separate evidence gates

1. **Compile:** the exact source/toolchain/target builds; retain command and hash.
2. **Correctness:** the runnable artifact passes the stated inputs and numerical
   contract on the named device. This covers only tested operations and shapes.
3. **Collection:** profiler replay succeeds and produces the expected actions;
   record missing or unsupported metrics and collector/reader versions.
4. **Benefit:** ordinary timing outside profiler replay supports a claim under
   recorded conditions. Compilation or correctness is not speedup evidence.

Preserve raw receipts with digests. A source-confidence `verified` tag means
source evidence, not that all four gates passed. Basic copy/reduce tests do not
validate matrix instructions, specialized memory or an operator's speedup.

## Maintenance

Keep sources and pinned artifacts single-copy. Add differences under
`wiki/architectures/<sm>/`; keep shared principles in `wiki/techniques/` and
shared mechanisms in `wiki/hardware/`. Do not add empty target directories or
copy a complete knowledge tree. Regenerate indices and run the full validator
after moving pages, preserving stable IDs and updating path-based links.
