#!/usr/bin/env python3
"""Unified query tool for the Blackwell kernel wiki.

Supports natural-language keyword queries, tag filters, repo filters, and type filters.

Usage:
    query.py "how to fuse gate-up GEMM"
    query.py --tag nvfp4 --type kernel
    query.py --repo cutlass --limit 20
    query.py --language cute-dsl
    query.py --symptom memory-bound

Returns a ranked list of matching pages with titles, paths, and key frontmatter fields.
"""

import argparse
import re
import sys
import yaml
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _wiki_root import WIKI_ROOT  # noqa: E402


_ALIAS_CACHE = None


def load_alias_expansions():
    """Return a dict mapping lowercased alias → canonical term, loaded from data/aliases.yaml.

    Each canonical is also mapped to itself, so lookups always return something
    sensible when the user already types the canonical form.
    """
    global _ALIAS_CACHE
    if _ALIAS_CACHE is not None:
        return _ALIAS_CACHE
    out = {}
    aliases_path = WIKI_ROOT / "data" / "aliases.yaml"
    try:
        raw = yaml.safe_load(aliases_path.read_text(encoding="utf-8")) or {}
    except Exception:
        _ALIAS_CACHE = {}
        return _ALIAS_CACHE
    for canonical, variants in raw.items():
        if not isinstance(canonical, str):
            continue
        out.setdefault(canonical.lower(), canonical)
        for v in (variants or []):
            if isinstance(v, str):
                out.setdefault(v.lower(), canonical)
    _ALIAS_CACHE = out
    return out


def expand_keyword(kw):
    """Return a list of search variants for a keyword: the original plus any
    canonical term the alias map points to."""
    aliases = load_alias_expansions()
    canonical = aliases.get(kw.lower())
    if canonical and canonical.lower() != kw.lower():
        return [kw, canonical]
    return [kw]


_TAGS_CACHE = None


def _load_tags():
    """Return data/tags.yaml parsed once (the controlled vocabulary)."""
    global _TAGS_CACHE
    if _TAGS_CACHE is not None:
        return _TAGS_CACHE
    try:
        _TAGS_CACHE = yaml.safe_load(
            (WIKI_ROOT / "data" / "tags.yaml").read_text(encoding="utf-8")) or {}
    except Exception:
        _TAGS_CACHE = {}
    return _TAGS_CACHE


def load_arch_canonicals():
    """Return the set of device-architecture canonical names (data/tags.yaml).

    These are the tokens a page may legitimately carry in its `architectures:`
    field; a compiler-target family such as `sm110f` is deliberately excluded.
    """
    return {str(a).lower() for a in (_load_tags().get("architectures") or [])}


def load_target_families():
    """Return {query-canonical: page-target-token} for compiler-target families
    that have NO device-architecture equivalent, e.g. {'sm110f': 'sm_110f'}.

    Derived from the `targets` and `architectures` vocabularies so a future
    family (sm120f, ...) is picked up without code changes: a target whose bare
    spelling is also a device architecture (sm_110 -> sm110) routes through
    device identity, not the separate compiler-target field.
    """
    arch_canonicals = load_arch_canonicals()
    families = {}
    for tgt in (_load_tags().get("targets") or []):
        canonical = str(tgt).replace("_", "").lower()  # sm_110f -> sm110f
        if canonical not in arch_canonicals:
            families[canonical] = str(tgt).lower()
    return families


def detect_query_archs(keywords):
    """Return the set of device-architecture canonicals implied by free-text
    keywords (e.g. 'Thor' -> 'sm110', 'B200' -> 'sm100').

    Only device architectures are returned; a compiler-target family token such
    as 'sm110f' is a target, not a device identity, and is handled by the
    explicit --architecture filter instead of this keyword-driven rerank.
    """
    aliases = load_alias_expansions()
    arch_canonicals = load_arch_canonicals()
    found = set()
    for kw in keywords:
        canonical = aliases.get(kw.lower(), kw.lower())
        if canonical in arch_canonicals:
            found.add(canonical)
    return found


def load_frontmatter(path):
    """Parse YAML frontmatter from a markdown file. Returns (fm_dict, body_str) or (None, None)."""
    try:
        content = path.read_text(encoding="utf-8")
    except Exception:
        return None, None
    m = re.match(r'^---\s*\r?\n(.*?)\r?\n---\s*\r?\n(.*)', content, re.DOTALL)
    if not m:
        return None, None
    try:
        fm = yaml.safe_load(m.group(1))
        if not isinstance(fm, dict):
            return None, None
        return fm, m.group(2)
    except yaml.YAMLError:
        return None, None


def load_all_pages():
    """Load frontmatter + body summary for every sources/*.md and wiki/*.md file."""
    pages = []
    for subdir in ["sources", "wiki"]:
        base = WIKI_ROOT / subdir
        if not base.exists():
            continue
        for md in base.rglob("*.md"):
            fm, body = load_frontmatter(md)
            if fm is None:
                continue
            pages.append({
                "path": str(md.relative_to(WIKI_ROOT)),
                "fm": fm,
                "body": body or "",
            })
    return pages


def detect_page_type(fm, path):
    """Return a page-type label for filtering: source-pr/source-blog/..., wiki-hardware/..."""
    if "type" in fm:
        return f"wiki-{fm['type']}"
    parts = path.split("/")
    if parts[0] == "sources" and len(parts) > 1:
        return f"source-{parts[1].rstrip('s')}"  # prs → source-pr
    return "unknown"


def score_keyword_match(fm, body, keywords):
    """Score a page by keyword matches in title, tags, and body (title > tags > body).

    Each user keyword is expanded through the alias map so that typing 'UMMA'
    also scores 'tcgen05', 'B200' also scores 'sm100', etc.
    """
    score = 0
    title_text = str(fm.get("title", "")).lower()
    tag_text = " ".join(
        str(v) for k in ("tags", "techniques", "hardware_features", "kernel_types",
                          "languages", "aliases", "symptoms")
        for v in (fm.get(k) or [])
    ).lower()
    body_lower = body.lower()
    for kw in keywords:
        best_variant_score = 0
        for variant in expand_keyword(kw):
            v_l = variant.lower()
            variant_score = 0
            if v_l in title_text:
                variant_score += 10
            if v_l in tag_text:
                variant_score += 5
            body_hits = body_lower.count(v_l)
            variant_score += min(body_hits, 3)
            if variant_score > best_variant_score:
                best_variant_score = variant_score
        score += best_variant_score
    return score


def apply_architecture_rerank(pages, keywords):
    """Assign a deterministic architecture tier so pages matching the query's
    device architecture sort above cross-architecture references.

    Pure post-filter: no page is removed and keyword scores are untouched. When
    the query names an architecture (e.g. 'Thor' -> sm110), a page whose
    `architectures:` intersects that set gets `_arch_tier = 1`; an
    architecture-bearing page targeting a *different* architecture is kept at
    tier 0 but marked `_transfer_limited` so its cross-architecture provenance
    stays explicit. The tier is a sort key (see main), not a score mutation, so
    it can never be overtaken by a keyword-dense page. A page carrying no
    architecture, or a query naming none, stays at tier 0.
    """
    query_archs = detect_query_archs(keywords)
    for p in pages:
        p["_transfer_limited"] = False
        p["_arch_tier"] = 0
    if not query_archs:
        return
    for p in pages:
        page_archs = {str(a).lower() for a in (p["fm"].get("architectures") or [])}
        if page_archs & query_archs:
            p["_arch_tier"] = 1
        elif page_archs:
            p["_transfer_limited"] = True


def filter_pages(pages, args):
    """Apply CLI filters to narrow the page set."""
    # Resolve the --architecture filter once (constant across all pages): a
    # compiler-target family routes against each page's `targets:` field, every
    # other token against the device `architectures:` field.
    arch_filter = None
    if args.architecture:
        tok = args.architecture.strip().lower()
        canonical = load_alias_expansions().get(tok, tok)
        families = load_target_families()
        if canonical in families:
            arch_filter = ("target", families[canonical])
        else:
            arch_filter = ("arch", {v.lower() for v in expand_keyword(args.architecture)})

    out = []
    for p in pages:
        fm = p["fm"]
        path = p["path"]
        ptype = detect_page_type(fm, path)
        p["_ptype"] = ptype

        if args.type and not ptype.endswith(args.type):
            # --type kernel matches wiki-kernel, --type pr matches source-pr
            if ptype != args.type:
                continue

        if args.tag:
            all_tags = set()
            for k in ("tags", "techniques", "hardware_features", "kernel_types", "languages"):
                all_tags.update(fm.get(k) or [])
            tag_variants = {v.lower() for v in expand_keyword(args.tag)}
            if not any(t.lower() in tag_variants for t in all_tags):
                continue

        if args.repo:
            repo = str(fm.get("repo", "")).lower()
            if args.repo.lower() not in repo:
                continue

        if args.language:
            langs = set(fm.get("languages") or [])
            tags = set(fm.get("tags") or [])
            if args.language not in langs and args.language not in tags:
                continue

        if arch_filter:
            kind, want = arch_filter
            if kind == "target":
                # Compiler-target family: match the page's `targets:` metadata,
                # never its device `architectures:` identity (AC-1.1).
                targets = {str(t).lower() for t in (fm.get("targets") or [])}
                if want not in targets:
                    continue
            else:
                archs = {a.lower() for a in (fm.get("architectures") or [])}
                if not (archs & want):
                    continue

        if args.symptom:
            symptoms = set(fm.get("symptoms") or [])
            if args.symptom not in symptoms:
                continue

        if args.confidence:
            if str(fm.get("confidence", "")) != args.confidence:
                continue

        if args.has_code:
            # NOTE: `.patch` is deliberately excluded. --has-code is documented
            # as "at least one source file", so a PR bundle that only shipped
            # diff.patch (no captured key-files/) should NOT surface here —
            # a raw diff is not browsable source. See: get_page.py --include-code
            # still prints diff.patch for bundles that have one.
            #
            # `.txt` IS counted: scripts/extract_blog_code.py writes unlabeled
            # fenced blocks under artifacts/blogs/<slug>/code/NN-<name>.txt via
            # the EXT_MAP.get(lang, "txt") fallback. These are real code
            # snippets whose fence just lacked a language tag (R20 widening).
            #
            # `.sh`, `.yaml`, `.json` are also counted (R33) to match
            # extract_blog_code.py's EXT_MAP mappings for bash / shell /
            # yaml / json fences. Keep this set in sync with
            # validate.py's ASSET_SOURCE_EXTS and get_page.py's
            # --include-code exts; the three together form the Phase-3
            # asset-source contract.
            exts = {".cu", ".cuh", ".ptx", ".py", ".cpp", ".h", ".hpp", ".inl",
                    ".pyx", ".cxx", ".cc", ".txt",
                    ".sh", ".yaml", ".json"}
            candidate_dirs = []

            # Primary: explicit page-level artifact_dir.
            ad = fm.get("artifact_dir")
            if ad:
                candidate_dirs.append(WIKI_ROOT / ad)

            # Fallback 1: conventional bundle locations per page type.
            # A source-blog's extracted code lives at artifacts/blogs/<slug>/code/;
            # a source-contest's reconstructed bundles live under
            # artifacts/contests/<contest>/<problem>/submissions/**.
            # This lets --has-code still work for pages whose artifact_dir
            # hasn't been backfilled yet (Codex R8 finding #1).
            slug = str(fm.get("id", "")).replace("blog-", "")
            if p["_ptype"] == "source-blog":
                candidate_dirs.append(WIKI_ROOT / "artifacts" / "blogs" / Path(path).stem / "code")
            elif p["_ptype"] == "source-contest":
                contest_dir_name = Path(path).parent.name
                problem_stem = Path(path).stem
                candidate_dirs.append(WIKI_ROOT / "artifacts" / "contests" / contest_dir_name / problem_stem)

            # Fallback 2: for source-pr pages, infer the conventional
            # artifacts/prs/<repo-short>/PR-<N>/ location from frontmatter
            # repo+pr fields.
            if p["_ptype"] == "source-pr" and fm.get("repo") and fm.get("pr"):
                repo_short = str(fm["repo"]).split("/")[-1].lower()
                candidate_dirs.append(WIKI_ROOT / "artifacts" / "prs" / repo_short / f"PR-{fm['pr']}")

            has_any = False
            for cand in candidate_dirs:
                if not cand.is_dir():
                    continue
                for f in cand.rglob("*"):
                    if f.is_file() and f.suffix.lower() in exts:
                        has_any = True
                        break
                if has_any:
                    break

            if not has_any:
                continue

        out.append(p)
    return out


def evidence_label(fm):
    """Return the page's evidence-status label for display.

    `evidence_status` records whether a claim is only compiled, actually
    executed, profiled, or still pending on the target device. It is a
    different axis from source `confidence`; a page that does not track it
    shows '-' rather than borrowing its confidence value.
    """
    status = fm.get("evidence_status")
    return str(status) if status else "-"


def arch_label(fm):
    archs = fm.get("architectures")
    if isinstance(archs, list) and archs:
        return ",".join(str(a) for a in archs)
    return "-"


def format_result(page, compact=False):
    """Format a single page as a readable line."""
    fm = page["fm"]
    title = fm.get("title", "Untitled")
    path = page["path"]
    pid = fm.get("id", "")
    ptype = page.get("_ptype", "?")
    transfer = page.get("_transfer_limited", False)
    flag = "  [transfer-limited]" if transfer else ""

    if compact:
        tgts = fm.get("targets")
        target_seg = f" target={','.join(str(t) for t in tgts)}" if tgts else ""
        return (f"  [{ptype}] {pid}: {title}  "
                f"arch={arch_label(fm)}{target_seg} evidence={evidence_label(fm)}{flag}"
                f"  ({path})")

    # Rich: include key metadata
    lines = [f"## {title}"]
    lines.append(f"- **id**: `{pid}`")
    lines.append(f"- **type**: `{ptype}`")
    lines.append(f"- **path**: `{path}`")
    if "architectures" in fm:
        lines.append(f"- **architectures**: {fm['architectures']}")
    if "targets" in fm:
        lines.append(f"- **targets**: {fm['targets']}")
    lines.append(f"- **evidence_status**: {evidence_label(fm)}")
    if transfer:
        lines.append("- **transfer**: cross-architecture reference — provenance "
                     "preserved; validate on the requested target before reuse")
    for k in ("confidence", "reproducibility"):
        if k in fm:
            lines.append(f"- **{k}**: {fm[k]}")
    for k in ("tags", "techniques", "hardware_features", "kernel_types", "languages"):
        v = fm.get(k)
        if v:
            lines.append(f"- **{k}**: {v}")
    if "performance_claims" in fm and isinstance(fm["performance_claims"], list):
        for claim in fm["performance_claims"][:2]:
            lines.append(f"- **perf**: {claim.get('value')} {claim.get('metric')} on {claim.get('gpu')} ({claim.get('dtype')}, {claim.get('shape')})")
    if "sources" in fm:
        lines.append(f"- **sources**: {fm['sources'][:5]}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Query the Blackwell kernel wiki")
    parser.add_argument("query", nargs="*", help="Free-text keywords")
    parser.add_argument("--type", help="Filter by page type (kernel, technique, hardware, pattern, language, migration, pr, blog, doc, contest)")
    parser.add_argument("--tag", help="Filter by tag (must appear in tags/techniques/hardware_features/kernel_types/languages)")
    parser.add_argument("--repo", help="Filter by source repo (partial match, e.g. 'cutlass')")
    parser.add_argument("--language", help="Filter by language/DSL (cute-dsl, cuda-cpp, ptx, triton, etc.)")
    parser.add_argument("--architecture", help="Filter by device architecture (sm100, sm110, sm110a, sm90, ...) or by compiler-target family (sm110f, matched against a page's targets metadata)")
    parser.add_argument("--symptom", help="Filter by pattern symptom (memory-bound, register-pressure, etc.)")
    parser.add_argument("--confidence", help="Filter by confidence (verified, source-reported, inferred, experimental)")
    parser.add_argument("--has-code", action="store_true", help="Only return pages whose artifact_dir contains at least one source file")
    parser.add_argument("--limit", type=int, default=10, help="Max results (default 10)")
    parser.add_argument("--compact", action="store_true", help="Compact one-line-per-result output")
    parser.add_argument("--paths-only", action="store_true", help="Output only file paths, one per line")
    args = parser.parse_args()

    pages = load_all_pages()
    pages = filter_pages(pages, args)

    # Score by keywords if any. Flatten multi-word quoted args into tokens so
    # `query.py "how to fuse dual GEMM"` behaves like `query.py how to fuse dual GEMM`.
    keywords = []
    for q in args.query:
        for tok in re.split(r"\s+", q.strip()):
            if tok:
                keywords.append(tok)
    if keywords:
        for p in pages:
            p["_score"] = score_keyword_match(p["fm"], p["body"], keywords)
        pages = [p for p in pages if p["_score"] > 0]
        apply_architecture_rerank(pages, keywords)
        pages.sort(key=lambda x: (-x["_arch_tier"], -x["_score"], x["path"]))
    else:
        pages.sort(key=lambda x: x["path"])

    pages = pages[:args.limit]

    if args.paths_only:
        for p in pages:
            print(p["path"])
        return

    if not pages:
        print("No matching pages.")
        return

    print(f"# {len(pages)} result(s)")
    print()
    for p in pages:
        print(format_result(p, compact=args.compact))
        if not args.compact:
            print()


if __name__ == "__main__":
    main()
