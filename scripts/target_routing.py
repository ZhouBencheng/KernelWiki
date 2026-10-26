"""Curated guidance applicability; deliberately separate from source architecture evidence."""
from functools import lru_cache
import re

from _wiki_root import WIKI_ROOT
from _yaml_compat import yaml


@lru_cache(maxsize=1)
def targets():
    return yaml.safe_load((WIKI_ROOT / "data" / "targets.yaml").read_text(encoding="utf-8"))


def resolve_target(value):
    """Return a known base device target, never infer one from a family or a/f suffix."""
    token = re.sub(r"[ _-]+", "", str(value).strip().lower())
    for target, record in targets().items():
        if token == target or token in {re.sub(r"[ _-]+", "", str(alias).lower()) for alias in record["aliases"]}:
            return target
    return None


def applicable_to(frontmatter, requested):
    """Only explicitly reviewed wiki guidance can enter a target-selected result."""
    scope = frontmatter.get("applicable_to", [])
    if not frontmatter.get("type") or not isinstance(scope, list):
        return False
    target = resolve_target(requested)
    return "portable" in scope or (target is not None and target in scope)


def applicability_errors(frontmatter):
    scope = frontmatter.get("applicable_to")
    if scope is None:
        return []
    if not frontmatter.get("type"):
        return ["applicable_to belongs to curated wiki guidance, not raw source evidence"]
    if not isinstance(scope, list) or not scope or any(not isinstance(x, str) for x in scope):
        return ["applicable_to must be a non-empty list of base targets or [portable]"]
    errors = []
    if len(set(scope)) != len(scope):
        errors.append("applicable_to contains duplicates")
    if "portable" in scope and scope != ["portable"]:
        errors.append("portable must be the only applicable_to value")
    if scope == ["portable"] and frontmatter.get("architectures") != []:
        errors.append("portable guidance must declare architectures: []")
    unknown = set(scope) - set(targets()) - {"portable"}
    if unknown:
        errors.append(f"unknown guidance target(s): {sorted(unknown)}")
    return errors


def target_registry_errors(raw=None):
    """Validate explicit device aliases without inferring architecture families."""
    raw = targets() if raw is None else raw
    if not isinstance(raw, dict) or not raw:
        return ["data/targets.yaml: top level must be a non-empty mapping"]

    errors = []
    aliases = {}
    for target, record in raw.items():
        if not isinstance(target, str) or not re.fullmatch(r"sm\d+", target):
            errors.append(f"data/targets.yaml: invalid base target {target!r}")
            continue
        if not isinstance(record, dict):
            errors.append(f"data/targets.yaml::{target}: profile must be a mapping")
            continue
        values = record.get("aliases")
        page = record.get("page")
        if not isinstance(values, list) or any(
            not isinstance(value, str) or not value.strip() for value in values
        ):
            errors.append(f"data/targets.yaml::{target}: aliases must be a list of non-empty strings")
            values = []
        if not isinstance(page, str) or not page.strip():
            errors.append(f"data/targets.yaml::{target}: page must be a non-empty ID")
        for value in [target, *values]:
            token = re.sub(r"[ _-]+", "", value.strip().lower())
            previous = aliases.get(token)
            if previous is not None and previous != target:
                errors.append(
                    f"data/targets.yaml: alias {value!r} maps to both {previous!r} and {target!r}"
                )
            else:
                aliases[token] = target
    return errors
