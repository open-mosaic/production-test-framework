# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2026 Delos Data, Inc.

"""Declare the environment variables a component reads, and report which are set.

Declarations are made in Python or loaded from a YAML file keyed by directory (see
load_declarations). EnvironmentCheck selects the declarations that apply to a set of
targets, such as the tests a run selected, reports them and enforces the required ones.
Values are never rendered, so reports are safe to print for variables holding secrets.
"""

import os
import re
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

# pyaml-env substitution syntax: ${VAR} and ${VAR:default}.
_SUBSTITUTION = re.compile(r"\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?::(?P<default>[^}]*))?\}")
_FIELDS = frozenset({"purpose", "required", "fallback", "used_by"})


@dataclass(frozen=True)
class EnvironmentVariable:
    """One environment variable a component reads, and what happens without it."""

    name: str
    purpose: str
    required: bool = False
    fallback: str | None = None


@dataclass(frozen=True)
class Scoped:
    """A variable that applies only where one of the named users (e.g. pytest fixtures) is used."""

    variable: EnvironmentVariable
    used_by: frozenset[str]


def used_by(names: str | Iterable[str], *variables: EnvironmentVariable) -> list[Scoped]:
    """Scope variables to where any of names is used."""
    names = frozenset([names] if isinstance(names, str) else names)
    return [Scoped(variable, names) for variable in variables]


def is_set(name: str) -> bool:
    """True when the variable holds something other than whitespace."""
    return bool(os.environ.get(name, "").strip())


def deduplicate(variables: Iterable[EnvironmentVariable]) -> list[EnvironmentVariable]:
    """Collapse repeated names, keeping the last declaration of each."""
    return list({variable.name: variable for variable in variables}.values())


def missing_required(variables: Iterable[EnvironmentVariable]) -> list[EnvironmentVariable]:
    """Required variables that are unset, sorted by name."""
    missing = [v for v in deduplicate(variables) if v.required and not is_set(v.name)]
    return sorted(missing, key=lambda v: v.name)


def render_report(variables: Iterable[EnvironmentVariable], *, title: str) -> str:
    """Render a set/UNSET table grouped into required and optional."""
    variables = deduplicate(variables)
    if not variables:
        return f"{title}: no environment variables declared"

    width = max(len(v.name) for v in variables)
    lines = [f"{title}: {len(variables)} environment variable(s)"]
    for group, required in (("required", True), ("optional", False)):
        members = sorted((v for v in variables if v.required is required), key=lambda v: v.name)
        if not members:
            continue
        lines.append(f"  {group}:")
        for variable in members:
            status = "set" if is_set(variable.name) else "UNSET"
            lines.append(f"    {variable.name:<{width}}  {status:<5}  {variable.purpose}")
            if status == "UNSET" and variable.fallback:
                lines.append(f"    {'':<{width}}         falls back to: {variable.fallback}")
    return "\n".join(lines)


def env_tag_variables(path: Path) -> list[EnvironmentVariable]:
    """Variables a YAML file substitutes into itself with pyaml-env's !ENV tag.

    A bare ${VAR} is required; ${VAR:default} is optional with the default as its fallback.
    Only lines carrying !ENV are scanned, since untagged ${...} is literal text.
    """
    found: list[EnvironmentVariable] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        tag = line.find("!ENV")
        if tag == -1:
            continue
        for match in _SUBSTITUTION.finditer(line[tag:]):
            default = match.group("default")
            found.append(
                EnvironmentVariable(
                    name=match.group("name"),
                    purpose=f"substituted into {path.name}",
                    required=default is None,
                    fallback=None if default is None else f"the default in {path.name}: {default!r}",
                )
            )
    return deduplicate(found)


class _UniqueKeyLoader(yaml.SafeLoader):
    """SafeLoader that rejects a key repeated within one mapping, instead of keeping the last."""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict:
        seen = set()
        for key_node, _ in node.value:
            # A << merge key is expected to be overridden by the keys beside it.
            if key_node.tag == "tag:yaml.org,2002:merge":
                continue
            key = self.construct_object(key_node, deep=deep)
            if key in seen:
                raise yaml.constructor.ConstructorError(None, None, f"duplicate key {key!r}", key_node.start_mark)
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


def _declaration(source: str, name: str, spec: object) -> EnvironmentVariable | Scoped:
    if not isinstance(spec, dict) or "purpose" not in spec:
        raise ValueError(f"{source}: {name} needs a purpose")
    if unknown := set(spec) - _FIELDS:
        raise ValueError(f"{source}: {name} has unknown field(s) {sorted(unknown)}; use {sorted(_FIELDS)}")
    required = bool(spec.get("required", False))
    fallback = spec.get("fallback")
    if required and fallback is not None:
        raise ValueError(f"{source}: {name} is required, so it cannot have a fallback")
    variable = EnvironmentVariable(
        name=name,
        purpose=spec["purpose"],
        required=required,
        fallback=None if fallback is None else str(fallback),
    )
    names = spec.get("used_by")
    return used_by(names, variable)[0] if names else variable


def load_declarations(path: Path, *, root: Path | None = None) -> dict[Path, list[EnvironmentVariable | Scoped]]:
    """Read a declarations file into {directory: declarations}; raises ValueError if malformed.

    Top-level keys are directories relative to root (default: the file's directory). Each
    maps variable names to purpose (required), required, fallback, and used_by (a name or
    list of names).
    """
    base = path.parent if root is None else root
    try:
        data = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"{path.name}: {exc}") from exc
    declarations = {}
    for directory, variables in data.items():
        where = (base / directory).resolve()
        if not where.is_dir():
            raise ValueError(f"{path.name}: {directory!r} is not a directory under {base}")
        if not isinstance(variables, dict):
            raise ValueError(f"{path.name}: {directory!r} must map variable names to declarations")
        declarations[where] = [_declaration(path.name, name, spec) for name, spec in variables.items()]
    return declarations


class MissingEnvironmentVariables(Exception):
    """Required environment variables are unset; the message names each one and its purpose."""

    def __init__(self, missing: list[EnvironmentVariable]):
        self.missing = missing
        detail = "\n".join(f"  {v.name}: {v.purpose}" for v in missing)
        super().__init__(f"{len(missing)} required environment variable(s) are unset:\n{detail}")


class EnvironmentCheck:
    """Select the declarations that apply to a set of targets (e.g. tests), report and enforce them.

    A declaration keyed by a directory applies to targets in it or below it. A Scoped one
    applies only to targets that use one of its names. When a variable is declared both
    optional and required, required wins.
    """

    def __init__(self, declarations: Mapping[Path, Iterable[EnvironmentVariable | Scoped]] | None = None):
        self._from_file = {Path(d).resolve(): list(e) for d, e in (declarations or {}).items()}
        self._declared: dict[Path, list[EnvironmentVariable | Scoped]] = {}
        self._scoped: dict[Path, list[Scoped]] = {}
        self._selected: dict[str, EnvironmentVariable] = {}

    @classmethod
    def from_file(cls, path: Path, *, root: Path | None = None) -> EnvironmentCheck:
        """Check against a declarations file; raises ValueError if it is malformed."""
        return cls(load_declarations(path, root=root))

    def declare(self, directory: Path, entries: Iterable[EnvironmentVariable | Scoped]) -> None:
        """Add computed declarations for targets in exactly this directory; call before its first use()."""
        self._declared.setdefault(Path(directory).resolve(), []).extend(entries)

    def use(self, directory: Path, names: str | Collection[str] = ()) -> None:
        """Record one target in directory that uses names (e.g. a test's fixtures)."""
        if isinstance(names, str):
            names = (names,)
        directory = Path(directory)
        if directory not in self._scoped:
            self._scoped[directory] = []
            for entry in self._entries(directory.resolve()):
                if isinstance(entry, Scoped):
                    self._scoped[directory].append(entry)
                else:
                    self._select(entry)
        for entry in self._scoped[directory]:
            if not entry.used_by.isdisjoint(names):
                self._select(entry.variable)

    @property
    def variables(self) -> list[EnvironmentVariable]:
        """The variables that apply to the targets used so far."""
        return list(self._selected.values())

    def missing(self) -> list[EnvironmentVariable]:
        """Required variables that apply and are unset, sorted by name."""
        return missing_required(self._selected.values())

    def report(self, title: str, *, full: bool) -> str:
        """The set/UNSET table when full, otherwise a one-line summary."""
        if full:
            return render_report(self._selected.values(), title=title)
        count = sum(is_set(name) for name in self._selected)
        return f"{title}: {len(self._selected)} environment variable(s), {count} set"

    def enforce(self) -> None:
        """Raise MissingEnvironmentVariables if any required variable that applies is unset."""
        if missing := self.missing():
            raise MissingEnvironmentVariables(missing)

    def _entries(self, directory: Path) -> list[EnvironmentVariable | Scoped]:
        inherited = [entry for where in (directory, *directory.parents) for entry in self._from_file.get(where, ())]
        return inherited + self._declared.get(directory, [])

    def _select(self, variable: EnvironmentVariable) -> None:
        current = self._selected.get(variable.name)
        if current is None or (variable.required and not current.required):
            self._selected[variable.name] = variable
