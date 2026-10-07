# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2026 Delos Data, Inc.

"""Unit tests for env_vars module."""

import os
from unittest.mock import patch

import pytest

from production_test_framework import env_vars
from production_test_framework.config import LGTMConfig
from production_test_framework.env_vars import EnvironmentVariable, Scoped


class TestIsSet:
    """Tests for is_set."""

    def test_unset(self):
        with patch.dict(os.environ, {}, clear=True):
            assert env_vars.is_set("FOO") is False

    def test_whitespace_counts_as_unset(self):
        with patch.dict(os.environ, {"FOO": "  "}, clear=True):
            assert env_vars.is_set("FOO") is False

    def test_set(self):
        with patch.dict(os.environ, {"FOO": "x"}, clear=True):
            assert env_vars.is_set("FOO") is True


class TestDeduplicate:
    """Tests for deduplicate."""

    def test_last_declaration_wins(self):
        first = EnvironmentVariable(name="FOO", purpose="first")
        last = EnvironmentVariable(name="FOO", purpose="last", required=True)
        result = env_vars.deduplicate([first, EnvironmentVariable(name="BAR", purpose="b"), last])
        assert {v.name: v.purpose for v in result} == {"FOO": "last", "BAR": "b"}


class TestMissingRequired:
    """Tests for missing_required."""

    def test_only_unset_required_variables_sorted(self):
        variables = [
            EnvironmentVariable(name="ZED", purpose="z", required=True),
            EnvironmentVariable(name="ABC", purpose="a", required=True),
            EnvironmentVariable(name="SET", purpose="s", required=True),
            EnvironmentVariable(name="OPT", purpose="o"),
        ]
        with patch.dict(os.environ, {"SET": "x"}, clear=True):
            assert [v.name for v in env_vars.missing_required(variables)] == ["ABC", "ZED"]


class TestRenderReport:
    """Tests for render_report."""

    def test_empty(self):
        assert env_vars.render_report([], title="t") == "t: no environment variables declared"

    def test_groups_status_and_fallback(self):
        variables = [
            EnvironmentVariable(name="NEED", purpose="needed", required=True),
            EnvironmentVariable(name="OPT", purpose="optional", fallback="a default"),
        ]
        with patch.dict(os.environ, {}, clear=True):
            report = env_vars.render_report(variables, title="suite")
        assert report.splitlines()[0] == "suite: 2 environment variable(s)"
        assert report.index("required:") < report.index("NEED") < report.index("optional:") < report.index("OPT")
        assert "falls back to: a default" in report

    def test_fallback_hidden_when_set(self):
        variables = [EnvironmentVariable(name="OPT", purpose="optional", fallback="a default")]
        with patch.dict(os.environ, {"OPT": "x"}, clear=True):
            assert "falls back" not in env_vars.render_report(variables, title="t")

    def test_never_renders_values(self):
        variables = [EnvironmentVariable(name="SECRET", purpose="password", required=True)]
        with patch.dict(os.environ, {"SECRET": "hunter2"}, clear=True):
            assert "hunter2" not in env_vars.render_report(variables, title="t")


class TestEnvTagVariables:
    """Tests for env_tag_variables."""

    def test_tagged_bare_default_and_untagged(self, tmp_path):
        path = tmp_path / "cluster.yaml"
        path.write_text(
            "password: !ENV ${OS_PASSWORD}\n"
            "user: !ENV ${OS_USER:admin}\n"
            "literal: ${NOT_TAGGED}\n"
            "both: !ENV ${A}-${B:x}\n"
        )
        found = {v.name: v for v in env_vars.env_tag_variables(path)}
        assert set(found) == {"OS_PASSWORD", "OS_USER", "A", "B"}
        assert found["OS_PASSWORD"].required is True
        assert found["OS_USER"].required is False
        assert found["OS_USER"].fallback == "the default in cluster.yaml: 'admin'"


class TestLGTMConfigEnvironmentVariables:
    """Tests for LGTMConfig.environment_variables."""

    def test_ssh_user_optional_for_localhost(self):
        with patch.dict(os.environ, {}, clear=True):
            variables = {v.name: v for v in LGTMConfig.environment_variables()}
        assert variables["ANSIBLE_REMOTE_USER"].required is False
        assert variables["ANSIBLE_REMOTE_USER"].fallback

    def test_ssh_user_required_for_remote_host(self):
        with patch.dict(os.environ, {"REMOTE_HOST": "remote.example.com"}, clear=True):
            variables = {v.name: v for v in LGTMConfig.environment_variables()}
        assert variables["ANSIBLE_REMOTE_USER"].required is True
        assert variables["ANSIBLE_REMOTE_USER"].fallback is None

    def test_declarations_come_from_the_package_file(self):
        with patch.dict(os.environ, {}, clear=True):
            variables = {v.name: v for v in LGTMConfig.environment_variables()}
        assert variables["REMOTE_HOST"] == EnvironmentVariable(
            name="REMOTE_HOST", purpose="target host for SSH and kubectl", fallback="localhost"
        )


class TestUsedBy:
    """Tests for used_by."""

    def test_single_name_is_not_split_into_characters(self):
        variable = EnvironmentVariable(name="FOO", purpose="p")
        assert env_vars.used_by("inner", variable) == [Scoped(variable, frozenset({"inner"}))]

    def test_list_of_names(self):
        variable = EnvironmentVariable(name="FOO", purpose="p")
        assert env_vars.used_by(["a", "b"], variable)[0].used_by == frozenset({"a", "b"})


class TestLoadDeclarations:
    """Tests for load_declarations."""

    def _load(self, tmp_path, text):
        (tmp_path / "suite").mkdir()
        path = tmp_path / "env_vars.yaml"
        path.write_text(text)
        return env_vars.load_declarations(path)

    def test_parses_plain_and_scoped_declarations(self, tmp_path):
        declarations = self._load(
            tmp_path,
            '".":\n'
            "  PLAIN:\n"
            "    purpose: p\n"
            "    fallback: 5556\n"
            "suite:\n"
            "  SCOPED:\n"
            "    purpose: s\n"
            "    required: true\n"
            "    used_by: inner\n",
        )
        root = tmp_path.resolve()
        assert declarations[root] == [EnvironmentVariable(name="PLAIN", purpose="p", fallback="5556")]
        assert declarations[root / "suite"] == [
            Scoped(EnvironmentVariable(name="SCOPED", purpose="s", required=True), frozenset({"inner"}))
        ]

    def test_merge_key_override_is_not_a_duplicate(self, tmp_path):
        declarations = self._load(
            tmp_path,
            '".":\n'
            "  FIRST: &base\n"
            "    purpose: first\n"
            "    fallback: f\n"
            "  SECOND:\n"
            "    <<: *base\n"
            "    purpose: second\n",
        )
        assert declarations[tmp_path.resolve()][1] == EnvironmentVariable(name="SECOND", purpose="second", fallback="f")

    @pytest.mark.parametrize(
        "text, message",
        [
            ('".":\n  FOO:\n    required: true\n', "needs a purpose"),
            ('".":\n  FOO:\n    purpose: p\n    requird: true\n', "unknown field"),
            ('".":\n  FOO:\n    purpose: p\n    required: true\n    fallback: f\n', "cannot have a fallback"),
            ("missing:\n  FOO:\n    purpose: p\n", "is not a directory"),
            ('".":\n  - FOO\n', "must map variable names"),
            ('".":\n  FOO:\n    purpose: p\n  FOO:\n    purpose: q\n', "duplicate key 'FOO'"),
            ('".":\n  FOO:\n    purpose: p\n".":\n  BAR:\n    purpose: p\n', "duplicate key '.'"),
            ('".":\n  FOO: [unclosed\n', "env_vars.yaml"),
        ],
        ids=[
            "no-purpose",
            "unknown-field",
            "required-with-fallback",
            "unknown-directory",
            "directory-not-a-mapping",
            "duplicate-variable",
            "duplicate-directory",
            "invalid-yaml",
        ],
    )
    def test_malformed_file_raises(self, tmp_path, text, message):
        with pytest.raises(ValueError, match=message):
            self._load(tmp_path, text)
