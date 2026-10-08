# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2025 Delos Data, Inc.

"""
Configuration management for LGTM stack tests.

Loads configuration from environment variables with defaults.
"""

import os
from dataclasses import dataclass, field, replace
from functools import cache
from pathlib import Path

from production_test_framework.env_vars import EnvironmentVariable, load_declarations

_ENV_VARS_FILE = Path(__file__).with_name("env_vars.yaml")


@cache
def _declared() -> dict[str, EnvironmentVariable]:
    """The package's env_vars.yaml declarations, by name."""
    return {v.name: v for v in load_declarations(_ENV_VARS_FILE)[_ENV_VARS_FILE.parent.resolve()]}


@dataclass
class LGTMConfig:
    """Configuration for LGTM stack deployment and testing."""

    # Target host configuration
    host: str = field(default_factory=lambda: os.getenv("REMOTE_HOST", "localhost"))

    # SSH/Ansible configuration
    ansible_remote_user: str = field(default_factory=lambda: os.getenv("ANSIBLE_REMOTE_USER", ""))

    grafana_port: int = 3000
    mimir_port: int = 9009
    loki_port: int = 3100
    otlp_grpc_port: int = 4317
    otlp_http_port: int = 4318

    def validate_ssh_config(self) -> bool:
        """Check if SSH configuration is complete. Always true for a localhost target."""
        # Imported here because helper -> ssh -> config would be a circular import.
        from production_test_framework.helper import is_localhost

        if is_localhost(self.host):
            return True
        return bool(self.ansible_remote_user)

    @classmethod
    def from_env(cls) -> LGTMConfig:
        """Create configuration from environment variables."""
        return cls()

    @classmethod
    def environment_variables(cls) -> list[EnvironmentVariable]:
        """The variables from_env reads; the SSH user is required only for a remote host."""
        from production_test_framework.helper import is_localhost

        ssh_user = _declared()["ANSIBLE_REMOTE_USER"]
        if not is_localhost(cls.from_env().host):
            ssh_user = replace(ssh_user, required=True, fallback=None)
        return [_declared()["REMOTE_HOST"], ssh_user]
