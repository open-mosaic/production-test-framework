# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2026 Delos Data, Inc.

"""
Run commands on this machine or, through the system ssh client, on another one, the same way.

A remote host needs passwordless SSH from this machine (``ssh-copy-id <host>``): every command
runs with ``BatchMode=yes``, so ssh fails instead of prompting. The user is the one in the
address (``user@host``), else whatever ``~/.ssh/config`` or ssh itself picks.
"""

import os
import platform
import shlex
import shutil
from collections.abc import Sequence
from functools import cached_property
from pathlib import Path

from .helper import is_localhost, run_command
from .ssh import CommandResult

__all__ = ["SSH_OPTIONS", "Host"]

SSH_OPTIONS = (
    "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=10",
    "-o", "ControlMaster=auto",
    "-o", "ControlPath=~/.ssh/ptf-%C",
    "-o", "ControlPersist=60",
)  # fmt: skip


class Host:
    """A machine to run commands on: this one for a localhost address, else over ssh."""

    def __init__(self, address: str = "localhost"):
        self.address = address.strip() or "localhost"

    def __str__(self) -> str:
        return self.address

    def __repr__(self) -> str:
        return f"Host({self.address!r})"

    @property
    def hostname(self) -> str:
        """The address without its user: what an HTTP client connects to."""
        return self.address.rpartition("@")[2]

    @property
    def local(self) -> bool:
        return is_localhost(self.hostname)

    def argv(self, command: Sequence[str]) -> list[str]:
        """*command* as run on this host: itself here, else wrapped in ssh."""
        if self.local:
            return list(command)
        return ["ssh", *SSH_OPTIONS, self.address, shlex.join(command)]

    def run(self, command: Sequence[str], timeout: int = 60, stdin_data: str | None = None) -> CommandResult:
        return run_command(self.argv(command), timeout=timeout, stdin_data=stdin_data)

    def shell(self, script: str, *args: str, timeout: int = 60, stdin_data: str | None = None) -> CommandResult:
        """Run *script* with ``sh -c``, given *args* as ``$1`` onwards."""
        return self.run(["sh", "-c", script, "sh", *args], timeout=timeout, stdin_data=stdin_data)

    def access_problem(self) -> str | None:
        """Why commands cannot run on this host without a password, or None when they can."""
        if self.local:
            return None
        result = self.run(["true"], timeout=30)
        if result.success:
            return None
        reason = (result.stderr or f"exit status {result.returncode}").splitlines()[-1].rstrip(".")
        return (
            f"{self.address} needs passwordless SSH from this machine: "
            f"`ssh -o BatchMode=yes {self.address} true` failed ({reason}). "
            f"Run `ssh-copy-id {self.address}`, then run again."
        )

    def which(self, tool: str) -> bool:
        if self.local:
            return shutil.which(tool) is not None
        return self.shell('command -v "$1"', tool).success

    def getenv(self, name: str) -> str | None:
        """*name* in the environment commands get on this host, or None when unset or empty."""
        if self.local:
            return os.getenv(name) or None
        result = self.run(["printenv", name])
        return (result.stdout or None) if result.success else None

    @cached_property
    def home(self) -> str:
        return str(Path.home()) if self.local else self.getenv("HOME") or "~"

    @cached_property
    def machine(self) -> str:
        """The CPU architecture, as ``uname -m`` names it."""
        if self.local:
            return platform.machine()
        return self.run(["uname", "-m"]).stdout

    def is_dir(self, path: str) -> bool:
        if self.local:
            return Path(path).is_dir()
        return self.run(["test", "-d", path]).success

    def free_bytes(self, path: str) -> int | None:
        """Free space on the filesystem that holds *path*, or its nearest existing parent."""
        if self.local:
            existing = next(p for p in (Path(path), *Path(path).parents) if p.exists())
            return shutil.disk_usage(existing).free
        result = self.shell(
            'p="$1"; while [ ! -e "$p" ]; do p=$(dirname "$p"); done; df -Pk "$p" | awk \'NR == 2 { print $4 }\'',
            path,
        )
        return int(result.stdout) * 1024 if result.success and result.stdout.isdigit() else None

    def write_file(self, path: str, text: str) -> CommandResult:
        """Write *text* to *path*, creating its directory."""
        return self.shell('mkdir -p "$(dirname "$1")" && cat > "$1"', path, stdin_data=text)
