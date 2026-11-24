# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2026 Delos Data, Inc.

"""Docker Compose helper for compose-based test suites.

ComposeStack brings a docker-compose project's containers up and down and reports their
state through one object, so tests manage the stack instead of shelling out to
`docker compose` by hand.
"""

import json
import re
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class ContainerStatus:
    service: str
    name: str
    state: str
    health: str  # "" when the service declares no healthcheck
    exit_code: int | None = None  # set by `docker compose ps` once the container has exited

    @property
    def is_completed(self) -> bool:
        """An exited one-shot container (an init job, say) that finished successfully."""
        return self.state == "exited" and self.exit_code == 0

    @property
    def is_running(self) -> bool:
        return self.state == "running"

    @property
    def is_healthy(self) -> bool:
        if self.health:
            return self.health == "healthy"
        return self.is_running


def _run(args: list[str], timeout: int = 300) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout)


class ComposeStack:
    """A docker compose project - its files and profiles - that can be brought up/down."""

    def __init__(self, files: list[str], *, profiles: tuple[str, ...] = ()) -> None:
        self._files = files
        self._profiles = profiles

    def _args(self, *rest: str) -> list[str]:
        args = ["docker", "compose"]
        for f in self._files:
            args += ["-f", f]
        for profile in self._profiles:
            args += ["--profile", profile]
        return args + list(rest)

    def up(self, *, wait_timeout: int = 180, log_tail: int | None = 200) -> None:
        """Bring the stack up detached and wait for containers to be up/healthy.

        On failure or timeout the error ends with the last log_tail log lines of each container
        that is not up (None leaves the logs out), so the cause shows in the test output.
        """
        args = self._args("up", "-d", "--wait", "--wait-timeout", str(wait_timeout))
        try:
            result = _run(args, timeout=wait_timeout + 60)
        except subprocess.TimeoutExpired as exc:
            message = f"`docker compose up` timed out after {exc.timeout}s"
            raise RuntimeError(message + self._failed_container_logs(log_tail)) from exc
        if result.returncode != 0:
            message = f"`docker compose up` failed (rc={result.returncode}):\n{result.stderr}"
            raise RuntimeError(message + self._failed_container_logs(log_tail))

    def _failed_container_logs(self, tail: int | None) -> str:
        """Logs of the containers that are not up, for an `up` error. Never raises: "" on any problem."""
        if tail is None:
            return ""
        try:
            failed = [c.service for c in self.ps() if not (c.is_healthy or c.is_completed)]
            return "".join(
                f"\n\n--- logs of {service} (last {tail} lines) ---\n{self.logs(service, tail=tail)}"
                for service in failed
            )
        except RuntimeError, OSError, subprocess.SubprocessError, ValueError:
            return ""  # keep the original compose error rather than hide it behind a log failure

    def logs(self, *services: str, tail: int | None = None) -> str:
        """Combined stdout/stderr logs of the given services (all by default), without colors."""
        rest = ["logs", "--no-color", *(["--tail", str(tail)] if tail is not None else []), *services]
        result = _run(self._args(*rest), timeout=60)
        if result.returncode != 0:
            raise RuntimeError(f"`docker compose logs` failed (rc={result.returncode}):\n{result.stderr}")
        return result.stdout + result.stderr

    def down(self, *, volumes: bool = True) -> None:
        """Tear the stack down, removing orphans (and volumes by default)."""
        rest = ["down", "--remove-orphans", *(["-v"] if volumes else [])]
        result = _run(self._args(*rest))
        if result.returncode != 0:
            # surface teardown failures: a left-up stack breaks the next run's bring-up
            raise RuntimeError(f"`docker compose down` failed (rc={result.returncode}):\n{result.stderr}")

    def services(self) -> list[str]:
        """Names of the stack's services."""
        result = _run(self._args("config", "--services"), timeout=60)
        if result.returncode != 0:
            raise RuntimeError(f"`docker compose config --services` failed (rc={result.returncode}):\n{result.stderr}")
        return result.stdout.split()

    def pull(self, *services: str, ignore_failures: bool = False) -> list[str]:
        """Pull the images the given services (all by default) reference in the compose files.

        A failed pull raises, unless ignore_failures is set; then the failures are returned instead.
        """
        args = ["pull", "--ignore-pull-failures", *services] if ignore_failures else ["pull", *services]
        result = _run(self._args(*args), timeout=900)
        if not ignore_failures:
            if result.returncode != 0:
                raise RuntimeError(f"`docker compose pull` failed (rc={result.returncode}):\n{result.stderr}")
            return []
        # --ignore-pull-failures exits 0 even when an image fails, so failures only show in stderr.
        errors = [line for line in result.stderr.splitlines() if "error" in line.lower() or "denied" in line.lower()]
        if result.returncode != 0 and not errors:
            errors = [result.stderr.strip() or f"`docker compose pull` failed (rc={result.returncode})"]
        return errors

    def conflicting_containers(self) -> list[str]:
        """Containers from other projects that use this stack's container names or ports."""
        result = _run(self._args("config", "--format", "json"), timeout=60)
        if result.returncode != 0:
            raise RuntimeError(f"`docker compose config` failed (rc={result.returncode}):\n{result.stderr}")
        spec = json.loads(result.stdout)
        services = spec.get("services", {}).values()
        names = {service.get("container_name") for service in services} - {None}
        ports = {
            str(port["published"]) for service in services for port in service.get("ports", []) if port.get("published")
        }

        listing = _run(
            ["docker", "ps", "-a", "--format", '{{.Names}}\t{{.Label "com.docker.compose.project"}}\t{{.Ports}}'],
            timeout=60,
        )
        if listing.returncode != 0:
            raise RuntimeError(f"`docker ps` failed (rc={listing.returncode}):\n{listing.stderr}")
        conflicts = []
        for line in listing.stdout.splitlines():
            name, project, published = (line.split("\t") + ["", ""])[:3]
            if project != spec.get("name") and (name in names or ports & set(re.findall(r":(\d+)->", published))):
                conflicts.append(name)
        return conflicts

    def remove_conflicting_containers(self) -> list[str]:
        """Remove the containers conflicting_containers finds. Returns their names."""
        conflicts = self.conflicting_containers()
        if conflicts:
            removed = _run(["docker", "rm", "-f", *conflicts], timeout=120)
            if removed.returncode != 0:
                raise RuntimeError(f"`docker rm -f` of {conflicts} failed (rc={removed.returncode}):\n{removed.stderr}")
        return conflicts

    def ps(self) -> list[ContainerStatus]:
        """Return the status of every container in the compose project."""
        result = _run(self._args("ps", "--all", "--format", "json"), timeout=60)
        if result.returncode != 0:
            raise RuntimeError(f"`docker compose ps` failed (rc={result.returncode}):\n{result.stderr}")
        text = result.stdout.strip()
        if not text:
            return []
        # Newer compose emits one JSON object per line; older emits a single array.
        records = (
            json.loads(text)
            if text.startswith("[")
            else [json.loads(line) for line in text.splitlines() if line.strip()]
        )
        return [
            ContainerStatus(
                service=rec.get("Service", ""),
                name=rec.get("Name", ""),
                state=rec.get("State", ""),
                health=rec.get("Health", ""),
                exit_code=rec.get("ExitCode") if rec.get("State") == "exited" else None,
            )
            for rec in records
        ]

    def port(self, service: str, container_port: int) -> str:
        """Host "host:port" that the service's container port is published on.

        Read from the running project so the host port isn't hardcoded.
        """
        result = _run(self._args("port", service, str(container_port)), timeout=30)
        out = result.stdout.strip()
        if result.returncode != 0 or not out:
            raise RuntimeError(f"`docker compose port {service} {container_port}` returned nothing:\n{result.stderr}")
        host, _, port = out.rpartition(":")  # e.g. "0.0.0.0:50051"
        if host in ("", "0.0.0.0", "::", "[::]"):
            host = "localhost"
        return f"{host}:{port}"
