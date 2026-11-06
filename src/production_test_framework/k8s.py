# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2025 Delos Data, Inc.

"""
Kubernetes client utilities.

Provides high-level interfaces for interacting with Kubernetes clusters,
with support for both local kubectl and remote k3s kubectl via SSH.
"""

import json
import logging
import os
import shlex
import socket
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from production_test_framework.config import LGTMConfig, _declared
from production_test_framework.env_vars import EnvironmentVariable
from production_test_framework.helper import is_localhost, poll_until, run_command
from production_test_framework.ssh import CommandResult, SSHExecutor

logger = logging.getLogger(__name__)

WORKLOAD_READINESS_JSONPATH = {
    "deployment": "{.spec.replicas},{.status.readyReplicas},{.status.availableReplicas}",
    "statefulset": "{.spec.replicas},{.status.readyReplicas},{.status.availableReplicas}",
    "daemonset": "{.status.desiredNumberScheduled},{.status.numberReady},{.status.numberAvailable}",
}

POD_RESTART_TIMEOUT_S = 300.0
POD_RESTART_INTERVAL_S = 5.0
ROLLOUT_TIMEOUT_S = 300

STARTUP_RESTART_GRACE_S = 5 * 60
MAX_POD_RESTARTS = 10
# How recently a restart must have happened to mean anything is wrong *now*. Without this a
# single node reboot restarts every pod in a namespace at once and fails the restart checks
# for as long as those pods live, which on a long-lived cluster is indefinitely.
RECENT_RESTART_WINDOW_S = 24 * 60 * 60

# Node conditions that stop a pod being scheduled or get it evicted, when True.
NODE_PRESSURE_CONDITIONS = ("MemoryPressure", "DiskPressure", "PIDPressure", "NetworkUnavailable")

# How much of each pod's history pod_report() carries by default.
POD_REPORT_EVENTS = 15
POD_REPORT_LOG_LINES = 60


def _describe_age(seconds: float) -> str:
    """
    Render a duration as a short string in whichever unit keeps it legible.
    """
    minutes = seconds / 60
    if minutes < 90:
        return f"{minutes:.0f}m"
    hours = minutes / 60
    return f"{hours:.1f}h" if hours < 48 else f"{hours / 24:.1f}d"


@dataclass
class Node:
    """Kubernetes node information."""

    name: str
    status: str
    roles: str
    version: str
    internal_ip: str | None = None

    @property
    def is_ready(self) -> bool:
        return self.status == "Ready"


@dataclass
class Pod:
    """Kubernetes pod information."""

    name: str
    namespace: str
    status: str
    ready: str
    restarts: int
    age: str

    @property
    def is_running(self) -> bool:
        return self.status == "Running"

    @property
    def is_completed(self) -> bool:
        return self.status == "Completed"

    @property
    def is_ready(self) -> bool:
        """Check if all containers in the pod are ready."""
        if "/" not in self.ready:
            return False
        ready, total = self.ready.split("/")
        return ready == total and int(ready) > 0


class KubernetesClient:
    """
    High-level Kubernetes client.

    Executes kubectl on the local machine when host is localhost.
    Otherwise executes kubectl on the remote host via SSH.
    """

    def __init__(self, config: LGTMConfig, ssh: SSHExecutor | None = None):
        self.config = config
        self.ssh = ssh or SSHExecutor(config)

    @classmethod
    def environment_variables(cls) -> list[EnvironmentVariable]:
        """The variables local kubectl reads."""
        return [_declared()["KUBECONFIG"]]

    def _run_kubectl_local(
        self,
        args: str,
        timeout: int = 60,
        kubeconfig: str | None = None,
        stdin_data: str | None = None,
    ) -> CommandResult:
        """
        Run kubectl on the local machine.

        With no kubeconfig given, a set KUBECONFIG is left for kubectl to resolve itself, since
        it may be a colon-separated list that --kubeconfig cannot take. Otherwise
        ~/.kube/config is used.
        """
        if kubeconfig is None and not os.environ.get("KUBECONFIG"):
            kubeconfig = "~/.kube/config"
        kube_args = ["--kubeconfig", str(Path(kubeconfig).expanduser())] if kubeconfig else []
        try:
            cmd = ["kubectl", *kube_args] + shlex.split(args)
        except ValueError as e:
            return CommandResult(returncode=-1, stdout="", stderr=f"Invalid kubectl args: {e}")
        return run_command(cmd, timeout=timeout, stdin_data=stdin_data)

    def _run_kubectl(self, args: str, timeout: int = 60, stdin_data: str | None = None) -> CommandResult:
        """Execute kubectl on localhost or on the remote host via SSH."""
        if is_localhost(self.config.host):
            return self._run_kubectl_local(args, timeout=timeout, stdin_data=stdin_data)
        return self.ssh.run_kubectl(args, timeout=timeout, stdin_data=stdin_data)

    def _get_json(self, args: str, timeout: int = 60) -> tuple[dict | None, CommandResult]:
        """Run kubectl and parse its JSON output; None when kubectl fails or prints something else."""
        result = self._run_kubectl(args, timeout=timeout)
        if not result.success:
            return None, result
        try:
            return json.loads(result.stdout), result
        except json.JSONDecodeError as e:
            return None, CommandResult(
                returncode=-1, stdout=result.stdout, stderr=f"kubectl {args} did not return JSON: {e}"
            )

    # -------------------------------------------------------------------------
    # Node Operations
    # -------------------------------------------------------------------------

    def get_nodes(self) -> tuple[list[Node], CommandResult]:
        """
        Get all nodes in the cluster.

        Returns:
            Tuple of (list of Node objects, raw CommandResult)
        """
        result = self._run_kubectl("get nodes -o wide --no-headers")
        nodes = []

        if result.success:
            for line in result.stdout.split("\n"):
                if not line.strip():
                    continue
                parts = line.split()
                if len(parts) >= 5:
                    nodes.append(
                        Node(
                            name=parts[0],
                            status=parts[1],
                            roles=parts[2],
                            version=parts[4],
                            internal_ip=parts[5] if len(parts) > 5 else None,
                        )
                    )

        return nodes, result

    def get_node_count(self) -> int:
        """Get the number of nodes in the cluster."""
        nodes, _ = self.get_nodes()
        return len(nodes)

    def all_nodes_ready(self) -> bool:
        """Check if all nodes are in Ready state."""
        nodes, result = self.get_nodes()
        return result.success and all(n.is_ready for n in nodes)

    def node_conditions(self) -> tuple[list[str], CommandResult]:
        """
        Describe each node: its Ready condition, plus any pressure condition that is True.

        A pressure condition is what keeps a pod Pending or gets it evicted, so this
        explains a pod that will not come up without having to describe the node.
        """
        nodes, result = self._get_json("get nodes -o json")
        if nodes is None:
            return [], result

        lines = []
        for node in nodes.get("items", []):
            conditions = {c["type"]: c for c in node.get("status", {}).get("conditions", [])}
            ready = conditions.get("Ready", {})
            pressure = [
                f"{name} ({conditions[name].get('message', '')})"
                for name in NODE_PRESSURE_CONDITIONS
                if conditions.get(name, {}).get("status") == "True"
            ]
            lines.append(
                f"{node['metadata']['name']}: Ready={ready.get('status', '?')} ({ready.get('reason', '')})"
                + (f", {', '.join(pressure)}" if pressure else ", no pressure conditions")
            )
        return lines, result

    # -------------------------------------------------------------------------
    # Pod Operations
    # -------------------------------------------------------------------------

    def get_pods(self, namespace: str | None = None, all_namespaces: bool = False) -> tuple[list[Pod], CommandResult]:
        """
        Get pods in the cluster.

        Args:
            namespace: Specific namespace to query
            all_namespaces: Query all namespaces

        Returns:
            Tuple of (list of Pod objects, raw CommandResult)
        """
        if all_namespaces:
            cmd = "get pods -A --no-headers"
        elif namespace:
            cmd = f"get pods -n {namespace} --no-headers"
        else:
            cmd = "get pods --no-headers"

        result = self._run_kubectl(cmd)
        pods = []

        if result.success:
            for line in result.stdout.split("\n"):
                if not line.strip():
                    continue
                parts = line.split()

                # Format: NAMESPACE NAME READY STATUS RESTARTS AGE (when -A)
                # Format: NAME READY STATUS RESTARTS AGE (single namespace)
                if all_namespaces and len(parts) >= 5:
                    pods.append(
                        Pod(
                            namespace=parts[0],
                            name=parts[1],
                            ready=parts[2],
                            status=parts[3],
                            restarts=int(parts[4].split("(")[0]) if parts[4] else 0,
                            age=parts[5] if len(parts) > 5 else "",
                        )
                    )
                elif not all_namespaces and len(parts) >= 4:
                    pods.append(
                        Pod(
                            namespace=namespace or "default",
                            name=parts[0],
                            ready=parts[1],
                            status=parts[2],
                            restarts=int(parts[3].split("(")[0]) if parts[3] else 0,
                            age=parts[4] if len(parts) > 4 else "",
                        )
                    )

        return pods, result

    def get_pods_in_namespace(self, namespace: str) -> list[Pod]:
        """Get all pods in a specific namespace."""
        pods, _ = self.get_pods(namespace=namespace)
        return pods

    def all_pods_running(self, namespace: str) -> bool:
        """Check if all pods in a namespace are Running."""
        pods, result = self.get_pods(namespace=namespace)
        return result.success and all(p.is_running for p in pods)

    def all_pods_ready(self, namespace: str) -> bool:
        """Check if all pods in a namespace are Ready."""
        pods, result = self.get_pods(namespace=namespace)
        return result.success and all(p.is_ready or p.is_completed for p in pods)

    def wait_for_pods_ready(self, namespace: str, label: str, timeout: int = 180) -> bool:
        """Wait for pods matching a label to be ready."""
        cmd = f"wait --for=condition=Ready pods -l {label} -n {namespace} --timeout={timeout}s"
        result = self._run_kubectl(cmd, timeout=timeout + 10)
        return result.success

    def pods_by_selector(self, namespace: str, selector: str) -> dict[str, tuple[str, bool]]:
        """
        Get {pod name: (uid, all containers ready)} for a label selector.
        """
        result = self._run_kubectl(
            f"get pods -n {namespace} -l {selector} -o jsonpath="
            "'{range .items[*]}{.metadata.name}={.metadata.uid}={.status.containerStatuses[*].ready};{end}'"
        )
        if not result.success:
            return {}

        pods: dict[str, tuple[str, bool]] = {}
        for entry in result.stdout.split(";"):
            if not entry.strip():
                continue
            name, uid, readiness = entry.split("=", 2)
            flags = readiness.split()
            pods[name] = (uid, bool(flags) and all(flag == "true" for flag in flags))
        return pods

    def pod_containers(self, namespace: str, selector: str) -> tuple[dict[str, set[str]], CommandResult]:
        """
        Get the container names of every pod matching a label selector.
        """
        result = self._run_kubectl(
            f"get pods -n {namespace} -l {selector} "
            "-o jsonpath='{range .items[*]}{.metadata.name}={.spec.containers[*].name};{end}'"
        )
        if not result.success:
            return {}, result

        pods: dict[str, set[str]] = {}
        for entry in result.stdout.split(";"):
            if not entry.strip():
                continue
            name, containers = entry.split("=", 1)
            pods[name] = set(containers.split())
        return pods, result

    def pod_metrics(self, namespace: str, pod: str, port: int) -> tuple[str, CommandResult]:
        """
        Get one pod's Prometheus text, fetched through the API server proxy.
        """
        result = self._run_kubectl(f"get --raw /api/v1/namespaces/{namespace}/pods/{pod}:{port}/proxy/metrics")
        return result.stdout, result

    def unstable_pods(
        self,
        namespace: str,
        *,
        name_prefix: str = "",
        max_restarts: int = MAX_POD_RESTARTS,
        startup_grace_s: float = STARTUP_RESTART_GRACE_S,
        recent_window_s: float = RECENT_RESTART_WINDOW_S,
    ) -> tuple[list[str], CommandResult]:
        """
        Describe every container that looks unstable, empty when all are healthy.

        Two signals, either of which reports a container on its own:

        * A restart within recent_window_s that was also more than startup_grace_s into the
          pod's life. Recency is what separates "wrong now" from "restarted once, weeks ago
          and stable since"; the grace window excuses components that restart while waiting
          on their dependencies during a deploy.
        * More than max_restarts restarts over the container's lifetime, whenever they
          happened -- a crash loop slow enough that no single window catches it.
        """
        pods, result = self._get_json(f"get pods -n {namespace} -o json")
        if pods is None:
            return [], result

        unstable: list[str] = []
        now = datetime.now(UTC)

        for pod in pods.get("items", []):
            name = pod["metadata"]["name"]
            if not name.startswith(name_prefix):
                continue

            pod_started_at = pod.get("status", {}).get("startTime")
            pod_started = datetime.fromisoformat(pod_started_at) if pod_started_at else None

            for status in pod.get("status", {}).get("containerStatuses", []):
                restarts = status.get("restartCount", 0)
                if restarts == 0:
                    continue

                terminated = status.get("lastState", {}).get("terminated") or {}
                finished_at = terminated.get("finishedAt")
                finished = datetime.fromisoformat(finished_at) if finished_at else None

                into_life_s = (finished - pod_started).total_seconds() if finished and pod_started else None
                since_restart_s = (now - finished).total_seconds() if finished else None

                faults: list[str] = []
                if (
                    since_restart_s is not None
                    and since_restart_s < recent_window_s
                    and into_life_s is not None
                    and into_life_s > startup_grace_s
                ):
                    faults.append(
                        f"last restart {_describe_age(since_restart_s)} ago ({terminated.get('reason', 'unknown')})"
                    )
                if restarts > max_restarts:
                    faults.append(f"over the {max_restarts} restart ceiling")

                if faults:
                    unstable.append(f"{name}/{status['name']} restarted {restarts}x: {'; '.join(faults)}")

        return unstable, result

    def restart_pods(
        self,
        namespace: str,
        selector: str,
        *,
        graceful: bool = True,
        timeout: float = POD_RESTART_TIMEOUT_S,
        interval: float = POD_RESTART_INTERVAL_S,
    ) -> tuple[dict[str, tuple[str, bool]], CommandResult]:
        """
        Delete the pods matching a selector and wait for replacements to be ready.
        """
        before = self.pods_by_selector(namespace, selector)
        if not before:
            return {}, CommandResult(
                returncode=-1,
                stdout="",
                stderr=f"No pods matching {selector} found in {namespace} to restart",
            )

        flags = "--wait=false" if graceful else "--wait=false --grace-period=0 --force"
        result = self._run_kubectl(f"delete pods -n {namespace} -l {selector} {flags}")
        if not result.success:
            return {}, result

        old_uids = {uid for uid, _ in before.values()}

        def replaced() -> bool:
            current = self.pods_by_selector(namespace, selector)
            if len(current) < len(before):
                return False
            return all(uid not in old_uids and ready for uid, ready in current.values())

        if not poll_until(replaced, timeout=timeout, interval=interval):
            current = self.pods_by_selector(namespace, selector)
            return current, CommandResult(
                returncode=-1,
                stdout=result.stdout,
                stderr=(
                    f"Pods matching {selector} in {namespace} were not ready "
                    f"within {timeout:.0f}s of deletion: {current}"
                ),
            )

        return self.pods_by_selector(namespace, selector), result

    def pod_report(
        self,
        namespace: str,
        selector: str,
        *,
        events: int = POD_REPORT_EVENTS,
        log_lines: int = POD_REPORT_LOG_LINES,
    ) -> str:
        """
        Everything needed to tell why the pods behind a selector did not come up.

        Their status, recent events, each container's current and previous logs,
        and the node conditions, as one block of text for a failure message.
        """
        listed = self._run_kubectl(f"get pods -n {namespace} -l {selector} -o wide")
        if not listed.success:
            return f"could not list pods matching {selector} in {namespace}: {listed.stderr.strip()}"

        rows = listed.stdout.strip().splitlines()
        lines = [f"--- pods matching {selector} in {namespace}"] + [f"  {row}" for row in rows or ["none"]]

        pods, _ = self._get_json(f"get pods -n {namespace} -l {selector} -o json")
        for pod in (pods or {}).get("items", []):
            name = pod["metadata"]["name"]
            recent = self._run_kubectl(
                f"get events -n {namespace} --field-selector involvedObject.name={name} --sort-by=.lastTimestamp"
            )
            event_rows = recent.stdout.strip().splitlines() if recent.success else [recent.stderr.strip()]
            lines.append(f"--- events for {name} (last {events})")
            lines += [f"  {row}" for row in event_rows[-events:] or ["none"]]

            # One container at a time: --all-containers gives up on the whole pod while
            # it is still initializing, which loses the init container a Pending pod is
            # stuck in. Previous logs only exist for a container that has restarted.
            status = pod.get("status", {})
            for container in status.get("initContainerStatuses", []) + status.get("containerStatuses", []):
                for flag in [""] + (["--previous"] if container.get("restartCount", 0) else []):
                    logs = self._run_kubectl(
                        f"logs {name} -n {namespace} -c {container['name']} --tail={log_lines} {flag}"
                    )
                    log_rows = logs.stdout.strip().splitlines() if logs.success else [logs.stderr.strip()]
                    which = " (previous, crashed)" if flag else ""
                    lines.append(f"--- logs of {name}/{container['name']}{which} (last {log_lines} lines)")
                    lines += [f"  {row}" for row in log_rows or ["none"]]

        nodes, result = self.node_conditions()
        lines.append("--- node conditions")
        if result.success:
            lines += [f"  {row}" for row in nodes] or ["  none"]
        else:
            lines.append(f"  could not list nodes: {result.stderr.strip()}")
        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # Workload Operations
    # -------------------------------------------------------------------------

    def workload_readiness(self, namespace: str, kind: str, name: str) -> tuple[tuple[int, int, int], CommandResult]:
        """
        Get the (desired, ready, available) replica counts of a workload.

        Args:
            namespace: Namespace holding the workload.
            kind: One of deployment, statefulset or daemonset.
            name: Workload name.
        """
        jsonpath = WORKLOAD_READINESS_JSONPATH.get(kind)
        if jsonpath is None:
            return (0, 0, 0), CommandResult(returncode=-1, stdout="", stderr=f"Unsupported workload kind {kind!r}")

        result = self._run_kubectl(f"get {kind} {name} -n {namespace} -o jsonpath='{jsonpath}'")
        if not result.success:
            return (0, 0, 0), result

        fields = result.stdout.split(",")
        if len(fields) != 3:
            return (0, 0, 0), CommandResult(
                returncode=-1,
                stdout=result.stdout,
                stderr=f"Unexpected readiness output for {kind} {name}: {result.stdout!r}",
            )

        desired, ready, available = (int(field) if field else 0 for field in fields)
        return (desired, ready, available), result

    def wait_for_rollout(
        self, namespace: str, kind: str, name: str, *, timeout: int = ROLLOUT_TIMEOUT_S
    ) -> CommandResult:
        """Wait for the rollout of a deployment, statefulset or daemonset to finish."""
        return self._run_kubectl(
            f"rollout status {kind}/{name} -n {namespace} --timeout={timeout}s",
            timeout=timeout + 30,
        )

    def _deployment_volumes(self, namespace: str, deployment: str) -> tuple[list[dict], CommandResult]:
        obj, result = self._get_json(f"get deployment {deployment} -n {namespace} -o json")
        if obj is None:
            return [], result
        try:
            volumes = obj["spec"]["template"]["spec"].get("volumes", [])
        except (KeyError, TypeError, AttributeError) as e:
            return [], CommandResult(
                returncode=-1,
                stdout=result.stdout,
                stderr=f"Unexpected JSON for deployment {deployment}: {e}",
            )
        return volumes, result

    def deployment_volume(self, namespace: str, deployment: str, volume: str) -> tuple[dict | None, CommandResult]:
        """Get the spec of a named volume in a Deployment's pod template; None when there is no such volume."""
        volumes, result = self._deployment_volumes(namespace, deployment)
        return next((v for v in volumes if v.get("name") == volume), None), result

    def replace_deployment_volume(
        self, namespace: str, deployment: str, spec: dict, *, timeout: int = ROLLOUT_TIMEOUT_S
    ) -> CommandResult:
        """
        Replace the pod-template volume named spec["name"] and wait for the rollout to finish.

        The patch checks the volume's name at its index, so a concurrent change to
        the volume list fails the patch instead of replacing the wrong volume.
        """
        volumes, result = self._deployment_volumes(namespace, deployment)
        if not result.success:
            return result
        name = spec.get("name")
        index = next((i for i, v in enumerate(volumes) if v.get("name") == name), None)
        if index is None:
            return CommandResult(returncode=-1, stdout="", stderr=f"Deployment {deployment} has no volume {name!r}")

        path = f"/spec/template/spec/volumes/{index}"
        patch = [
            {"op": "test", "path": f"{path}/name", "value": name},
            {"op": "replace", "path": path, "value": spec},
        ]
        result = self._run_kubectl(
            f"patch deployment {deployment} -n {namespace} --type=json -p {shlex.quote(json.dumps(patch))}"
        )
        if not result.success:
            return result
        return self.wait_for_rollout(namespace, "deployment", deployment, timeout=timeout)

    # -------------------------------------------------------------------------
    # Namespace Operations
    # -------------------------------------------------------------------------

    def get_namespaces(self) -> tuple[list[str], CommandResult]:
        """Get all namespaces in the cluster."""
        result = self._run_kubectl("get namespaces --no-headers -o custom-columns=NAME:.metadata.name")
        namespaces = []

        if result.success:
            namespaces = [ns.strip() for ns in result.stdout.split("\n") if ns.strip()]

        return namespaces, result

    def namespace_exists(self, namespace: str) -> bool:
        """Check if a namespace exists."""
        result = self._run_kubectl(f"get namespace {namespace}")
        return result.success

    def all_namespaces_exist(self, namespaces: list[str]) -> tuple[bool, list[str]]:
        """
        Check if all specified namespaces exist.

        Returns:
            Tuple of (all_exist: bool, missing_namespaces: List[str])
        """
        existing, _ = self.get_namespaces()
        missing = [ns for ns in namespaces if ns not in existing]
        return len(missing) == 0, missing

    # -------------------------------------------------------------------------
    # PVC Operations
    # -------------------------------------------------------------------------

    def get_pvcs(self, namespace: str) -> tuple[list[str], CommandResult]:
        """Get all PVCs in a namespace."""
        result = self._run_kubectl(f"get pvc -n {namespace} --no-headers -o custom-columns=NAME:.metadata.name")
        pvcs = []

        if result.success:
            pvcs = [pvc.strip() for pvc in result.stdout.split("\n") if pvc.strip()]

        return pvcs, result

    def namespace_has_pvcs(self, namespace: str) -> bool:
        """Check if a namespace has any PVCs."""
        pvcs, result = self.get_pvcs(namespace)
        return result.success and len(pvcs) > 0

    def pvc_phase(self, namespace: str, claim: str) -> tuple[str, CommandResult]:
        """
        Get the phase of a persistent volume claim
        """
        result = self._run_kubectl(f"get pvc {claim} -n {namespace} -o jsonpath='{{.status.phase}}'")
        return result.stdout, result

    # -------------------------------------------------------------------------
    # Service Operations
    # -------------------------------------------------------------------------

    def service_exists(self, name: str, namespace: str) -> bool:
        """Check if a service exists."""
        result = self._run_kubectl(f"get service {name} -n {namespace}")
        return result.success

    def get_service_port(self, name: str, namespace: str) -> int | None:
        """Get the port of a service."""
        result = self._run_kubectl(f"get service {name} -n {namespace} -o jsonpath='{{.spec.ports[0].port}}'")
        if result.success and result.stdout.isdigit():
            return int(result.stdout)
        return None

    def delete_service(self, name: str, namespace: str) -> bool:
        """Delete a service."""
        result = self._run_kubectl(f"delete service {name} -n {namespace}")
        return result.success

    def service_endpoint_addresses(self, namespace: str, service: str) -> list[str]:
        """
        Get the ready endpoint addresses of a service, empty if it has none.

        Not-ready and terminating endpoints are left out, so a wait on this does not
        return while a replacement pod is still starting.
        """
        result = self._run_kubectl(
            f"get endpointslices -n {namespace} -l kubernetes.io/service-name={service} "
            "-o jsonpath='{.items[*].endpoints[?(@.conditions.ready==true)].addresses[*]}'"
        )
        return result.stdout.split() if result.success else []

    def service_pod_report(self, namespace: str, service: str) -> str:
        """
        pod_report() for the pods a service selects.
        """
        svc, result = self._get_json(f"get service {service} -n {namespace} -o json")
        if svc is None:
            return f"could not read svc/{service} in {namespace}: {result.stderr.strip()}"
        selector = svc.get("spec", {}).get("selector") or {}
        if not selector:
            return f"svc/{service} in {namespace} has no selector, so there are no pods to report on"
        return self.pod_report(namespace, ",".join(f"{key}={value}" for key, value in selector.items()))

    # -------------------------------------------------------------------------
    # Config and Secret Operations
    # -------------------------------------------------------------------------

    def current_context(self) -> str | None:
        """Get the kubeconfig context kubectl is using, None when it cannot say."""
        result = self._run_kubectl("config current-context", timeout=10)
        return result.stdout if result.success and result.stdout else None

    def get_secret(self, name: str, namespace: str) -> tuple[dict, CommandResult]:
        """
        Get a secret as its JSON object, empty when it cannot be read.

        The values under "data" are still base64-encoded, as the API returns them.
        """
        secret, result = self._get_json(f"get secret {name} -n {namespace} -o json", timeout=30)
        return secret or {}, result

    # -------------------------------------------------------------------------
    # Cluster Operations
    # -------------------------------------------------------------------------

    def apply_manifest(self, manifest: str, namespace: str) -> CommandResult:
        """
        Apply manifest YAML to a namespace, passing it to kubectl on stdin.

        Takes the YAML itself rather than a path, so a caller that builds its
        manifests in memory does not have to write them to disk first.
        """
        return self._run_kubectl(f"apply -n {namespace} -f -", stdin_data=manifest)

    def apply_manifest_file(self, manifest: Path, namespace: str) -> bool:
        """Apply a manifest file to the cluster."""
        result = self.apply_manifest(manifest.read_text(), namespace)
        if result.success:
            logger.info("applied %s to namespace %s", manifest, namespace)
        else:
            logger.error("applying %s to namespace %s failed: %s", manifest, namespace, result.stderr)
        return result.success


class KubectlPortForwarder:
    """
    Manage kubectl port-forward tunnels via SSH.

    This class:
    1. Runs `kubectl port-forward` on the remote host in the background
    2. Creates an SSH tunnel to forward the local port to the kubectl port

    This allows accessing Kubernetes services from the local machine.
    """

    def __init__(self, ssh_executor: SSHExecutor):
        """
        Initialize KubectlPortForwarder.

        Args:
            ssh_executor: SSHExecutor instance for SSH operations
        """
        self._ssh = ssh_executor
        self._tunnels: list = []
        self._kubectl_pids: list[int] = []

    def _start_ssh_tunnel(
        self,
        local_port: int,
        remote_port: int,
        remote_host: str = "127.0.0.1",
    ) -> bool:
        """
        Start an SSH port forwarding tunnel.

        Args:
            local_port: Local port to bind
            remote_port: Remote port to forward to
            remote_host: Remote host to connect to (default: localhost on remote)

        Returns:
            True if tunnel started successfully
        """
        try:
            transport = self._ssh.get_transport()
            if transport is None:
                return False

            # Create local socket
            local_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            local_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            local_socket.bind(("127.0.0.1", local_port))
            local_socket.listen(1)

            # Start forwarding thread
            tunnel_info = {
                "socket": local_socket,
                "running": True,
                "thread": None,
            }

            def forward_handler():
                while tunnel_info["running"]:
                    try:
                        local_socket.settimeout(1.0)
                        conn, addr = local_socket.accept()
                    except TimeoutError:
                        continue
                    except Exception:
                        break

                    try:
                        channel = transport.open_channel(
                            "direct-tcpip",
                            (remote_host, remote_port),
                            addr,
                        )
                        if channel is None:
                            conn.close()
                            continue

                        # Bidirectional forwarding
                        def forward(src, dst):
                            try:
                                while True:
                                    data = src.recv(4096)
                                    if not data:
                                        break
                                    dst.sendall(data)
                            except Exception:
                                pass

                        t1 = threading.Thread(target=forward, args=(conn, channel))
                        t2 = threading.Thread(target=forward, args=(channel, conn))
                        t1.daemon = True
                        t2.daemon = True
                        t1.start()
                        t2.start()
                    except Exception:
                        conn.close()

            thread = threading.Thread(target=forward_handler)
            thread.daemon = True
            thread.start()
            tunnel_info["thread"] = thread

            self._tunnels.append(tunnel_info)
            return True
        except Exception:
            return False

    def _wait_for_remote_listener(self, port: int, timeout: float) -> bool:
        """Poll the remote host until something is listening on 127.0.0.1:port."""

        def listening() -> bool:
            result = self._ssh.run(f"ss -ltn 'sport = :{port}'")
            if not result.success:
                # No usable `ss` on the host: the bind cannot be verified, so let
                # the caller proceed rather than failing a tunnel that may be fine.
                return True
            return f":{port}" in result.stdout

        return poll_until(listening, timeout=timeout, interval=0.5)

    def start_service_tunnel(
        self,
        local_port: int,
        service_name: str,
        service_port: int,
        namespace: str = "default",
        remote_kubectl_port: int = 0,
        use_sudo: bool = True,
        wait_ready: bool = True,
        ready_timeout: float = 10.0,
    ) -> bool:
        """
        Start a kubectl port-forward and SSH tunnel to a Kubernetes service.

        Args:
            local_port: Local port to bind on the local machine
            service_name: Name of the Kubernetes service (e.g., "grafana")
            service_port: Port on the service to forward to (e.g., 80)
            namespace: Kubernetes namespace (default: "default")
            remote_kubectl_port: Port on remote host for kubectl to bind to.
                                If 0, uses local_port value.
            use_sudo: Whether to use sudo for kubectl (default: True)
            wait_ready: Wait for kubectl to bind the remote port (default: True)
            ready_timeout: Timeout in seconds to wait for ready (default: 10.0)

        Returns:
            True if both kubectl port-forward and SSH tunnel started successfully
        """
        # Use same port on remote as local if not specified
        if remote_kubectl_port == 0:
            remote_kubectl_port = local_port

        # Kill any existing kubectl port-forward on that port
        kill_cmd = f"lsof -ti:{remote_kubectl_port} | xargs kill -9 2>/dev/null || true"
        if use_sudo:
            kill_cmd = f"sudo {kill_cmd}"
        self._ssh.run(kill_cmd)

        # Start kubectl port-forward in background on remote host
        kubectl_cmd = (
            f"kubectl -n {namespace} port-forward "
            f"svc/{service_name} {remote_kubectl_port}:{service_port} "
            f"--address 127.0.0.1"
        )
        if use_sudo:
            kubectl_cmd = f"sudo {kubectl_cmd}"

        # Keep kubectl's own output: it is the only explanation of a failure to
        # bind (missing service, wrong namespace, no kubeconfig permission).
        log_path = f"/tmp/kubectl-port-forward-{remote_kubectl_port}.log"

        # Run kubectl port-forward in background and capture PID
        bg_cmd = f"nohup {kubectl_cmd} > {log_path} 2>&1 & echo $!"
        result = self._ssh.run(bg_cmd)

        if not result.success:
            return False

        # Store PID for cleanup
        try:
            pid = int(result.stdout.strip())
            self._kubectl_pids.append(pid)
        except ValueError:
            pass

        # `nohup ... &` returns as soon as the shell forks, well before kubectl
        # binds the port. Without waiting here the SSH tunnel below comes up over
        # nothing, start_service_tunnel reports success, and the caller's first
        # request dies with "connection reset by peer" instead of a tunnel error.
        if wait_ready and not self._wait_for_remote_listener(remote_kubectl_port, ready_timeout):
            logger.error(
                "kubectl port-forward did not bind 127.0.0.1:%s on the remote host within %ss",
                remote_kubectl_port,
                ready_timeout,
            )
            log = self._ssh.run(f"tail -n 5 {log_path}")
            if log.stdout:
                logger.error("kubectl output:\n%s", log.stdout)
            return False

        # Start SSH tunnel from local port to remote kubectl port
        return self._start_ssh_tunnel(
            local_port=local_port,
            remote_port=remote_kubectl_port,
            remote_host="127.0.0.1",
        )

    def stop_all(self, use_sudo: bool = True):
        """Stop all kubectl port-forward processes and SSH tunnels."""
        # Stop SSH tunnels
        for tunnel in self._tunnels:
            tunnel["running"] = False
            try:
                tunnel["socket"].close()
            except Exception:
                pass
        self._tunnels.clear()

        # Kill kubectl processes on remote
        for pid in self._kubectl_pids:
            kill_cmd = f"kill -9 {pid} 2>/dev/null || true"
            if use_sudo:
                kill_cmd = f"sudo {kill_cmd}"
            self._ssh.run(kill_cmd)

        self._kubectl_pids.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.stop_all()


@dataclass
class LocalPortForward:
    """Local port-forward process."""

    process: subprocess.Popen
    local_port: int
    service: str
    namespace: str


class LocalKubectlPortForwarder:
    """
    Manage kubectl port-forward for local clusters.

    This class runs kubectl port-forward directly on the local machine,
    suitable for when the Kubernetes cluster is running locally.
    """

    def __init__(
        self,
        namespace: str = "default",
        kubeconfig: str | None = None,
    ):
        """
        Initialize LocalKubectlPortForwarder.

        Args:
            namespace: Default Kubernetes namespace (default: "default")
            kubeconfig: Path to kubeconfig file (default: None, uses kubectl default)
        """
        self.namespace = namespace
        self.kubeconfig = kubeconfig
        self._forwards: list[LocalPortForward] = []

    def start_service_tunnel(
        self,
        local_port: int,
        service_name: str,
        service_port: int,
        namespace: str | None = None,
        wait_ready: bool = True,
        ready_timeout: float = 10.0,
    ) -> bool:
        """
        Start kubectl port-forward to a Kubernetes service.

        Args:
            local_port: Local port to bind
            service_name: Name of the Kubernetes service (e.g., "grafana")
            service_port: Port on the service to forward to (e.g., 80)
            namespace: Kubernetes namespace (default: uses instance default)
            wait_ready: Wait for the port-forward to be ready (default: True)
            ready_timeout: Timeout in seconds to wait for ready (default: 10.0)

        Returns:
            True if port-forward started successfully
        """
        ns = namespace or self.namespace

        cmd = [
            "kubectl",
            "-n",
            ns,
            "port-forward",
            f"svc/{service_name}",
            f"{local_port}:{service_port}",
            "--address",
            "127.0.0.1",
        ]

        if self.kubeconfig:
            cmd.extend(["--kubeconfig", self.kubeconfig])

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            # Wait for port-forward to be ready
            if wait_ready:
                start_time = time.time()
                while time.time() - start_time < ready_timeout:
                    # Check if process failed
                    if proc.poll() is not None:
                        return False

                    # Try to connect to the port
                    try:
                        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        sock.settimeout(0.5)
                        result = sock.connect_ex(("127.0.0.1", local_port))
                        sock.close()
                        if result == 0:
                            break
                    except Exception:
                        pass

                    time.sleep(0.2)

            self._forwards.append(
                LocalPortForward(
                    process=proc,
                    local_port=local_port,
                    service=service_name,
                    namespace=ns,
                )
            )
            return True
        except Exception:
            return False

    def stop_service_tunnel(self, service_name: str, namespace: str | None = None) -> bool:
        """
        Stop a specific port-forward by service name.

        Args:
            service_name: Name of the service to stop forwarding
            namespace: Kubernetes namespace (default: uses instance default)

        Returns:
            True if a tunnel was found and stopped
        """
        ns = namespace or self.namespace
        for fwd in self._forwards:
            if fwd.service == service_name and fwd.namespace == ns:
                fwd.process.terminate()
                try:
                    fwd.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    fwd.process.kill()
                self._forwards.remove(fwd)
                return True
        return False

    def stop_all(self):
        """Stop all port-forward processes."""
        for fwd in self._forwards:
            fwd.process.terminate()
            try:
                fwd.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                fwd.process.kill()
        self._forwards.clear()

    def is_running(self, service_name: str, namespace: str | None = None) -> bool:
        """
        Check if a port-forward is running for a service.

        Args:
            service_name: Name of the service
            namespace: Kubernetes namespace (default: uses instance default)

        Returns:
            True if port-forward is running
        """
        ns = namespace or self.namespace
        for fwd in self._forwards:
            if fwd.service == service_name and fwd.namespace == ns:
                return fwd.process.poll() is None
        return False

    def get_local_port(self, service_name: str, namespace: str | None = None) -> int | None:
        """
        Get the local port for a service's port-forward.

        Args:
            service_name: Name of the service
            namespace: Kubernetes namespace (default: uses instance default)

        Returns:
            Local port number or None if not found
        """
        ns = namespace or self.namespace
        for fwd in self._forwards:
            if fwd.service == service_name and fwd.namespace == ns:
                return fwd.local_port
        return None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.stop_all()
