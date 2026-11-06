# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2025 Delos Data, Inc.

"""Unit tests for k8s module."""

import json
import shlex
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from production_test_framework.config import LGTMConfig
from production_test_framework.k8s import (
    KubectlPortForwarder,
    KubernetesClient,
    LocalKubectlPortForwarder,
    LocalPortForward,
    Node,
    Pod,
)
from production_test_framework.ssh import CommandResult


class TestNode:
    """Tests for Node dataclass."""

    def test_is_ready_true(self):
        node = Node(name="node1", status="Ready", roles="control-plane", version="v1.28")
        assert node.is_ready is True

    def test_is_ready_false(self):
        node = Node(name="node1", status="NotReady", roles="control-plane", version="v1.28")
        assert node.is_ready is False


class TestPod:
    """Tests for Pod dataclass."""

    def test_is_running_true(self):
        pod = Pod("pod1", "default", "Running", "1/1", 0, "5m")
        assert pod.is_running is True

    def test_is_running_false(self):
        pod = Pod("pod1", "default", "Pending", "0/1", 0, "1m")
        assert pod.is_running is False

    def test_is_completed_true(self):
        pod = Pod("pod1", "default", "Completed", "1/1", 0, "10m")
        assert pod.is_completed is True

    def test_is_ready_true(self):
        pod = Pod("pod1", "default", "Running", "2/2", 0, "5m")
        assert pod.is_ready is True

    def test_is_ready_false_no_slash(self):
        pod = Pod("pod1", "default", "Running", "1", 0, "5m")
        assert pod.is_ready is False

    def test_is_ready_false_partial(self):
        pod = Pod("pod1", "default", "Running", "1/2", 0, "5m")
        assert pod.is_ready is False


class TestKubernetesClient:
    """Tests for KubernetesClient with mocked SSH."""

    @pytest.fixture
    def mock_ssh(self):
        ssh = MagicMock()
        ssh.run_kubectl = MagicMock()
        return ssh

    @pytest.fixture
    def k8s_client(self, lgtm_config, mock_ssh):
        return KubernetesClient(lgtm_config, ssh=mock_ssh)

    def test_get_nodes_parses_output(self, k8s_client, mock_ssh):
        mock_ssh.run_kubectl.return_value = CommandResult(
            returncode=0,
            stdout="node1   Ready   control-plane   1.28   192.168.1.1\nnode2   Ready   <none>   1.28   192.168.1.2\n",
            stderr="",
        )

        nodes, result = k8s_client.get_nodes()

        assert result.success is True
        assert len(nodes) == 2
        assert nodes[0].name == "node1"
        assert nodes[0].status == "Ready"
        assert nodes[0].internal_ip is None
        assert nodes[1].name == "node2"

    def test_get_node_count(self, k8s_client, mock_ssh):
        mock_ssh.run_kubectl.return_value = CommandResult(
            returncode=0,
            stdout="node1   Ready   control-plane   1.28   192.168.1.1\n",
            stderr="",
        )
        assert k8s_client.get_node_count() == 1

    def test_all_nodes_ready_true(self, k8s_client, mock_ssh):
        mock_ssh.run_kubectl.return_value = CommandResult(
            returncode=0,
            stdout="node1   Ready   control-plane   1.28   192.168.1.1\n",
            stderr="",
        )
        assert k8s_client.all_nodes_ready() is True

    def test_all_nodes_ready_false_when_not_ready(self, k8s_client, mock_ssh):
        mock_ssh.run_kubectl.return_value = CommandResult(
            returncode=0,
            stdout="node1   NotReady   control-plane   1.28   192.168.1.1\n",
            stderr="",
        )
        assert k8s_client.all_nodes_ready() is False

    @staticmethod
    def _nodes_json(*nodes: tuple[str, dict[str, tuple[str, str]]]) -> str:
        """`get nodes -o json` for (name, {condition type: (status, message)}) pairs."""
        return json.dumps(
            {
                "items": [
                    {
                        "metadata": {"name": name},
                        "status": {
                            "conditions": [
                                {"type": kind, "status": status, "reason": "KubeletReady", "message": message}
                                for kind, (status, message) in conditions.items()
                            ]
                        },
                    }
                    for name, conditions in nodes
                ]
            }
        )

    def test_node_conditions_healthy_node(self, k8s_client, mock_ssh):
        mock_ssh.run_kubectl.return_value = CommandResult(
            returncode=0,
            stdout=self._nodes_json(("node1", {"Ready": ("True", ""), "DiskPressure": ("False", "")})),
            stderr="",
        )

        lines, result = k8s_client.node_conditions()

        assert result.success is True
        assert lines == ["node1: Ready=True (KubeletReady), no pressure conditions"]

    def test_node_conditions_reports_only_true_pressure(self, k8s_client, mock_ssh):
        mock_ssh.run_kubectl.return_value = CommandResult(
            returncode=0,
            stdout=self._nodes_json(
                (
                    "node1",
                    {
                        "Ready": ("True", ""),
                        "DiskPressure": ("True", "kubelet has disk pressure"),
                        "MemoryPressure": ("False", "kubelet has sufficient memory"),
                    },
                ),
                ("node2", {"Ready": ("False", "")}),
            ),
            stderr="",
        )

        lines, _ = k8s_client.node_conditions()

        assert lines == [
            "node1: Ready=True (KubeletReady), DiskPressure (kubelet has disk pressure)",
            "node2: Ready=False (KubeletReady), no pressure conditions",
        ]

    def test_node_conditions_kubectl_failure(self, k8s_client, mock_ssh):
        mock_ssh.run_kubectl.return_value = CommandResult(returncode=1, stdout="", stderr="connection refused")

        lines, result = k8s_client.node_conditions()

        assert lines == []
        assert result.success is False

    def test_get_pods_all_namespaces(self, k8s_client, mock_ssh):
        mock_ssh.run_kubectl.return_value = CommandResult(
            returncode=0,
            stdout="default   pod1  1/1   Running   0   5m\nkube-system   coredns  2/2   Running   0   10m\n",
            stderr="",
        )

        pods, result = k8s_client.get_pods(all_namespaces=True)

        assert result.success is True
        assert len(pods) == 2
        assert pods[0].namespace == "default" and pods[0].name == "pod1"
        assert pods[1].namespace == "kube-system" and pods[1].name == "coredns"
        mock_ssh.run_kubectl.assert_called_with("get pods -A --no-headers", timeout=60, stdin_data=None)

    def test_get_pods_single_namespace(self, k8s_client, mock_ssh):
        mock_ssh.run_kubectl.return_value = CommandResult(
            returncode=0,
            stdout="pod1  1/1   Running   0   5m\n",
            stderr="",
        )

        pods, _ = k8s_client.get_pods(namespace="default")

        assert len(pods) == 1
        assert pods[0].namespace == "default"
        assert pods[0].name == "pod1"
        mock_ssh.run_kubectl.assert_called_with("get pods -n default --no-headers", timeout=60, stdin_data=None)

    def test_get_pods_in_namespace(self, k8s_client, mock_ssh):
        mock_ssh.run_kubectl.return_value = CommandResult(
            returncode=0,
            stdout="pod1  1/1   Running   0   5m\n",
            stderr="",
        )
        pods = k8s_client.get_pods_in_namespace("default")
        assert len(pods) == 1
        assert pods[0].name == "pod1"

    def test_all_pods_running(self, k8s_client, mock_ssh):
        mock_ssh.run_kubectl.return_value = CommandResult(
            returncode=0,
            stdout="pod1  1/1   Running   0   5m\npod2  1/1   Running   0   5m\n",
            stderr="",
        )
        assert k8s_client.all_pods_running("default") is True

    def test_all_pods_ready_includes_completed(self, k8s_client, mock_ssh):
        mock_ssh.run_kubectl.return_value = CommandResult(
            returncode=0,
            stdout="job-pod  1/1   Completed   0   5m\n",
            stderr="",
        )
        assert k8s_client.all_pods_ready("default") is True

    def test_get_namespaces(self, k8s_client, mock_ssh):
        mock_ssh.run_kubectl.return_value = CommandResult(
            returncode=0,
            stdout="default\nkube-system\nmonitoring\n",
            stderr="",
        )
        namespaces, result = k8s_client.get_namespaces()
        assert result.success is True
        assert set(namespaces) == {"default", "kube-system", "monitoring"}

    def test_namespace_exists_true(self, k8s_client, mock_ssh):
        mock_ssh.run_kubectl.return_value = CommandResult(returncode=0, stdout="", stderr="")
        assert k8s_client.namespace_exists("default") is True

    def test_namespace_exists_false(self, k8s_client, mock_ssh):
        mock_ssh.run_kubectl.return_value = CommandResult(returncode=1, stdout="", stderr="")
        assert k8s_client.namespace_exists("missing") is False

    def test_all_namespaces_exist(self, k8s_client, mock_ssh):
        mock_ssh.run_kubectl.return_value = CommandResult(
            returncode=0,
            stdout="default\nkube-system\n",
            stderr="",
        )
        all_exist, missing = k8s_client.all_namespaces_exist(["default", "kube-system", "other"])
        assert all_exist is False
        assert missing == ["other"]

    def test_get_pvcs(self, k8s_client, mock_ssh):
        mock_ssh.run_kubectl.return_value = CommandResult(
            returncode=0,
            stdout="pvc-1\npvc-2\n",
            stderr="",
        )
        pvcs, result = k8s_client.get_pvcs("default")
        assert result.success is True
        assert pvcs == ["pvc-1", "pvc-2"]

    def test_namespace_has_pvcs_true(self, k8s_client, mock_ssh):
        mock_ssh.run_kubectl.return_value = CommandResult(
            returncode=0,
            stdout="pvc-1\n",
            stderr="",
        )
        assert k8s_client.namespace_has_pvcs("default") is True

    def test_service_exists(self, k8s_client, mock_ssh):
        mock_ssh.run_kubectl.return_value = CommandResult(returncode=0, stdout="", stderr="")
        assert k8s_client.service_exists("grafana", "default") is True

    def test_get_service_port(self, k8s_client, mock_ssh):
        mock_ssh.run_kubectl.return_value = CommandResult(
            returncode=0,
            stdout="3000",
            stderr="",
        )
        assert k8s_client.get_service_port("grafana", "default") == 3000

    def test_get_service_port_invalid_returns_none(self, k8s_client, mock_ssh):
        mock_ssh.run_kubectl.return_value = CommandResult(
            returncode=0,
            stdout="",
            stderr="",
        )
        assert k8s_client.get_service_port("svc", "default") is None

    def test_delete_service(self, k8s_client, mock_ssh):
        mock_ssh.run_kubectl.return_value = CommandResult(returncode=0, stdout="", stderr="")
        assert k8s_client.delete_service("grafana", "default") is True

    def test_wait_for_pods_ready(self, k8s_client, mock_ssh):
        mock_ssh.run_kubectl.return_value = CommandResult(returncode=0, stdout="", stderr="")
        assert k8s_client.wait_for_pods_ready("default", "app=grafana", timeout=60) is True
        call_args = mock_ssh.run_kubectl.call_args[0][0]
        assert "wait" in call_args
        assert "condition=Ready" in call_args
        assert "app=grafana" in call_args
        assert "timeout=60s" in call_args

    def test_apply_manifest_file(self, k8s_client, mock_ssh):
        mock_ssh.run_kubectl.return_value = CommandResult(returncode=0, stdout="", stderr="")
        manifest = Path("/tmp/manifest.yaml")
        with patch.object(Path, "read_text", return_value="apiVersion: v1\nkind: ConfigMap\n"):
            result = k8s_client.apply_manifest_file(manifest, "default")
        assert result is True
        mock_ssh.run_kubectl.assert_called_once()
        assert mock_ssh.run_kubectl.call_args[1]["stdin_data"] == "apiVersion: v1\nkind: ConfigMap\n"

    def test_apply_manifest_file_failure(self, k8s_client, mock_ssh):
        mock_ssh.run_kubectl.return_value = CommandResult(returncode=1, stdout="", stderr="invalid")
        with patch.object(Path, "read_text", return_value="kind: ConfigMap\n"):
            assert k8s_client.apply_manifest_file(Path("/tmp/manifest.yaml"), "default") is False

    def test_apply_manifest_passes_text_on_stdin(self, k8s_client, mock_ssh):
        mock_ssh.run_kubectl.return_value = CommandResult(returncode=0, stdout="configmap/x created", stderr="")
        result = k8s_client.apply_manifest("apiVersion: v1\nkind: ConfigMap\n", "lgtma")
        assert result.success
        assert result.stdout == "configmap/x created"
        args, kwargs = mock_ssh.run_kubectl.call_args
        assert args[0] == "apply -n lgtma -f -"
        assert kwargs["stdin_data"] == "apiVersion: v1\nkind: ConfigMap\n"

    def test_apply_manifest_returns_kubectl_error(self, k8s_client, mock_ssh):
        mock_ssh.run_kubectl.return_value = CommandResult(returncode=1, stdout="", stderr="error: bad manifest")
        result = k8s_client.apply_manifest("not: [valid", "default")
        assert not result.success
        assert result.stderr == "error: bad manifest"

    def test_current_context(self, k8s_client, mock_ssh):
        mock_ssh.run_kubectl.return_value = CommandResult(returncode=0, stdout="k3d-mc", stderr="")
        assert k8s_client.current_context() == "k3d-mc"
        assert mock_ssh.run_kubectl.call_args[0][0] == "config current-context"

    def test_current_context_none_when_unset(self, k8s_client, mock_ssh):
        mock_ssh.run_kubectl.return_value = CommandResult(returncode=1, stdout="", stderr="current-context is not set")
        assert k8s_client.current_context() is None

    def test_get_secret_parses_json(self, k8s_client, mock_ssh):
        payload = {"kind": "Secret", "data": {"client-id": "bWM="}}
        mock_ssh.run_kubectl.return_value = CommandResult(returncode=0, stdout=json.dumps(payload), stderr="")
        secret, result = k8s_client.get_secret("mc-auth", "mc")
        assert secret == payload
        assert result.success
        assert mock_ssh.run_kubectl.call_args[0][0] == "get secret mc-auth -n mc -o json"

    def test_get_secret_empty_on_bad_json(self, k8s_client, mock_ssh):
        mock_ssh.run_kubectl.return_value = CommandResult(returncode=0, stdout="Error: not json", stderr="")

        secret, result = k8s_client.get_secret("auth", "ns")

        assert secret == {}
        assert result.success is False
        assert "did not return JSON" in result.stderr

    def test_get_json_parses_output(self, k8s_client, mock_ssh):
        mock_ssh.run_kubectl.return_value = CommandResult(returncode=0, stdout='{"kind": "Service"}', stderr="")

        obj, result = k8s_client._get_json("get service svc -n ns -o json", timeout=20)

        assert obj == {"kind": "Service"}
        assert result.success is True
        mock_ssh.run_kubectl.assert_called_once_with("get service svc -n ns -o json", timeout=20, stdin_data=None)

    def test_get_json_passes_kubectl_failure_through(self, k8s_client, mock_ssh):
        failure = CommandResult(returncode=1, stdout="", stderr="NotFound")
        mock_ssh.run_kubectl.return_value = failure
        assert k8s_client._get_json("get service svc -n ns -o json") == (None, failure)

    def test_service_pod_report_bad_json(self, k8s_client, mock_ssh):
        mock_ssh.run_kubectl.return_value = CommandResult(returncode=0, stdout="<html>", stderr="")
        assert k8s_client.service_pod_report("ns", "svc").startswith(
            "could not read svc/svc in ns: kubectl get service"
        )

    def test_get_secret_empty_on_failure(self, k8s_client, mock_ssh):
        mock_ssh.run_kubectl.return_value = CommandResult(returncode=1, stdout="", stderr='secrets "x" not found')
        secret, result = k8s_client.get_secret("x", "mc")
        assert secret == {}
        assert result.stderr == 'secrets "x" not found'


class TestKubernetesClientLocalKubeconfig:
    """Tests for which kubeconfig local kubectl is pointed at."""

    @pytest.fixture
    def local_client(self):
        return KubernetesClient(LGTMConfig(host="localhost"), ssh=MagicMock())

    def _kubectl_cmd(self, local_client) -> list[str]:
        with patch("production_test_framework.k8s.run_command") as run:
            run.return_value = CommandResult(returncode=0, stdout="", stderr="")
            local_client._run_kubectl("get nodes")
        return run.call_args[0][0]

    def test_defaults_to_home_kubeconfig(self, local_client, monkeypatch):
        monkeypatch.delenv("KUBECONFIG", raising=False)
        cmd = self._kubectl_cmd(local_client)
        assert cmd[:3] == ["kubectl", "--kubeconfig", str(Path("~/.kube/config").expanduser())]

    def test_leaves_kubeconfig_env_to_kubectl(self, local_client, monkeypatch):
        monkeypatch.setenv("KUBECONFIG", "/tmp/a:/tmp/b")
        assert self._kubectl_cmd(local_client) == ["kubectl", "get", "nodes"]

    def test_explicit_kubeconfig_wins(self, local_client, monkeypatch):
        monkeypatch.setenv("KUBECONFIG", "/tmp/a")
        with patch("production_test_framework.k8s.run_command") as run:
            run.return_value = CommandResult(returncode=0, stdout="", stderr="")
            local_client._run_kubectl_local("get nodes", kubeconfig="/tmp/explicit")
        assert run.call_args[0][0] == ["kubectl", "--kubeconfig", "/tmp/explicit", "get", "nodes"]


class TestKubernetesClientWorkloadState:
    """Tests for the pod, workload, PVC and endpoint readers with mocked SSH."""

    @pytest.fixture
    def mock_ssh(self):
        ssh = MagicMock()
        ssh.run_kubectl = MagicMock()
        return ssh

    @pytest.fixture
    def k8s_client(self, lgtm_config, mock_ssh):
        return KubernetesClient(lgtm_config, ssh=mock_ssh)

    # -------------------------------------------------------------------------
    # pods_by_selector
    # -------------------------------------------------------------------------

    def test_pods_by_selector_parses_uid_and_readiness(self, k8s_client, mock_ssh):
        mock_ssh.run_kubectl.return_value = CommandResult(
            returncode=0,
            stdout="loki-write-0=uid-a=true true;loki-write-1=uid-b=true false;",
            stderr="",
        )

        pods = k8s_client.pods_by_selector("lgtma", "app=loki")

        assert pods == {"loki-write-0": ("uid-a", True), "loki-write-1": ("uid-b", False)}

    def test_pods_by_selector_marks_pod_without_containers_not_ready(self, k8s_client, mock_ssh):
        mock_ssh.run_kubectl.return_value = CommandResult(returncode=0, stdout="loki-write-0=uid-a=;", stderr="")
        assert k8s_client.pods_by_selector("lgtma", "app=loki") == {"loki-write-0": ("uid-a", False)}

    def test_pods_by_selector_empty_when_kubectl_fails(self, k8s_client, mock_ssh):
        mock_ssh.run_kubectl.return_value = CommandResult(returncode=1, stdout="", stderr="boom")
        assert k8s_client.pods_by_selector("lgtma", "app=loki") == {}

    # -------------------------------------------------------------------------
    # pod_containers
    # -------------------------------------------------------------------------

    def test_pod_containers_parses_container_names(self, k8s_client, mock_ssh):
        mock_ssh.run_kubectl.return_value = CommandResult(
            returncode=0,
            stdout="grafana-0=grafana sc-dashboard sc-alert;loki-gateway-0=nginx;",
            stderr="",
        )

        pods, result = k8s_client.pod_containers("lgtma", "app=grafana")

        assert result.success is True
        assert pods == {"grafana-0": {"grafana", "sc-dashboard", "sc-alert"}, "loki-gateway-0": {"nginx"}}

    def test_pod_containers_empty_when_kubectl_fails(self, k8s_client, mock_ssh):
        mock_ssh.run_kubectl.return_value = CommandResult(returncode=1, stdout="", stderr="boom")

        pods, result = k8s_client.pod_containers("lgtma", "app=grafana")

        assert pods == {}
        assert result.success is False

    # -------------------------------------------------------------------------
    # pod_metrics
    # -------------------------------------------------------------------------

    def test_pod_metrics_reads_through_the_api_server_proxy(self, k8s_client, mock_ssh):
        mock_ssh.run_kubectl.return_value = CommandResult(returncode=0, stdout="up 1\n", stderr="")

        metrics, result = k8s_client.pod_metrics("lgtma", "loki-write-0", 3100)

        assert result.success is True
        assert metrics == "up 1\n"
        assert mock_ssh.run_kubectl.call_args[0][0] == (
            "get --raw /api/v1/namespaces/lgtma/pods/loki-write-0:3100/proxy/metrics"
        )

    def test_pod_metrics_reports_failure(self, k8s_client, mock_ssh):
        mock_ssh.run_kubectl.return_value = CommandResult(returncode=1, stdout="", stderr="no route to pod")

        metrics, result = k8s_client.pod_metrics("lgtma", "loki-write-0", 3100)

        assert metrics == ""
        assert result.success is False
        assert result.stderr == "no route to pod"

    # -------------------------------------------------------------------------
    # workload_readiness
    # -------------------------------------------------------------------------

    @pytest.mark.parametrize(
        "kind,field",
        [("deployment", ".spec.replicas"), ("statefulset", ".spec.replicas"), ("daemonset", ".status.numberReady")],
    )
    def test_workload_readiness_uses_the_fields_of_each_kind(self, k8s_client, mock_ssh, kind, field):
        mock_ssh.run_kubectl.return_value = CommandResult(returncode=0, stdout="3,3,3", stderr="")

        counts, result = k8s_client.workload_readiness("lgtma", kind, "loki-read")

        assert result.success is True
        assert counts == (3, 3, 3)
        assert field in mock_ssh.run_kubectl.call_args[0][0]

    def test_workload_readiness_treats_absent_counts_as_zero(self, k8s_client, mock_ssh):
        mock_ssh.run_kubectl.return_value = CommandResult(returncode=0, stdout="2,,", stderr="")
        assert k8s_client.workload_readiness("lgtma", "deployment", "loki-read")[0] == (2, 0, 0)

    def test_workload_readiness_rejects_unknown_kind_without_calling_kubectl(self, k8s_client, mock_ssh):
        counts, result = k8s_client.workload_readiness("lgtma", "cronjob", "loki-read")

        assert counts == (0, 0, 0)
        assert result.success is False
        assert "cronjob" in result.stderr
        mock_ssh.run_kubectl.assert_not_called()

    def test_workload_readiness_reports_unexpected_output(self, k8s_client, mock_ssh):
        mock_ssh.run_kubectl.return_value = CommandResult(returncode=0, stdout="3,3", stderr="")

        counts, result = k8s_client.workload_readiness("lgtma", "deployment", "loki-read")

        assert counts == (0, 0, 0)
        assert result.success is False

    def test_workload_readiness_reports_kubectl_failure(self, k8s_client, mock_ssh):
        mock_ssh.run_kubectl.return_value = CommandResult(returncode=1, stdout="", stderr="not found")

        counts, result = k8s_client.workload_readiness("lgtma", "deployment", "missing")

        assert counts == (0, 0, 0)
        assert result.stderr == "not found"

    # -------------------------------------------------------------------------
    # deployment volumes and rollouts
    # -------------------------------------------------------------------------

    @staticmethod
    def _deployment_json(*volumes):
        return json.dumps({"spec": {"template": {"spec": {"volumes": list(volumes)}}}})

    def test_deployment_volume_returns_named_volume(self, k8s_client, mock_ssh):
        pvc = {"name": "data", "persistentVolumeClaim": {"claimName": "data"}}
        mock_ssh.run_kubectl.return_value = CommandResult(
            returncode=0, stdout=self._deployment_json({"name": "config", "emptyDir": {}}, pvc), stderr=""
        )

        spec, result = k8s_client.deployment_volume("ns", "app", "data")

        assert result.success is True
        assert spec == pvc
        mock_ssh.run_kubectl.assert_called_once_with("get deployment app -n ns -o json", timeout=60, stdin_data=None)

    def test_deployment_volume_missing_is_none(self, k8s_client, mock_ssh):
        mock_ssh.run_kubectl.return_value = CommandResult(returncode=0, stdout=self._deployment_json(), stderr="")
        assert k8s_client.deployment_volume("ns", "app", "data")[0] is None

    def test_deployment_volume_reports_bad_json(self, k8s_client, mock_ssh):
        mock_ssh.run_kubectl.return_value = CommandResult(returncode=0, stdout="not json", stderr="")

        spec, result = k8s_client.deployment_volume("ns", "app", "data")

        assert spec is None
        assert result.success is False

    def test_replace_deployment_volume_patches_by_index_and_waits(self, k8s_client, mock_ssh):
        ok = CommandResult(returncode=0, stdout="", stderr="")
        mock_ssh.run_kubectl.side_effect = [
            CommandResult(returncode=0, stdout=self._deployment_json({"name": "config"}, {"name": "data"}), stderr=""),
            ok,
            ok,
        ]
        spec = {"name": "data", "emptyDir": {"medium": "Memory", "sizeLimit": "32Mi"}}

        result = k8s_client.replace_deployment_volume("ns", "app", spec, timeout=90)

        assert result.success is True
        patch_args = mock_ssh.run_kubectl.call_args_list[1].args[0]
        assert patch_args.startswith("patch deployment app -n ns --type=json -p ")
        patch = json.loads(shlex.split(patch_args)[-1])
        assert patch == [
            {"op": "test", "path": "/spec/template/spec/volumes/1/name", "value": "data"},
            {"op": "replace", "path": "/spec/template/spec/volumes/1", "value": spec},
        ]
        assert mock_ssh.run_kubectl.call_args_list[2].args[0] == "rollout status deployment/app -n ns --timeout=90s"
        assert mock_ssh.run_kubectl.call_args_list[2].kwargs["timeout"] == 120

    def test_replace_deployment_volume_unknown_volume_does_not_patch(self, k8s_client, mock_ssh):
        mock_ssh.run_kubectl.return_value = CommandResult(
            returncode=0, stdout=self._deployment_json({"name": "config"}), stderr=""
        )

        result = k8s_client.replace_deployment_volume("ns", "app", {"name": "data", "emptyDir": {}})

        assert result.success is False
        assert mock_ssh.run_kubectl.call_count == 1

    def test_replace_deployment_volume_stops_when_patch_fails(self, k8s_client, mock_ssh):
        mock_ssh.run_kubectl.side_effect = [
            CommandResult(returncode=0, stdout=self._deployment_json({"name": "data"}), stderr=""),
            CommandResult(returncode=1, stdout="", stderr="the server rejected our request"),
        ]

        result = k8s_client.replace_deployment_volume("ns", "app", {"name": "data", "emptyDir": {}})

        assert result.stderr == "the server rejected our request"
        assert mock_ssh.run_kubectl.call_count == 2

    # -------------------------------------------------------------------------
    # pvc_phase
    # -------------------------------------------------------------------------

    def test_pvc_phase_returns_phase(self, k8s_client, mock_ssh):
        mock_ssh.run_kubectl.return_value = CommandResult(returncode=0, stdout="Bound", stderr="")

        phase, result = k8s_client.pvc_phase("lgtma", "storage-loki-write-0")

        assert result.success is True
        assert phase == "Bound"

    def test_pvc_phase_reports_failure(self, k8s_client, mock_ssh):
        mock_ssh.run_kubectl.return_value = CommandResult(returncode=1, stdout="", stderr="not found")

        phase, result = k8s_client.pvc_phase("lgtma", "missing")

        assert phase == ""
        assert result.success is False

    # -------------------------------------------------------------------------
    # service_endpoint_addresses
    # -------------------------------------------------------------------------

    def test_service_endpoint_addresses_reads_endpointslices(self, k8s_client, mock_ssh):
        mock_ssh.run_kubectl.return_value = CommandResult(returncode=0, stdout="10.42.0.5 10.42.1.7", stderr="")

        addresses = k8s_client.service_endpoint_addresses("lgtma", "loki-gateway")

        assert addresses == ["10.42.0.5", "10.42.1.7"]
        assert "endpointslices" in mock_ssh.run_kubectl.call_args[0][0]
        assert "kubernetes.io/service-name=loki-gateway" in mock_ssh.run_kubectl.call_args[0][0]

    def test_service_endpoint_addresses_empty_when_kubectl_fails(self, k8s_client, mock_ssh):
        mock_ssh.run_kubectl.return_value = CommandResult(returncode=1, stdout="", stderr="boom")
        assert k8s_client.service_endpoint_addresses("lgtma", "loki-gateway") == []

    def test_service_endpoint_addresses_only_asks_for_ready_endpoints(self, k8s_client, mock_ssh):
        mock_ssh.run_kubectl.return_value = CommandResult(returncode=0, stdout="10.42.0.5", stderr="")

        k8s_client.service_endpoint_addresses("lgtma", "grafana")

        assert "endpoints[?(@.conditions.ready==true)]" in mock_ssh.run_kubectl.call_args[0][0]

    # -------------------------------------------------------------------------
    # pod_report / service_pod_report
    # -------------------------------------------------------------------------

    @staticmethod
    def _fake_kubectl(responses: dict[str, CommandResult]):
        """
        A run_kubectl stand-in answering by the first key found in the command.

        Unmatched commands fail, so a test only has to describe what it cares about.
        """

        def run(args, **_kwargs):
            for key, response in responses.items():
                if key in args:
                    return response
            return CommandResult(returncode=1, stdout="", stderr=f"unexpected: {args}")

        return run

    @staticmethod
    def _ok(stdout: str) -> CommandResult:
        return CommandResult(returncode=0, stdout=stdout, stderr="")

    @staticmethod
    def _report_pod_json(name: str, *, init: tuple[str, ...] = (), containers: dict[str, int] | None = None) -> str:
        """`get pods -o json` for one pod: init container names, and {container: restartCount}."""
        return json.dumps(
            {
                "items": [
                    {
                        "metadata": {"name": name},
                        "status": {
                            "initContainerStatuses": [{"name": c, "restartCount": 0} for c in init],
                            "containerStatuses": [
                                {"name": c, "restartCount": restarts} for c, restarts in (containers or {}).items()
                            ],
                        },
                    }
                ]
            }
        )

    def test_pod_report_covers_a_pod_stuck_initializing(self, k8s_client, mock_ssh):
        mock_ssh.run_kubectl.side_effect = self._fake_kubectl(
            {
                "-o wide": self._ok("NAME      READY   STATUS     RESTARTS\ngrafana-0 0/1     Init:0/1   0\n"),
                "get pods -n lgtma -l app=grafana -o json": self._ok(
                    self._report_pod_json("grafana-0", init=("init-chown-data",), containers={"grafana": 0})
                ),
                "get events": self._ok("LAST SEEN   TYPE   REASON\n5s   Normal   Pulling\n"),
                "-c init-chown-data": self._ok("chown: changing ownership\n"),
                "-c grafana": CommandResult(returncode=1, stdout="", stderr="waiting to start: PodInitializing"),
                "get nodes -o json": self._ok(json.dumps({"items": []})),
            }
        )

        report = k8s_client.pod_report("lgtma", "app=grafana")

        assert "--- pods matching app=grafana in lgtma" in report
        assert "Init:0/1" in report
        assert "5s   Normal   Pulling" in report
        # The init container a Pending pod is stuck in is reported on its own ...
        assert "--- logs of grafana-0/init-chown-data (last 60 lines)\n  chown: changing ownership" in report
        # ... and a container that has not started yet reports why instead of hiding the others.
        assert "waiting to start: PodInitializing" in report
        assert "--- node conditions\n  none" in report
        commands = [call[0][0] for call in mock_ssh.run_kubectl.call_args_list]
        assert not any("--previous" in command for command in commands)

    def test_pod_report_fetches_previous_logs_for_a_restarted_container(self, k8s_client, mock_ssh):
        mock_ssh.run_kubectl.side_effect = self._fake_kubectl(
            {
                "-o wide": self._ok(
                    "NAME   READY   STATUS             RESTARTS\nweb-0  0/1     CrashLoopBackOff   3\n"
                ),
                "get pods -n apps -l app=web -o json": self._ok(self._report_pod_json("web-0", containers={"web": 3})),
                "get events": self._ok(""),
                "--previous": self._ok("panic: database is locked\n"),
                "-c web": self._ok("starting\n"),
                "get nodes -o json": self._ok(json.dumps({"items": []})),
            }
        )

        report = k8s_client.pod_report("apps", "app=web")

        assert "--- logs of web-0/web (last 60 lines)\n  starting" in report
        assert "--- logs of web-0/web (previous, crashed) (last 60 lines)\n  panic: database is locked" in report

    def test_pod_report_keeps_only_the_latest_events(self, k8s_client, mock_ssh):
        events = "LAST SEEN   TYPE   REASON\n" + "".join(f"{n}s   Normal   Event{n}\n" for n in range(10))
        mock_ssh.run_kubectl.side_effect = self._fake_kubectl(
            {
                "-o wide": self._ok("NAME   READY\nweb-0  1/1\n"),
                "get pods -n apps -l app=web -o json": self._ok(self._report_pod_json("web-0")),
                "get events": self._ok(events),
                "get nodes -o json": self._ok(json.dumps({"items": []})),
            }
        )

        report = k8s_client.pod_report("apps", "app=web", events=2)

        assert "--- events for web-0 (last 2)\n  8s   Normal   Event8\n  9s   Normal   Event9\n" in report
        assert "Event7" not in report

    def test_pod_report_when_pods_cannot_be_listed(self, k8s_client, mock_ssh):
        mock_ssh.run_kubectl.return_value = CommandResult(returncode=1, stdout="", stderr="forbidden")

        report = k8s_client.pod_report("apps", "app=web")

        assert report == "could not list pods matching app=web in apps: forbidden"

    def test_service_pod_report_reports_on_the_service_selector(self, k8s_client):
        service = {"spec": {"selector": {"app.kubernetes.io/name": "grafana", "app.kubernetes.io/instance": "grafana"}}}
        with (
            patch.object(k8s_client, "_run_kubectl", return_value=self._ok(json.dumps(service))) as run,
            patch.object(k8s_client, "pod_report", return_value="report") as pod_report,
        ):
            assert k8s_client.service_pod_report("lgtma", "grafana") == "report"

        assert run.call_args[0][0] == "get service grafana -n lgtma -o json"
        pod_report.assert_called_once_with("lgtma", "app.kubernetes.io/name=grafana,app.kubernetes.io/instance=grafana")

    def test_service_pod_report_without_a_selector(self, k8s_client, mock_ssh):
        mock_ssh.run_kubectl.return_value = self._ok(json.dumps({"spec": {}}))

        report = k8s_client.service_pod_report("default", "external")

        assert report == "svc/external in default has no selector, so there are no pods to report on"

    def test_service_pod_report_when_the_service_cannot_be_read(self, k8s_client, mock_ssh):
        mock_ssh.run_kubectl.return_value = CommandResult(returncode=1, stdout="", stderr="not found")

        report = k8s_client.service_pod_report("lgtma", "grafana")

        assert report == "could not read svc/grafana in lgtma: not found"

    # -------------------------------------------------------------------------
    # unstable_pods
    # -------------------------------------------------------------------------

    @staticmethod
    def _ago(*, days=0.0, hours=0.0, minutes=0.0):
        """An RFC3339 timestamp that far in the past, in the shape kubectl emits."""
        moment = datetime.now(UTC) - timedelta(days=days, hours=hours, minutes=minutes)
        return moment.isoformat().replace("+00:00", "Z")

    @staticmethod
    def _pod_json(*, name, started_at, restarts, finished_at=None, reason="Error"):
        status = {"name": "loki", "restartCount": restarts}
        if finished_at is not None:
            status["lastState"] = {"terminated": {"finishedAt": finished_at, "reason": reason}}
        return {
            "metadata": {"name": name},
            "status": {"startTime": started_at, "containerStatuses": [status]},
        }

    def _pods_result(self, *pods):
        return CommandResult(returncode=0, stdout=json.dumps({"items": list(pods)}), stderr="")

    def test_unstable_pods_empty_when_nothing_restarted(self, k8s_client, mock_ssh):
        mock_ssh.run_kubectl.return_value = self._pods_result(
            self._pod_json(name="loki-write-0", started_at=self._ago(days=30), restarts=0)
        )

        unstable, result = k8s_client.unstable_pods("lgtma")

        assert result.success is True
        assert unstable == []

    def test_unstable_pods_tolerates_restarts_inside_the_grace_window(self, k8s_client, mock_ssh):
        mock_ssh.run_kubectl.return_value = self._pods_result(
            self._pod_json(
                name="mimir-ingester-zone-a-0",
                started_at=self._ago(hours=2),
                restarts=3,
                finished_at=self._ago(hours=2, minutes=-1),
            )
        )

        unstable, _ = k8s_client.unstable_pods("lgtma")

        assert unstable == []

    def test_unstable_pods_flags_a_restart_after_the_grace_window(self, k8s_client, mock_ssh):
        mock_ssh.run_kubectl.return_value = self._pods_result(
            self._pod_json(
                name="loki-write-0",
                started_at=self._ago(hours=6),
                restarts=1,
                finished_at=self._ago(hours=1),
                reason="OOMKilled",
            )
        )

        unstable, _ = k8s_client.unstable_pods("lgtma")

        assert len(unstable) == 1
        assert "loki-write-0/loki" in unstable[0]
        assert "OOMKilled" in unstable[0]

    def test_unstable_pods_ignores_a_restart_that_has_since_settled(self, k8s_client, mock_ssh):
        """
        The production case: a node reboot restarted every pod weeks ago and nothing since.
        """
        mock_ssh.run_kubectl.return_value = self._pods_result(
            self._pod_json(
                name="loki-write-0",
                started_at=self._ago(days=83),
                restarts=3,
                finished_at=self._ago(days=53),
                reason="Unknown",
            )
        )

        unstable, _ = k8s_client.unstable_pods("lgtma")

        assert unstable == []

    def test_unstable_pods_still_flags_a_settled_container_over_the_ceiling(self, k8s_client, mock_ssh):
        """A count high enough to mean a slow crash loop counts whenever it accumulated."""
        mock_ssh.run_kubectl.return_value = self._pods_result(
            self._pod_json(
                name="loki-write-0",
                started_at=self._ago(days=83),
                restarts=11,
                finished_at=self._ago(days=53),
            )
        )

        unstable, _ = k8s_client.unstable_pods("lgtma")

        assert len(unstable) == 1
        assert "over the 10 restart ceiling" in unstable[0]
        assert "ago" not in unstable[0]

    def test_unstable_pods_reports_the_age_of_the_restart_not_its_offset(self, k8s_client, mock_ssh):
        """The message says how long ago, which is the number a reader needs to act."""
        mock_ssh.run_kubectl.return_value = self._pods_result(
            self._pod_json(
                name="loki-write-0",
                started_at=self._ago(days=30),
                restarts=1,
                finished_at=self._ago(minutes=20),
                reason="OOMKilled",
            )
        )

        unstable, _ = k8s_client.unstable_pods("lgtma")

        assert len(unstable) == 1
        assert "20m ago" in unstable[0]

    def test_unstable_pods_flags_a_count_over_the_ceiling(self, k8s_client, mock_ssh):
        mock_ssh.run_kubectl.return_value = self._pods_result(
            self._pod_json(
                name="loki-write-0",
                started_at=self._ago(hours=2),
                restarts=11,
                finished_at=self._ago(hours=2, minutes=-1),
            )
        )

        unstable, _ = k8s_client.unstable_pods("lgtma")

        assert len(unstable) == 1
        assert "over the 10 restart ceiling" in unstable[0]

    def test_unstable_pods_honours_the_name_prefix(self, k8s_client, mock_ssh):
        mock_ssh.run_kubectl.return_value = self._pods_result(
            self._pod_json(
                name="tempo-ingester-0",
                started_at=self._ago(hours=6),
                restarts=1,
                finished_at=self._ago(hours=1),
            )
        )

        assert k8s_client.unstable_pods("lgtma", name_prefix="loki")[0] == []
        assert k8s_client.unstable_pods("lgtma", name_prefix="tempo")[0] != []

    def test_unstable_pods_reports_kubectl_failure(self, k8s_client, mock_ssh):
        mock_ssh.run_kubectl.return_value = CommandResult(returncode=1, stdout="", stderr="boom")

        unstable, result = k8s_client.unstable_pods("lgtma")

        assert unstable == []
        assert result.success is False

    # -------------------------------------------------------------------------
    # restart_pods
    # -------------------------------------------------------------------------

    @staticmethod
    def _kubectl_router(list_results, delete_result):
        """Answer pod listings from a queue and the delete with a fixed result."""
        listings = iter(list_results)

        def route(args, **kwargs):
            if args.startswith("delete pods"):
                return delete_result
            return next(listings)

        return route

    def test_restart_pods_waits_for_new_uids_to_be_ready(self, k8s_client, mock_ssh):
        before = CommandResult(returncode=0, stdout="loki-write-0=uid-old=true;", stderr="")
        replaced = CommandResult(returncode=0, stdout="loki-write-0=uid-new=true;", stderr="")
        mock_ssh.run_kubectl.side_effect = self._kubectl_router(
            [before, replaced, replaced], CommandResult(returncode=0, stdout="deleted", stderr="")
        )

        pods, result = k8s_client.restart_pods("lgtma", "app=loki", timeout=1.0, interval=0.01)

        assert result.success is True
        assert pods == {"loki-write-0": ("uid-new", True)}

    def test_restart_pods_sends_sigkill_when_not_graceful(self, k8s_client, mock_ssh):
        before = CommandResult(returncode=0, stdout="loki-write-0=uid-old=true;", stderr="")
        replaced = CommandResult(returncode=0, stdout="loki-write-0=uid-new=true;", stderr="")
        mock_ssh.run_kubectl.side_effect = self._kubectl_router(
            [before, replaced, replaced], CommandResult(returncode=0, stdout="deleted", stderr="")
        )

        k8s_client.restart_pods("lgtma", "app=loki", graceful=False, timeout=1.0, interval=0.01)

        delete_args = [call[0][0] for call in mock_ssh.run_kubectl.call_args_list if call[0][0].startswith("delete")]
        assert delete_args == ["delete pods -n lgtma -l app=loki --wait=false --grace-period=0 --force"]

    def test_restart_pods_reports_when_the_selector_matches_nothing(self, k8s_client, mock_ssh):
        mock_ssh.run_kubectl.return_value = CommandResult(returncode=0, stdout="", stderr="")

        pods, result = k8s_client.restart_pods("lgtma", "app=loki")

        assert pods == {}
        assert result.success is False
        assert "No pods matching app=loki" in result.stderr

    def test_restart_pods_reports_a_failed_delete(self, k8s_client, mock_ssh):
        before = CommandResult(returncode=0, stdout="loki-write-0=uid-old=true;", stderr="")
        mock_ssh.run_kubectl.side_effect = self._kubectl_router(
            [before], CommandResult(returncode=1, stdout="", stderr="forbidden")
        )

        pods, result = k8s_client.restart_pods("lgtma", "app=loki")

        assert pods == {}
        assert result.stderr == "forbidden"

    def test_restart_pods_reports_replacements_that_never_became_ready(self, k8s_client, mock_ssh):
        before = CommandResult(returncode=0, stdout="loki-write-0=uid-old=true;", stderr="")
        not_ready = CommandResult(returncode=0, stdout="loki-write-0=uid-new=false;", stderr="")
        mock_ssh.run_kubectl.side_effect = self._kubectl_router(
            [before] + [not_ready] * 20, CommandResult(returncode=0, stdout="deleted", stderr="")
        )

        pods, result = k8s_client.restart_pods("lgtma", "app=loki", timeout=0.05, interval=0.01)

        assert result.success is False
        assert "were not ready" in result.stderr
        assert pods == {"loki-write-0": ("uid-new", False)}


class TestKubernetesClientLocalhost:
    """KubernetesClient uses local kubectl when host is loopback."""

    @pytest.fixture
    def mock_ssh(self):
        ssh = MagicMock()
        ssh.run_kubectl = MagicMock()
        return ssh

    @pytest.fixture
    def localhost_config(self):
        return LGTMConfig(host="localhost", remote_user="u")

    def test_localhost_uses_subprocess_not_ssh(self, localhost_config, mock_ssh):
        client = KubernetesClient(localhost_config, ssh=mock_ssh)
        with patch("production_test_framework.helper.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="node1   Ready   control-plane   1.28   192.168.1.1\n",
                stderr="",
            )
            nodes, result = client.get_nodes()
            assert result.success is True
            assert len(nodes) == 1
            mock_ssh.run_kubectl.assert_not_called()
            mock_run.assert_called_once()
            cmd = mock_run.call_args[0][0]
            assert cmd[0] == "kubectl"
            assert cmd[1] == "--kubeconfig"
            assert "get nodes -o wide --no-headers" in " ".join(cmd)

    def test_127_0_0_1_uses_subprocess(self, mock_ssh):
        cfg = LGTMConfig(host="127.0.0.1", remote_user="u")
        client = KubernetesClient(cfg, ssh=mock_ssh)
        with patch("production_test_framework.helper.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            client.get_namespaces()
            mock_ssh.run_kubectl.assert_not_called()
            mock_run.assert_called_once()


class TestKubectlPortForwarder:
    """Tests for the SSH-based KubectlPortForwarder with a mocked SSHExecutor."""

    @staticmethod
    def _ssh_mock(listening_port: int | None) -> MagicMock:
        """SSHExecutor stub whose `ss -ltn` reports listening_port bound, if any."""
        ssh = MagicMock()

        def run(command: str, *args, **kwargs) -> CommandResult:
            if command.startswith("ss -ltn"):
                stdout = f"LISTEN 0 4096 127.0.0.1:{listening_port} 0.0.0.0:*" if listening_port else ""
                return CommandResult(returncode=0, stdout=stdout, stderr="")
            if command.startswith("tail"):
                return CommandResult(returncode=0, stdout="error: services 'mimir-gateway' not found", stderr="")
            return CommandResult(returncode=0, stdout="12345", stderr="")

        ssh.run.side_effect = run
        return ssh

    def test_start_service_tunnel_waits_for_remote_bind(self):
        ssh = self._ssh_mock(listening_port=9009)
        forwarder = KubectlPortForwarder(ssh)

        with patch.object(forwarder, "_start_ssh_tunnel", return_value=True) as mock_tunnel:
            assert forwarder.start_service_tunnel(
                local_port=9009,
                service_name="mimir-gateway",
                service_port=80,
                namespace="lgtma",
                use_sudo=False,
            )

        commands = [c.args[0] for c in ssh.run.call_args_list]
        kubectl = next(c for c in commands if "port-forward" in c)
        assert "kubectl -n lgtma port-forward svc/mimir-gateway 9009:80" in kubectl
        assert not kubectl.startswith("sudo"), "use_sudo=False must not prepend sudo"
        assert any(c.startswith("ss -ltn") for c in commands), "must verify the remote port is bound"
        mock_tunnel.assert_called_once()
        assert 12345 in forwarder._kubectl_pids

    @patch("production_test_framework.helper.time.sleep")
    def test_start_service_tunnel_fails_when_remote_never_binds(self, mock_sleep):
        # kubectl exits immediately (wrong namespace, missing service): nothing
        # ever listens, so the tunnel must be reported as failed rather than
        # handed back broken.
        ssh = self._ssh_mock(listening_port=None)
        forwarder = KubectlPortForwarder(ssh)

        with patch.object(forwarder, "_start_ssh_tunnel", return_value=True) as mock_tunnel:
            assert not forwarder.start_service_tunnel(
                local_port=9009,
                service_name="mimir-gateway",
                service_port=80,
                namespace="mosaic",
                ready_timeout=1.0,
            )

        mock_tunnel.assert_not_called()

    def test_start_service_tunnel_proceeds_when_ss_is_unavailable(self):
        # A host without iproute2 cannot be probed; that must not fail every
        # tunnel, so an unusable probe falls back to the previous behavior.
        ssh = MagicMock()
        ssh.run.side_effect = lambda command, *a, **kw: (
            CommandResult(returncode=127, stdout="", stderr="ss: command not found")
            if command.startswith("ss -ltn")
            else CommandResult(returncode=0, stdout="12345", stderr="")
        )
        forwarder = KubectlPortForwarder(ssh)

        with patch.object(forwarder, "_start_ssh_tunnel", return_value=True) as mock_tunnel:
            assert forwarder.start_service_tunnel(
                local_port=9009,
                service_name="mimir-gateway",
                service_port=80,
                ready_timeout=1.0,
            )

        mock_tunnel.assert_called_once()

    def test_start_service_tunnel_can_skip_the_readiness_wait(self):
        ssh = self._ssh_mock(listening_port=None)
        forwarder = KubectlPortForwarder(ssh)

        with patch.object(forwarder, "_start_ssh_tunnel", return_value=True):
            assert forwarder.start_service_tunnel(
                local_port=9009,
                service_name="mimir-gateway",
                service_port=80,
                wait_ready=False,
            )

        assert not any(c.args[0].startswith("ss -ltn") for c in ssh.run.call_args_list)


class TestLocalKubectlPortForwarder:
    """Tests for LocalKubectlPortForwarder with mocked subprocess."""

    @patch("production_test_framework.k8s.subprocess.Popen")
    def test_start_service_tunnel_builds_correct_command(self, mock_popen):
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        with patch("production_test_framework.k8s.socket.socket") as mock_socket_cls:
            mock_sock = MagicMock()
            mock_socket_cls.return_value = mock_sock
            mock_sock.connect_ex.return_value = 0

            forwarder = LocalKubectlPortForwarder(namespace="default")
            result = forwarder.start_service_tunnel(
                local_port=3000,
                service_name="grafana",
                service_port=80,
                wait_ready=True,
                ready_timeout=1.0,
            )

            assert result is True
            call_args = mock_popen.call_args[0][0]
            assert "kubectl" in call_args
            assert "-n" in call_args
            assert "default" in call_args
            assert "port-forward" in call_args
            assert "svc/grafana" in call_args
            assert "3000:80" in call_args

    def test_stop_service_tunnel_removes_forward(self):
        forwarder = LocalKubectlPortForwarder()
        mock_proc = MagicMock()
        forwarder._forwards.append(
            LocalPortForward(process=mock_proc, local_port=3000, service="grafana", namespace="default")
        )
        result = forwarder.stop_service_tunnel("grafana")
        assert result is True
        mock_proc.terminate.assert_called_once()
        assert len(forwarder._forwards) == 0

    def test_stop_service_tunnel_not_found(self):
        forwarder = LocalKubectlPortForwarder()
        assert forwarder.stop_service_tunnel("nonexistent") is False

    def test_is_running(self):
        forwarder = LocalKubectlPortForwarder()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        forwarder._forwards.append(
            LocalPortForward(process=mock_proc, local_port=3000, service="grafana", namespace="default")
        )
        assert forwarder.is_running("grafana") is True
        mock_proc.poll.return_value = 1
        assert forwarder.is_running("grafana") is False

    def test_get_local_port(self):
        forwarder = LocalKubectlPortForwarder()
        forwarder._forwards.append(
            LocalPortForward(
                process=MagicMock(),
                local_port=3000,
                service="grafana",
                namespace="default",
            )
        )
        assert forwarder.get_local_port("grafana") == 3000
        assert forwarder.get_local_port("other") is None
