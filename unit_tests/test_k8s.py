# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2025 Delos Data, Inc.

"""Unit tests for k8s module."""

import json
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

    # -------------------------------------------------------------------------
    # unstable_pods
    # -------------------------------------------------------------------------

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
            self._pod_json(name="loki-write-0", started_at="2026-08-18T10:00:00Z", restarts=0)
        )

        unstable, result = k8s_client.unstable_pods("lgtma")

        assert result.success is True
        assert unstable == []

    def test_unstable_pods_tolerates_restarts_inside_the_grace_window(self, k8s_client, mock_ssh):
        mock_ssh.run_kubectl.return_value = self._pods_result(
            self._pod_json(
                name="mimir-ingester-zone-a-0",
                started_at="2026-08-18T10:00:00Z",
                restarts=3,
                finished_at="2026-08-18T10:01:00Z",
            )
        )

        unstable, _ = k8s_client.unstable_pods("lgtma")

        assert unstable == []

    def test_unstable_pods_flags_a_restart_after_the_grace_window(self, k8s_client, mock_ssh):
        mock_ssh.run_kubectl.return_value = self._pods_result(
            self._pod_json(
                name="loki-write-0",
                started_at="2026-08-18T10:00:00Z",
                restarts=1,
                finished_at="2026-08-18T11:00:00Z",
                reason="OOMKilled",
            )
        )

        unstable, _ = k8s_client.unstable_pods("lgtma")

        assert len(unstable) == 1
        assert "loki-write-0/loki" in unstable[0]
        assert "OOMKilled" in unstable[0]

    def test_unstable_pods_flags_a_count_over_the_ceiling(self, k8s_client, mock_ssh):
        mock_ssh.run_kubectl.return_value = self._pods_result(
            self._pod_json(
                name="loki-write-0",
                started_at="2026-08-18T10:00:00Z",
                restarts=11,
                finished_at="2026-08-18T10:01:00Z",
            )
        )

        unstable, _ = k8s_client.unstable_pods("lgtma")

        assert len(unstable) == 1
        assert "over the 10 ceiling" in unstable[0]

    def test_unstable_pods_honours_the_name_prefix(self, k8s_client, mock_ssh):
        mock_ssh.run_kubectl.return_value = self._pods_result(
            self._pod_json(
                name="tempo-ingester-0",
                started_at="2026-08-18T10:00:00Z",
                restarts=1,
                finished_at="2026-08-18T11:00:00Z",
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
        return LGTMConfig(host="localhost", ansible_remote_user="u")

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
        cfg = LGTMConfig(host="127.0.0.1", ansible_remote_user="u")
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
