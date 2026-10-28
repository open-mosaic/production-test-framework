# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2026 Delos Data, Inc.

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from production_test_framework.switch.arista.arista_eos_switch import AristaEosSwitch
from production_test_framework.switch.mac import normalize_mac
from production_test_framework.switch.models import NetworkSwitchConfig
from production_test_framework.switch.portname import PortNameResolver

FIXTURES = Path(__file__).parent / "fixtures" / "switch" / "arista"


def _switch() -> AristaEosSwitch:
    switch = AristaEosSwitch(NetworkSwitchConfig(host="h", username="u", password="p", verify_tls=False))
    switch._node = MagicMock()
    return switch


def _port_names(interface_names: list[str]) -> PortNameResolver:
    """The port ID mapping the driver learns from "show interfaces status"."""
    port_names = PortNameResolver()
    port_names.learn(interface_names)
    return port_names


def _mac_neighbor(mac: str) -> dict:
    """One lldpNeighborInfo entry advertising a MAC port id."""
    return {"lldpNeighborInfo": [{"neighborInterfaceInfo": {"interfaceIdType": "macAddress", "interfaceId": mac}}]}


@pytest.fixture
def lldp_payload() -> dict:
    return json.loads((FIXTURES / "lldp_neighbors_detail.json").read_text())


def test_parse_lldp_neighbors(lldp_payload: dict) -> None:
    port_names = _port_names([f"Ethernet{cage}" for cage in range(1, 6)])
    neighbors = _switch()._parse_lldp_neighbors(lldp_payload["lldpNeighbors"], port_names)

    # Ethernet1/3 have MAC port ids; Ethernet5 (interface-name port id) and
    # Management1 (not a front-panel port) are dropped. Sorted by switch_port.
    assert [(n.switch_port, n.interface, n.chassis_mac) for n in neighbors] == [
        (1, "Ethernet1", "00:50:56:00:00:11"),
        (3, "Ethernet3", "aa:bb:cc:00:00:22"),
    ]


def test_lldp_neighbors_only_management_neighbor_is_empty() -> None:
    # Mirrors a real switch where the sole neighbor is learned on Management1 and
    # advertises an interface-name port id: not front-panel and not a MAC -> empty.
    payload = {
        "lldpNeighbors": {
            "Ethernet1": {"lldpNeighborInfo": []},
            "Ethernet49/1": {"lldpNeighborInfo": []},
            "Management1": {
                "lldpNeighborInfo": [
                    {
                        "chassisIdType": "macAddress",
                        "chassisId": "001c.7361.af79",
                        "neighborInterfaceInfo": {"interfaceIdType": "interfaceName", "interfaceId": '"Ethernet19"'},
                    }
                ]
            },
        }
    }
    port_names = _port_names(["Ethernet1", "Ethernet49/1"])
    assert _switch()._parse_lldp_neighbors(payload["lldpNeighbors"], port_names) == []


def test_lldp_neighbors_calls_show_lldp_detail(lldp_payload: dict) -> None:
    switch = _switch()
    # Port IDs come from the status table, so the property runs that command too.
    statuses = {"interfaceStatuses": {f"Ethernet{cage}": {} for cage in range(1, 6)}}
    responses = {"show lldp neighbors detail": lldp_payload, "show interfaces status": statuses}
    switch._node.run_commands.side_effect = lambda commands: [responses[commands[0]]]

    neighbors = switch.lldp_neighbors

    assert len(neighbors) == 2
    assert [call.args[0] for call in switch._node.run_commands.call_args_list] == [
        ["show lldp neighbors detail"],
        ["show interfaces status"],
    ]


def test_parse_lldp_neighbors_breakout_shifts_port_ids() -> None:
    """A neighbour's port ID is its interface's position in the switch's interface
    list, so each subport of a broken-out QSFP gets its own ID rather than the cage
    number they share.
    """
    payload = {
        "Ethernet48": _mac_neighbor("0050.5600.0011"),
        "Ethernet49/2": _mac_neighbor("0050.5600.0022"),
    }
    port_names = _port_names(["Ethernet48", "Ethernet49/1", "Ethernet49/2"])

    neighbors = _switch()._parse_lldp_neighbors(payload, port_names)

    assert [(n.switch_port, n.interface) for n in neighbors] == [
        (1, "Ethernet48"),
        (3, "Ethernet49/2"),
    ]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("001c.7300.abcd", "00:1c:73:00:ab:cd"),
        ("00:1C:73:00:AB:CD", "00:1c:73:00:ab:cd"),
        ("001c7300abcd", "00:1c:73:00:ab:cd"),
        ("00-1c-73-00-ab-cd", "00:1c:73:00:ab:cd"),
        ("not-a-mac", ""),
        ("zz1c7300abcd", ""),
        ("", ""),
    ],
)
def test_normalize_mac(raw: str, expected: str) -> None:
    assert normalize_mac(raw) == expected
