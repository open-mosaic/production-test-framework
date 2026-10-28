# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2026 Delos Data, Inc.

"""Cisco NX-OS driver, against NX-API replies recorded from a Nexus N9364E (NX-OS 10.5).

The LLDP fixture adds one EPI-style neighbor (a MAC port id on a data port) to the
recorded one, since the bench Nexus has no data-port links.
"""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import requests

from production_test_framework.switch.cisco.nxos_switch import CiscoNxosSwitch, _full_name, _rows
from production_test_framework.switch.exceptions import SwitchAPIError
from production_test_framework.switch.models import (
    LldpNeighbor,
    MacEntry,
    NetworkSwitchConfig,
    Port,
    PortBridge,
)
from production_test_framework.switch.switch_status import create_switch

FIXTURES = Path(__file__).parent / "fixtures" / "switch" / "cisco"


def _fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / name).read_text())


def _switch(replies: dict[str, Any] | None = None) -> CiscoNxosSwitch:
    """A driver whose show commands return replies[command] (a fixture name or a body)."""
    switch = CiscoNxosSwitch(NetworkSwitchConfig(host="h", username="u", password="p", verify_tls=False))
    switch._session = MagicMock()

    def run_show(command: str) -> dict[str, Any]:
        reply = (replies or {}).get(command)
        if reply is None:
            raise AssertionError(f"unexpected command {command!r}")
        return _fixture(reply) if isinstance(reply, str) else reply

    switch._run_show = run_show  # type: ignore[method-assign]
    return switch


def _response(status: int, body: Any) -> MagicMock:
    response = MagicMock()
    response.status_code = status
    response.json.return_value = body
    response.text = json.dumps(body)
    return response


# --- plumbing -----------------------------------------------------------------


def test_create_switch_builds_cisco_driver() -> None:
    switch = create_switch("cisco-nxos", "h", "u", "p")
    assert isinstance(switch, CiscoNxosSwitch)
    assert switch._url == "https://h:443/ins"


def test_run_show_posts_json_rpc_and_returns_the_body() -> None:
    switch = CiscoNxosSwitch(NetworkSwitchConfig(host="h", username="u", password="p", verify_tls=False))
    switch._session = MagicMock()
    switch._session.post.return_value = _response(200, {"jsonrpc": "2.0", "result": {"body": {"hostname": "nx"}}})

    assert switch._run_show("show hostname") == {"hostname": "nx"}
    _, kwargs = switch._session.post.call_args
    assert kwargs["json"]["method"] == "cli"
    assert kwargs["json"]["params"]["cmd"] == "show hostname"
    assert kwargs["verify"] is False


def test_run_show_empty_result_is_an_empty_body() -> None:
    """NX-API answers a show with nothing to show (e.g. an unknown VLAN) with a null result."""
    switch = CiscoNxosSwitch(NetworkSwitchConfig(host="h", username="u", password="p"))
    switch._session = MagicMock()
    switch._session.post.return_value = _response(200, {"jsonrpc": "2.0", "result": None, "id": 1})
    assert switch._run_show("show vlan id 3000") == {}


def test_command_error_raises_with_the_switch_message() -> None:
    switch = CiscoNxosSwitch(NetworkSwitchConfig(host="h", username="u", password="p"))
    switch._session = MagicMock()
    error = {"code": -32602, "message": "Invalid params", "data": {"msg": "% Invalid command\n"}}
    switch._session.post.return_value = _response(400, {"jsonrpc": "2.0", "error": error})
    with pytest.raises(SwitchAPIError, match="% Invalid command"):
        switch._run_show("show lldp neighbors detail")


@pytest.mark.parametrize("data", [None, "% Invalid command", {"msg": None}])
def test_command_error_with_odd_data_still_raises_switch_api_error(data: object) -> None:
    switch = CiscoNxosSwitch(NetworkSwitchConfig(host="h", username="u", password="p"))
    switch._session = MagicMock()
    error = {"code": -32602, "message": "Invalid params", "data": data}
    switch._session.post.return_value = _response(400, {"jsonrpc": "2.0", "error": error})
    with pytest.raises(SwitchAPIError, match="Invalid params"):
        switch._run_show("show version")


def test_unreachable_switch_raises() -> None:
    switch = CiscoNxosSwitch(NetworkSwitchConfig(host="h", username="u", password="p"))
    switch._session = MagicMock()
    switch._session.post.side_effect = requests.ConnectionError("refused")
    with pytest.raises(SwitchAPIError, match="refused"):
        switch._run_show("show version")


def test_config_runs_in_configure_mode() -> None:
    switch = CiscoNxosSwitch(NetworkSwitchConfig(host="h", username="u", password="p"))
    switch._session = MagicMock()
    switch._session.post.return_value = _response(200, {"jsonrpc": "2.0", "result": None, "id": 1})

    switch.set_port_admin_state("Ethernet1/2", False)
    switch.set_port_admin_state("Ethernet1/2", True)
    switch.delete_vlan("3900")

    commands = [call.kwargs["json"]["params"]["cmd"] for call in switch._session.post.call_args_list]
    assert commands == [
        "configure terminal ; interface Ethernet1/2 ; shutdown",
        "configure terminal ; interface Ethernet1/2 ; no shutdown",
        "configure terminal ; no vlan 3900",
    ]


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ({"T": {"R": {"a": 1}}}, [{"a": 1}]),  # one row comes as an object
        ({"T": {"R": [{"a": 1}, {"a": 2}]}}, [{"a": 1}, {"a": 2}]),
        ({}, []),  # an empty table is left out
    ],
)
def test_rows(body: dict, expected: list) -> None:
    assert _rows(body, "T", "R") == expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [("Eth1/1", "Ethernet1/1"), ("Ethernet1/1", "Ethernet1/1"), ("Eth1/49/2", "Ethernet1/49/2"), ("mgmt0", "mgmt0")],
)
def test_full_name(name: str, expected: str) -> None:
    assert _full_name(name) == expected


# --- reads --------------------------------------------------------------------


def test_status() -> None:
    switch = _switch({"show version": "show_version.json", "show hostname": {"hostname": "switch"}})
    status = switch.status
    assert status.hostname == "switch"
    assert status.model == "cisco Nexus N9364E-SG2-O Chassis"
    assert status.serial_number == "FLM29353EL9"
    assert status.software_version == "10.5(3)"
    assert status.firmware_version == "01.08"
    assert int(status.uptime) > 0


def test_ports() -> None:
    ports = _switch({"show interface": "show_interface.json"}).ports
    by_id = {port.id: port for port in ports}
    assert by_id["mgmt0"] == Port(id="mgmt0", admin_up=True, oper_up=True)
    assert by_id["Ethernet1/1"] == Port(id="Ethernet1/1", admin_up=True, oper_up=False)
    assert by_id["Ethernet1/2"] == Port(id="Ethernet1/2", admin_up=False, oper_up=False)
    # Ethernet1/2 sorts before Ethernet1/10, numerically.
    ethernet = [port.id for port in ports if port.id.startswith("Ethernet")]
    assert ethernet == ["Ethernet1/1", "Ethernet1/2", "Ethernet1/10"]


def test_port() -> None:
    body = {"TABLE_interface": {"ROW_interface": {"interface": "Ethernet1/2", "state": "down", "admin_state": "down"}}}
    assert _switch({"show interface Ethernet1/2": body}).port("Ethernet1/2") == Port(
        id="Ethernet1/2", admin_up=False, oper_up=False
    )


def test_port_description() -> None:
    body = {"TABLE_interface": {"ROW_interface": {"interface": "Ethernet1/2", "desc": "to bootes"}}}
    assert _switch({"show interface Ethernet1/2": body}).port("Ethernet1/2").description == "to bootes"


def test_port_not_found() -> None:
    with pytest.raises(SwitchAPIError, match="not found"):
        _switch({"show interface Ethernet1/99": {}}).port("Ethernet1/99")


def test_vlans() -> None:
    switch = _switch({"show vlan brief": "show_vlan_brief.json", "show interface": "show_interface.json"})
    vlans = {vlan.id: vlan for vlan in switch.vlans}
    assert sorted(vlans, key=int) == ["1", "3900", "3901"]
    assert vlans["1"].ports == ()
    assert [port.id for port in vlans["3901"].ports] == ["Ethernet1/1", "Ethernet1/2"]
    # Member ports carry their state from show interface.
    assert vlans["3901"].ports[1] == Port(id="Ethernet1/2", admin_up=False, oper_up=False)


def test_vlan() -> None:
    switch = _switch({"show vlan id 3900": "show_vlan_id_3900.json", "show interface": "show_interface.json"})
    vlan = switch.vlan("3900")
    assert vlan.id == "3900"
    assert [port.id for port in vlan.ports] == ["Ethernet1/1"]


def test_vlan_not_found() -> None:
    with pytest.raises(SwitchAPIError, match="VLAN 3000 not found"):
        _switch({"show vlan id 3000": {}}).vlan("3000")


def test_vlan_command_error_is_not_found() -> None:
    switch = _switch()
    switch._run_show = MagicMock(side_effect=SwitchAPIError("Invalid range", 400))  # type: ignore[method-assign]
    with pytest.raises(SwitchAPIError, match="VLAN 4095 not found"):
        switch.vlan("4095")


def test_vlan_unreachable_switch_is_not_reported_as_not_found() -> None:
    switch = _switch()
    switch._run_show = MagicMock(  # type: ignore[method-assign]
        side_effect=SwitchAPIError("NX-API request failed: show vlan id 10: refused")
    )
    with pytest.raises(SwitchAPIError, match="refused"):
        switch.vlan("10")


def test_lldp_neighbors() -> None:
    switch = _switch(
        {"show lldp neighbors detail": "show_lldp_neighbors_detail.json", "show interface": "show_interface.json"}
    )
    # The mgmt0 neighbor is not on a data port, and advertises an interface name, not a MAC.
    assert switch.lldp_neighbors == [
        LldpNeighbor(interface="Ethernet1/2", switch_port=2, chassis_mac="02:00:00:00:00:05"),
    ]


def test_mac_table() -> None:
    table = _switch({"show mac address-table": "show_mac_address_table.json"}).mac_table
    # The supervisor's own entry (sup-eth1) is not on a data port.
    assert table == [MacEntry(mac="02:00:00:00:39:00", port="Ethernet1/1", vlan=3900, static=True)]


def test_mac_table_by_interface() -> None:
    switch = _switch({"show mac address-table": "show_mac_address_table.json"})
    assert switch.mac_table_by_interface == {"Ethernet1/1": {"02:00:00:00:39:00"}}


def test_mac_table_keeps_port_channels_and_drops_supervisor_entries() -> None:
    rows = [
        {"disp_mac_addr": "0200.0000.0001", "disp_type": "dynamic", "disp_vlan": "10", "disp_port": "Po1"},
        {"disp_mac_addr": "0200.0000.0002", "disp_type": "static", "disp_vlan": "-", "disp_port": "sup-eth1(R)"},
        {"disp_mac_addr": "0200.0000.0003", "disp_type": "static", "disp_vlan": "1", "disp_port": "Vlan1(R)"},
    ]
    body = {"TABLE_mac_address": {"ROW_mac_address": rows}}
    table = _switch({"show mac address-table": body}).mac_table
    assert [(entry.port, entry.mac) for entry in table] == [("Po1", "02:00:00:00:00:01")]


def test_mac_table_empty() -> None:
    assert _switch({"show mac address-table": {}}).mac_table == []


@pytest.mark.parametrize(
    ("port", "fixture", "expected"),
    [
        (
            "Ethernet1/1",
            "show_interface_switchport_trunk.json",
            PortBridge(interface="Ethernet1/1", mode="trunk", native_vlan=3900, learning=False),
        ),
        (
            "Ethernet1/2",
            "show_interface_switchport_access.json",
            PortBridge(interface="Ethernet1/2", mode="access", native_vlan=3901, learning=True),
        ),
        ("Ethernet1/1", "show_interface_switchport_routed.json", PortBridge(interface="Ethernet1/1")),
    ],
)
def test_port_bridge(port: str, fixture: str, expected: PortBridge) -> None:
    switch = _switch({f"show interface {port} switchport": fixture})
    assert switch.port_bridge(port) == expected


def test_port_bridge_not_found() -> None:
    with pytest.raises(SwitchAPIError, match="not found"):
        _switch({"show interface Ethernet1/99 switchport": {}}).port_bridge("Ethernet1/99")


def test_flow_control_not_supported() -> None:
    with pytest.raises(NotImplementedError):
        _ = _switch().flow_control
