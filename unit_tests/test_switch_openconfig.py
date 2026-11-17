# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2026 Delos Data, Inc.

from ipaddress import ip_interface
from typing import Any
from unittest.mock import MagicMock, patch

import grpc
import pytest

from production_test_framework.switch.exceptions import SwitchAPIError, SwitchNotSupportedError
from production_test_framework.switch.models import FecEncoding, LldpNeighbor, MacEntry, NetworkSwitchConfig, Port
from production_test_framework.switch.openconfig import openconfig_switch
from production_test_framework.switch.openconfig.openconfig_switch import OpenConfigSwitch, _without

NI = "network-instances/network-instance[name=default]"


class _Rpc(grpc.RpcError):
    def __init__(self, code: grpc.StatusCode, details: str) -> None:
        super().__init__(details)
        self._code, self._details = code, details

    def code(self) -> grpc.StatusCode:
        return self._code

    def details(self) -> str:
        return self._details


class _Client:
    """Answers Get from canned responses, keyed by the requested paths, and records each Set."""

    def __init__(self, responses: dict[tuple[str, ...], Any]) -> None:
        self.responses = responses
        self.sets: list[dict[str, Any]] = []

    def get(self, path: list[str], encoding: str | None = None) -> dict[str, Any]:
        response = self.responses.get(tuple(path), {})
        if isinstance(response, Exception):
            raise response
        return response

    def set(self, delete=None, replace=None, update=None, encoding=None) -> None:
        self.sets.append({"delete": delete, "replace": replace, "update": update})


def _switch(responses: dict[tuple[str, ...], Any] | None = None, encoding: str = "json_ietf"):
    switch = OpenConfigSwitch(NetworkSwitchConfig(host="h", username="u", password="p", port=9339))
    switch._encoding = encoding
    client = _Client(responses or {})

    def call(what, fn):
        try:
            return fn(client)
        except grpc.RpcError as exc:
            raise openconfig_switch._switch_error(what, exc) from exc

    switch._call = call  # type: ignore[method-assign]
    return switch, client


def _response(*updates: tuple[str | None, Any], prefix: str | None = None) -> dict[str, Any]:
    return {"notification": [{"prefix": prefix, "update": [{"path": p, "val": v} for p, v in updates]}]}


def _interface(name: str, kind: str = "ethernetCsmacd", admin: str = "UP", oper: str = "UP", **state) -> dict:
    return {
        "name": name,
        "state": {"type": f"iana-if-type:{kind}", "admin-status": admin, "oper-status": oper, **state},
    }


# --- reading ports, whichever way the switch splits its answer ---------------


def test_ports_from_one_json_tree() -> None:
    tree = {
        "openconfig-interfaces:interface": [
            _interface("Ethernet1/10", oper="DOWN"),
            _interface("Ethernet1/2", description="uplink"),
            _interface("mgmt0"),
            _interface("Vlan5", kind="l3ipvlan"),
        ]
    }
    switch, _ = _switch({("interfaces/interface/state",): _response(("interfaces", tree))})
    assert switch.ports == [
        Port(id="Ethernet1/2", admin_up=True, oper_up=True, description="uplink"),
        Port(id="Ethernet1/10", admin_up=True, oper_up=False),
        Port(id="mgmt0", admin_up=True, oper_up=True),
    ]


def test_ports_from_an_update_per_entry_and_per_leaf() -> None:
    response = _response(
        ("interfaces/interface[name=swp10]/state", {"admin-status": "UP", "oper-status": "UP"}),
        ("interfaces/interface[name=swp2]/state/admin-status", "DOWN"),
        ("interfaces/interface[name=swp2]/state/oper-status", "DOWN"),
    )
    switch, _ = _switch({("interfaces/interface/state",): response})
    assert switch.ports == [
        Port(id="swp2", admin_up=False, oper_up=False),
        Port(id="swp10", admin_up=True, oper_up=True),
    ]


def test_port_reads_one_interface_under_a_prefix() -> None:
    response = _response(
        ("state", {"admin-status": "DOWN", "oper-status": "DOWN"}), prefix="interfaces/interface[name=Ethernet1/1]"
    )
    switch, _ = _switch({("interfaces/interface[name=Ethernet1/1]/state",): response})
    assert switch.port("Ethernet1/1") == Port(id="Ethernet1/1", admin_up=False, oper_up=False)


def test_port_not_found() -> None:
    switch, _ = _switch({("interfaces/interface[name=Ethernet9/9]/state",): _Rpc(grpc.StatusCode.NOT_FOUND, "no such")})
    with pytest.raises(SwitchAPIError, match="port Ethernet9/9 not found"):
        switch.port("Ethernet9/9")


# --- VLANs, LLDP, MAC table ---------------------------------------------------


def _switched(name: str, mode: str, **state) -> dict:
    return _interface(name) | {
        "ethernet": {"switched-vlan": {"state": {"interface-mode": f"openconfig-vlan-types:{mode}", **state}}}
    }


_MEMBERS = {
    ("interfaces/interface/state", "interfaces/interface/ethernet/switched-vlan/state"): _response(
        (
            "interfaces",
            {
                "interface": [
                    _switched("Ethernet1/3", "TRUNK", **{"native-vlan": 1, "trunk-vlans": [10, "20..21"]}),
                    _switched("Ethernet1/1", "ACCESS", **{"access-vlan": 10}),
                    _interface("Ethernet1/2"),
                ]
            },
        )
    )
}


def test_vlans_with_their_member_ports() -> None:
    vlans = {"network-instance": [{"name": "default", "vlans": {"vlan": [{"vlan-id": 20}, {"vlan-id": 10}]}}]}
    switch, _ = _switch({(f"{NI}/vlans",): _response(("network-instances", vlans))} | _MEMBERS)
    got = {vlan.id: [port.id for port in vlan.ports] for vlan in switch.vlans}
    assert got == {"10": ["Ethernet1/1", "Ethernet1/3"], "20": ["Ethernet1/3"]}


def test_vlan_members_from_one_json_tree_per_path() -> None:
    switched = {"switched-vlan": {"state": {"interface-mode": "ACCESS", "access-vlan": 10}}}
    response = _response(
        ("interfaces", {"interface": [_interface("Ethernet1/1", oper="DOWN")]}),
        ("interfaces", {"interface": [{"name": "Ethernet1/1", "ethernet": switched}]}),
    )
    vlans = {"network-instance": [{"name": "default", "vlans": {"vlan": [{"vlan-id": 10}]}}]}
    switch, _ = _switch({(f"{NI}/vlans",): _response(("network-instances", vlans)), next(iter(_MEMBERS)): response})
    assert [vlan.ports for vlan in switch.vlans] == [(Port(id="Ethernet1/1", admin_up=True, oper_up=False),)]


def test_vlan_not_found() -> None:
    switch, _ = _switch()
    with pytest.raises(SwitchAPIError, match="VLAN 30 not found"):
        switch.vlan("30")


def test_vlan_reads_one_vlan() -> None:
    response = _response((f"{NI}/vlans/vlan[vlan-id=21]/state", {"vlan-id": 21, "name": "v21"}))
    switch, _ = _switch({(f"{NI}/vlans/vlan[vlan-id=21]",): response} | _MEMBERS)
    assert [port.id for port in switch.vlan("21").ports] == ["Ethernet1/3"]


def test_lldp_neighbors_with_a_mac_port_id_on_data_ports() -> None:
    lldp = {
        "interfaces": {
            "interface": [
                {
                    "name": "Ethernet1/2",
                    "neighbors": {"neighbor": [{"id": "n1", "state": {"port-id": "02dd.0000.0201"}}]},
                },
                {"name": "Ethernet1/1", "neighbors": {"neighbor": [{"id": "n2", "state": {"port-id": "Ethernet7"}}]}},
                {"name": "mgmt0", "neighbors": {"neighbor": [{"id": "n3", "state": {"port-id": "aa:bb:cc:dd:ee:ff"}}]}},
            ]
        }
    }
    interfaces = {"interface": [_interface("Ethernet1/1"), _interface("Ethernet1/2"), _interface("mgmt0")]}
    switch, _ = _switch(
        {
            ("lldp/interfaces/interface/neighbors", "interfaces/interface/state"): _response(
                ("lldp", lldp), ("interfaces", interfaces)
            )
        }
    )
    assert switch.lldp_neighbors == [
        LldpNeighbor(interface="Ethernet1/2", switch_port=2, chassis_mac="02:dd:00:00:02:01")
    ]


def test_mac_table() -> None:
    entries = [
        {
            "mac-address": "00:1C:73:00:AB:CD",
            "vlan": 10,
            "state": {"entry-type": "STATIC"},
            "interface": {"interface-ref": {"state": {"interface": "Ethernet1/1"}}},
        },
        {
            "mac-address": "00:1c:73:00:ab:ce",
            "vlan": 10,
            "state": {"entry-type": "DYNAMIC"},
            "interface": {"interface-ref": {"state": {"interface": "Ethernet1/2"}}},
        },
        {"mac-address": "00:1c:73:00:ab:cf", "vlan": 10, "state": {"entry-type": "DYNAMIC"}},  # no port
    ]
    response = _response((f"{NI}/fdb/mac-table/entries", {"entry": entries}))
    switch, _ = _switch({(f"{NI}/fdb/mac-table/entries",): response})
    assert switch.mac_table == [
        MacEntry(mac="00:1c:73:00:ab:cd", port="Ethernet1/1", vlan=10, static=True),
        MacEntry(mac="00:1c:73:00:ab:ce", port="Ethernet1/2", vlan=10, static=False),
    ]


def test_flow_control_is_not_supported() -> None:
    switch, _ = _switch()
    with pytest.raises(NotImplementedError, match="no OpenConfig model"):
        _ = switch.flow_control
    with pytest.raises(SwitchNotSupportedError):
        _ = switch.flow_control


# --- port bridge --------------------------------------------------------------


@pytest.mark.parametrize(
    ("state", "mode", "native_vlan"),
    [
        ({"interface-mode": "openconfig-vlan-types:TRUNK", "native-vlan": 1, "access-vlan": 9}, "trunk", 1),
        ({"interface-mode": "ACCESS", "access-vlan": 10}, "access", 10),
    ],
)
def test_port_bridge(state: dict, mode: str, native_vlan: int) -> None:
    path = "interfaces/interface[name=Ethernet1/1]/ethernet/switched-vlan/state"
    switch, _ = _switch(
        {
            (path,): _response((path, state)),
            (f"{NI}/fdb/state/mac-learning",): _response((f"{NI}/fdb/state/mac-learning", True)),
        }
    )
    bridge = switch.port_bridge("Ethernet1/1")
    assert (bridge.mode, bridge.native_vlan, bridge.learning) == (mode, native_vlan, True)


def test_port_bridge_of_a_routed_port_and_unsupported_learning() -> None:
    switch, _ = _switch(
        {(f"{NI}/fdb/state/mac-learning",): _Rpc(grpc.StatusCode.UNIMPLEMENTED, "fdb")},
    )
    bridge = switch.port_bridge("Ethernet1/1")
    assert (bridge.mode, bridge.native_vlan, bridge.learning) == (None, None, None)


# --- writes -------------------------------------------------------------------


def test_set_port_admin_state() -> None:
    switch, client = _switch()
    switch.set_port_admin_state("Ethernet1/1", False)
    assert client.sets == [
        {"delete": None, "replace": None, "update": [("interfaces/interface[name=Ethernet1/1]/config/enabled", False)]}
    ]


def test_delete_vlan_removes_it_from_every_port_in_one_set() -> None:
    def port(name: str, **config) -> dict:
        return {"name": name, "ethernet": {"switched-vlan": {"config": config}}}

    tree = {
        "interface": [
            port("Ethernet1/1", **{"access-vlan": 10}),
            port("Ethernet1/2", **{"native-vlan": 10, "trunk-vlans": [10]}),
            port("Ethernet1/3", **{"trunk-vlans": [5, "8..12"]}),
            port("Ethernet1/4", **{"trunk-vlans": [5]}),
        ]
    }
    switch, client = _switch({("interfaces/interface/ethernet/switched-vlan/config",): _response(("interfaces", tree))})
    switch.delete_vlan("10")
    base = "interfaces/interface[name={}]/ethernet/switched-vlan/config"
    assert client.sets == [
        {
            "delete": [
                f"{NI}/vlans/vlan[vlan-id=10]",
                base.format("Ethernet1/1") + "/access-vlan",
                base.format("Ethernet1/2") + "/native-vlan",
                base.format("Ethernet1/2") + "/trunk-vlans",
            ],
            "replace": [(base.format("Ethernet1/3") + "/trunk-vlans", [5, "8..9", "11..12"])],
            "update": None,
        }
    ]


@pytest.mark.parametrize(
    ("trunks", "vid", "rest"),
    [
        ([10, 20], 10, [20]),
        (["10..20"], 10, ["11..20"]),
        (["10..20"], 20, ["10..19"]),
        (["10..20"], 15, ["10..14", "16..20"]),
        (["10..12"], 11, [10, 12]),
        (["10..20"], 30, ["10..20"]),
    ],
)
def test_without(trunks: list, vid: int, rest: list) -> None:
    assert _without(trunks, vid) == rest


# --- FEC ----------------------------------------------------------------------

_FEC = "interfaces/interface[name=Ethernet1/1]/ethernet/config/fec-mode"
_SPEED = "interfaces/interface[name=Ethernet1/1]/ethernet/state/port-speed"


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (None, None),
        ("openconfig-if-ethernet:FEC_AUTO", None),
        ("openconfig-if-ethernet:FEC_DISABLED", FecEncoding.OFF),
        ("FEC_FC", FecEncoding.FIRE_CODE),
        ("FEC_RS528", FecEncoding.REED_SOLOMON),
        ("FEC_RS544_2X_INTERLEAVE", FecEncoding.REED_SOLOMON),
        ("openconfig-if-ethernet:FEC_RS544_2X_INTERLEAVE_CL172", FecEncoding.REED_SOLOMON),
    ],
)
def test_port_fec(mode: str | None, expected: FecEncoding | None) -> None:
    switch, _ = _switch({(_FEC,): _response((_FEC, mode)) if mode else {}})
    assert switch.port_fec("Ethernet1/1") == expected


def test_port_fec_unknown_mode() -> None:
    switch, _ = _switch({(_FEC,): _response((_FEC, "FEC_NEW"))})
    with pytest.raises(SwitchAPIError, match="unknown FEC mode 'FEC_NEW'"):
        switch.port_fec("Ethernet1/1")


def test_port_fec_the_switch_does_not_map() -> None:
    switch, _ = _switch({(_FEC,): _Rpc(grpc.StatusCode.UNKNOWN, "Property is not mapped")})
    with pytest.raises(SwitchNotSupportedError, match="not supported by the switch"):
        switch.port_fec("Ethernet1/1")


@pytest.mark.parametrize(
    ("fec", "speed", "encoding", "value"),
    [
        (FecEncoding.OFF, None, "json_ietf", "openconfig-if-ethernet:FEC_DISABLED"),
        (FecEncoding.FIRE_CODE, None, "json", "FEC_FC"),
        (FecEncoding.REED_SOLOMON, "openconfig-if-ethernet:SPEED_100GB", "json", "FEC_RS528"),
        (FecEncoding.REED_SOLOMON, "SPEED_400GB", "json", "FEC_RS544"),
        (FecEncoding.REED_SOLOMON, None, "json", "FEC_RS528"),
    ],
)
def test_set_port_fec(fec: FecEncoding, speed: str | None, encoding: str, value: str) -> None:
    switch, client = _switch({(_SPEED,): _response((_SPEED, speed)) if speed else {}}, encoding=encoding)
    switch.set_port_fec("Ethernet1/1", fec)
    assert client.sets == [{"delete": None, "replace": None, "update": [(_FEC, value)]}]


def test_set_port_fec_to_the_switch_default() -> None:
    switch, client = _switch()
    switch.set_port_fec("Ethernet1/1", None)
    assert client.sets == [{"delete": [_FEC], "replace": None, "update": None}]


# --- SVI ----------------------------------------------------------------------

_SVI = "interfaces/interface[name=vlan5]"


def test_svi_address_prefers_ipv4_and_skips_link_local() -> None:
    routed = {
        "ipv6": {
            "addresses": {
                "address": [
                    {"ip": "fe80::1", "config": {"prefix-length": 64}},
                    {"ip": "fd00:dec0::1", "state": {"prefix-length": 64}},
                ]
            }
        }
    }
    switch, _ = _switch({(f"{_SVI}/routed-vlan",): _response((f"{_SVI}/routed-vlan", routed))})
    assert switch.svi_address("5") == ip_interface("fd00:dec0::1/64")

    routed["ipv4"] = {"addresses": {"address": [{"ip": "10.0.0.1", "config": {"ip": "10.0.0.1", "prefix-length": 24}}]}}
    switch, _ = _switch({(f"{_SVI}/routed-vlan",): _response((f"{_SVI}/routed-vlan", routed))})
    assert switch.svi_address("5") == ip_interface("10.0.0.1/24")


def test_svi_address_without_an_svi() -> None:
    switch, _ = _switch({(f"{_SVI}/routed-vlan",): _Rpc(grpc.StatusCode.NOT_FOUND, "no vlan5")})
    assert switch.svi_address("5") is None


def test_set_svi_address_replaces_every_address() -> None:
    switch, client = _switch()
    switch.set_svi_address("5", "fd00:dec0::1/64")
    assert client.sets == [
        {
            "delete": [f"{_SVI}/routed-vlan/ipv4/addresses", f"{_SVI}/routed-vlan/ipv6/addresses"],
            "replace": None,
            "update": [
                (
                    f"{_SVI}/routed-vlan/ipv6/addresses/address[ip=fd00:dec0::1]/config",
                    {"ip": "fd00:dec0::1", "prefix-length": 64},
                )
            ],
        }
    ]


def test_set_svi_address_none_removes_the_svi_and_tolerates_none_there() -> None:
    switch, client = _switch()
    switch.set_svi_address("5", None)
    assert client.sets == [{"delete": [_SVI], "replace": None, "update": None}]

    switch, _ = _switch()
    switch._set = MagicMock(side_effect=openconfig_switch._PathNotFound("gone"))  # type: ignore[method-assign]
    switch.set_svi_address("5", None)


def test_svi_name_is_configurable() -> None:
    switch, client = _switch()
    switch._svi_name = "Vlan{}"
    switch.set_svi_address("7", None)
    assert client.sets[0]["delete"] == ["interfaces/interface[name=Vlan7]"]


# --- status -------------------------------------------------------------------


def test_status(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(openconfig_switch.time, "time", lambda: 1_000.0)
    components = {
        "component": [
            {"name": "OS", "state": {"type": "openconfig-platform-types:OPERATING_SYSTEM", "software-version": "10.5"}},
            {
                "name": "Chassis",
                "state": {"type": "CHASSIS", "part-no": "N9K-C93180", "serial-no": "SN1", "firmware-version": "bios1"},
            },
        ]
    }
    switch, _ = _switch(
        {
            ("system/state",): _response(("system/state", {"hostname": "nexus", "boot-time": "400000000000"})),
            ("components/component/state",): _response(("components", components)),
        }
    )
    status = switch.status
    assert (status.hostname, status.uptime, status.model, status.serial_number) == ("nexus", "600", "N9K-C93180", "SN1")
    assert (status.firmware_version, status.software_version) == ("bios1", "10.5")


def test_status_without_components() -> None:
    switch, _ = _switch(
        {
            ("system/state",): _response(("system/state", {"hostname": "nexus"})),
            ("components/component/state",): _Rpc(grpc.StatusCode.UNIMPLEMENTED, "components"),
        }
    )
    status = switch.status
    assert (status.hostname, status.uptime, status.model) == ("nexus", None, None)


# --- the gNMI session and its errors ------------------------------------------


@pytest.mark.parametrize(
    ("code", "details", "error"),
    [
        (grpc.StatusCode.UNIMPLEMENTED, "Get", SwitchNotSupportedError),
        (grpc.StatusCode.INVALID_ARGUMENT, "Property is not mapped", SwitchNotSupportedError),
        (grpc.StatusCode.INVALID_ARGUMENT, "bad value", SwitchAPIError),
        (grpc.StatusCode.NOT_FOUND, "missing", openconfig_switch._PathNotFound),
    ],
)
def test_switch_error_from_a_wrapped_rpc_error(code: grpc.StatusCode, details: str, error: type) -> None:
    wrapped = openconfig_switch.gNMIException("Get failed", _Rpc(code, details))
    got = openconfig_switch._switch_error("Get", wrapped)
    assert type(got) is error
    assert details in str(got)


def test_call_picks_json_ietf_once_and_wraps_errors() -> None:
    client = MagicMock()
    client.capabilities.return_value = {"supported_encodings": ["proto", "json", "json_ietf"]}
    client.get.return_value = {}
    session = MagicMock()
    session.__enter__.return_value = client
    with patch.object(openconfig_switch, "gNMIclient", return_value=session) as gnmi:
        switch = OpenConfigSwitch(
            NetworkSwitchConfig(host="h", username="u", password="p", port=50051, verify_tls=False)
        )
        assert switch._get("system/state") == {}
        assert switch._encoding == "json_ietf"
        gnmi.assert_called_once_with(target=("h", 50051), username="u", password="p", skip_verify=True, gnmi_timeout=60)

        client.get.side_effect = openconfig_switch.gNMIException("Get", _Rpc(grpc.StatusCode.UNAVAILABLE, "down"))
        with pytest.raises(SwitchAPIError, match="down"):
            switch._get("system/state")
    assert client.capabilities.call_count == 1


def test_call_wraps_an_unreachable_switch() -> None:
    session = MagicMock()
    session.__enter__.side_effect = grpc.FutureTimeoutError()
    with patch.object(openconfig_switch, "gNMIclient", return_value=session):
        switch = OpenConfigSwitch(NetworkSwitchConfig(host="h", username="u", password="p"))
        with pytest.raises(SwitchAPIError, match="gNMI Get"):
            switch._get("system/state")


def test_call_rejects_a_switch_without_json() -> None:
    client = MagicMock()
    client.capabilities.return_value = {"supported_encodings": ["proto"]}
    session = MagicMock()
    session.__enter__.return_value = client
    with patch.object(openconfig_switch, "gNMIclient", return_value=session):
        switch = OpenConfigSwitch(NetworkSwitchConfig(host="h", username="u", password="p"))
        with pytest.raises(SwitchNotSupportedError, match="neither JSON encoding"):
            switch._get("system/state")
