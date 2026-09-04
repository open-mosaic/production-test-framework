# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2026 Delos Data, Inc.

from unittest.mock import MagicMock

from production_test_framework.switch.arista.arista_eos_switch import AristaEosSwitch
from production_test_framework.switch.models import NetworkSwitchConfig

_MAC_TABLE = {
    "unicastTable": {
        "tableEntries": [
            {"macAddress": "02:dd:00:00:22:00", "interface": "Ethernet33", "vlanId": 1, "entryType": "static"},
            {"macAddress": "02:DD:00:00:01:00", "interface": "Ethernet1", "vlanId": 3000, "entryType": "static"},
            {"macAddress": "aa:bb:cc:dd:ee:ff", "interface": "Ethernet5", "vlanId": 100, "entryType": "dynamic"},
            # no interface -> dropped
            {"macAddress": "00:11:22:33:44:55", "interface": "", "vlanId": 1, "entryType": "dynamic"},
        ]
    }
}


def _switch() -> AristaEosSwitch:
    switch = AristaEosSwitch(NetworkSwitchConfig(host="h", username="u", password="p", verify_tls=False))
    switch._node = MagicMock()
    return switch


def test_parse_mac_table() -> None:
    entries = _switch()._parse_mac_table(_MAC_TABLE["unicastTable"]["tableEntries"])
    # MAC normalised to lowercase; the entry with no interface is dropped.
    assert [(e.port, e.mac, e.vlan, e.static) for e in entries] == [
        ("Ethernet33", "02:dd:00:00:22:00", 1, True),
        ("Ethernet1", "02:dd:00:00:01:00", 3000, True),
        ("Ethernet5", "aa:bb:cc:dd:ee:ff", 100, False),
    ]


def test_mac_table_calls_show_mac_address_table() -> None:
    switch = _switch()
    switch._node.run_commands.return_value = [_MAC_TABLE]
    entries = switch.mac_table

    assert len(entries) == 3
    switch._node.run_commands.assert_called_once_with(["show mac address-table"])


_MAC_TABLE_BY_INTERFACE = {
    "unicastTable": {
        "tableEntries": [
            {"macAddress": "02:dd:00:00:29:00", "interface": "Ethernet41", "vlanId": 1, "entryType": "static"},
            {"macAddress": "02:dd:00:00:2a:00", "interface": "Ethernet41", "vlanId": 100, "entryType": "static"},
            {"macAddress": "02:dd:00:00:2b:00", "interface": "Ethernet49/1", "vlanId": 1, "entryType": "static"},
            {"macAddress": "02:dd:00:00:2c:00", "interface": "Ethernet49/2", "vlanId": 1, "entryType": "static"},
        ]
    }
}


def test_mac_table_by_interface_keeps_breakout_lanes_distinct() -> None:
    switch = _switch()
    switch._node.run_commands.return_value = [_MAC_TABLE_BY_INTERFACE]

    # an interface's MACs collect into one set; breakout lanes 49/1 and 49/2 stay distinct.
    assert switch.mac_table_by_interface == {
        "Ethernet41": {"02:dd:00:00:29:00", "02:dd:00:00:2a:00"},
        "Ethernet49/1": {"02:dd:00:00:2b:00"},
        "Ethernet49/2": {"02:dd:00:00:2c:00"},
    }
