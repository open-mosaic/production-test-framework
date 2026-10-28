# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2026 Delos Data, Inc.

from ipaddress import ip_interface
from unittest.mock import MagicMock

import pytest

from production_test_framework.switch.arista.arista_eos_switch import AristaEosSwitch
from production_test_framework.switch.models import FecEncoding, NetworkSwitchConfig


def _switch(running_config: str = "") -> AristaEosSwitch:
    switch = AristaEosSwitch(NetworkSwitchConfig(host="h", username="u", password="p", verify_tls=False))
    switch._node = MagicMock()
    switch._node.run_commands.return_value = [{"output": running_config}]
    return switch


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ("interface Ethernet1\n   switchport mode trunk\n", None),
        ("interface Ethernet1\n   no error-correction encoding\n", FecEncoding.OFF),
        ("interface Ethernet1\n   error-correction encoding fire-code\n", FecEncoding.FIRE_CODE),
        ("interface Ethernet1\n   error-correction encoding reed-solomon\n", FecEncoding.REED_SOLOMON),
        (
            "interface Ethernet1\n   error-correction encoding fire-code\n   error-correction encoding reed-solomon\n",
            None,
        ),
    ],
)
def test_port_fec_reads_running_config(config: str, expected: FecEncoding | None) -> None:
    switch = _switch(config)
    assert switch.port_fec("Ethernet1") == expected
    switch._node.run_commands.assert_called_once_with(["show running-config interfaces Ethernet1"], encoding="text")


@pytest.mark.parametrize(
    ("fec", "commands"),
    [
        (None, ["default error-correction encoding"]),
        (FecEncoding.OFF, ["no error-correction encoding"]),
        (FecEncoding.FIRE_CODE, ["error-correction encoding fire-code", "no error-correction encoding reed-solomon"]),
        (
            FecEncoding.REED_SOLOMON,
            ["error-correction encoding reed-solomon", "no error-correction encoding fire-code"],
        ),
    ],
)
def test_set_port_fec(fec: FecEncoding | None, commands: list[str]) -> None:
    switch = _switch()
    switch.set_port_fec("Ethernet1", fec)
    switch._node.config.assert_called_once_with(["interface Ethernet1", *commands])


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ("", None),
        ("interface Vlan100\n   no autostate\n", None),
        ("interface Vlan100\n   ip address 10.0.0.1/24\n   ip address 10.0.1.1/24 secondary\n", "10.0.0.1/24"),
        ("interface Vlan100\n   ipv6 address fd00::1/64\n", "fd00::1/64"),
        ("interface Vlan100\n   ip address dhcp\n", None),
        ("interface Vlan100\n   ip address dhcp\n   ipv6 address fd00::1/64\n", "fd00::1/64"),
    ],
)
def test_svi_address_reads_running_config(config: str, expected: str | None) -> None:
    switch = _switch(config)
    assert switch.svi_address("100") == (ip_interface(expected) if expected else None)
    switch._node.run_commands.assert_called_once_with(["show running-config interfaces Vlan100"], encoding="text")


@pytest.mark.parametrize(
    ("address", "last"),
    [("fd00:dec0::1/64", "ipv6 address fd00:dec0::1/64"), (ip_interface("10.0.0.1/24"), "ip address 10.0.0.1/24")],
)
def test_set_svi_address_replaces_the_address(address: object, last: str) -> None:
    switch = _switch()
    switch.set_svi_address("4000", address)
    switch._node.config.assert_called_once_with(["interface Vlan4000", "no ip address", "no ipv6 address", last])


def test_set_svi_address_none_removes_the_svi() -> None:
    switch = _switch()
    switch.set_svi_address("4000", None)
    switch._node.config.assert_called_once_with(["no interface Vlan4000"])


@pytest.mark.parametrize("address", ["10.0.0.1", "10.0.0.1/33", "eth1/1"])
def test_set_svi_address_rejects_bad_addresses(address: str) -> None:
    switch = _switch()
    with pytest.raises(ValueError):
        switch.set_svi_address("4000", address)
    switch._node.config.assert_not_called()
