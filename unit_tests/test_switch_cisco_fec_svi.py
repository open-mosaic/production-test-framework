# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2026 Delos Data, Inc.

from ipaddress import ip_interface
from unittest.mock import MagicMock

import pytest

from production_test_framework.switch.cisco.nxos_switch import CiscoNxosSwitch
from production_test_framework.switch.exceptions import SwitchAPIError
from production_test_framework.switch.models import FecEncoding, NetworkSwitchConfig


def _switch(text: str | Exception = "") -> CiscoNxosSwitch:
    """A driver whose text commands return text (or raise it) and whose config commands are
    recorded in _run_config."""
    switch = CiscoNxosSwitch(NetworkSwitchConfig(host="h", username="u", password="p", verify_tls=False))
    switch._session = MagicMock()
    switch._run_text = MagicMock(side_effect=text if isinstance(text, Exception) else None, return_value=text)
    switch._run_config = MagicMock()  # type: ignore[method-assign]
    return switch


def _running(*lines: str) -> str:
    return "\n!Command: show running-config\n\nversion 10.5(3)\n\n" + "\n".join(lines) + "\n"


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        (None, None),
        ("  fec auto", None),
        ("  fec off", FecEncoding.OFF),
        ("  fec fc-fec", FecEncoding.FIRE_CODE),
        ("  fec rs-fec", FecEncoding.REED_SOLOMON),
        ("  fec kp-fec", FecEncoding.REED_SOLOMON),
        ("  fec rs-ieee", FecEncoding.REED_SOLOMON),
    ],
)
def test_port_fec_reads_running_config(line: str | None, expected: FecEncoding | None) -> None:
    config = _running("interface Ethernet1/1", "  no shutdown", *([line] if line else []))
    switch = _switch(config)
    assert switch.port_fec("Ethernet1/1") == expected
    switch._run_text.assert_called_once_with("show running-config interface Ethernet1/1")


def test_port_fec_unknown_mode_raises() -> None:
    switch = _switch(_running("interface Ethernet1/1", "  fec some-new-mode"))
    with pytest.raises(SwitchAPIError, match="unknown FEC mode 'some-new-mode'"):
        switch.port_fec("Ethernet1/1")


@pytest.mark.parametrize(
    ("fec", "command"),
    [
        (None, "no fec"),
        (FecEncoding.OFF, "fec off"),
        (FecEncoding.FIRE_CODE, "fec fc-fec"),
        (FecEncoding.REED_SOLOMON, "fec rs-fec"),  # at any speed: the CLI refuses kp-fec
    ],
)
def test_set_port_fec(fec: FecEncoding | None, command: str) -> None:
    switch = _switch()
    switch.set_port_fec("Ethernet1/1", fec)
    switch._run_config.assert_called_once_with(["interface Ethernet1/1", command])


@pytest.mark.parametrize(
    ("lines", "expected"),
    [
        (["interface Vlan100"], None),
        (["interface Vlan100", "  ip address 10.0.0.1/24", "  ip address 10.0.1.1/24 secondary"], "10.0.0.1/24"),
        (["interface Vlan100", "  ipv6 address fd00::1/64"], "fd00::1/64"),
        (["interface Vlan100", "  ipv6 address use-link-local-only"], None),
        (["interface Vlan100", "  ipv6 address fe80::1/64 link-local", "  ip address 10.0.0.1/24"], "10.0.0.1/24"),
    ],
)
def test_svi_address_reads_running_config(lines: list[str], expected: str | None) -> None:
    switch = _switch(_running(*lines))
    assert switch.svi_address("100") == (ip_interface(expected) if expected else None)
    switch._run_text.assert_called_once_with("show running-config interface vlan100")


def test_svi_address_missing_svi_is_none() -> None:
    # NX-OS refuses to show the running config of an SVI that doesn't exist.
    switch = _switch(SwitchAPIError("NX-API command failed: show running-config interface vlan100: Invalid range"))
    assert switch.svi_address("100") is None


def test_svi_address_other_errors_raise() -> None:
    switch = _switch(SwitchAPIError("NX-API request failed: connection refused"))
    with pytest.raises(SwitchAPIError, match="refused"):
        switch.svi_address("100")


@pytest.mark.parametrize(
    ("address", "last"),
    [("fd00:dec0::1/64", "ipv6 address fd00:dec0::1/64"), (ip_interface("10.0.0.1/24"), "ip address 10.0.0.1/24")],
)
def test_set_svi_address_replaces_the_address(address: object, last: str) -> None:
    switch = _switch()
    switch.set_svi_address("4000", address)
    switch._run_config.assert_called_once_with(["interface vlan4000", "no ip address", "no ipv6 address", last])


def test_set_svi_address_none_removes_the_svi() -> None:
    switch = _switch()
    switch.set_svi_address("4000", None)
    switch._run_config.assert_called_once_with(["no interface vlan4000"])


def test_run_text_uses_cli_ascii() -> None:
    switch = CiscoNxosSwitch(NetworkSwitchConfig(host="h", username="u", password="p", verify_tls=False))
    switch._session = MagicMock()
    response = MagicMock(status_code=200)
    response.json.return_value = {"jsonrpc": "2.0", "result": {"msg": "interface Vlan1\n"}}
    switch._session.post.return_value = response

    assert switch._run_text("show running-config interface vlan1") == "interface Vlan1\n"
    assert switch._session.post.call_args.kwargs["json"]["method"] == "cli_ascii"


def test_run_text_without_a_message_raises() -> None:
    # An empty reply must not read as "no FEC line" or "no SVI".
    switch = CiscoNxosSwitch(NetworkSwitchConfig(host="h", username="u", password="p", verify_tls=False))
    switch._session = MagicMock()
    response = MagicMock(status_code=200)
    response.json.return_value = {"jsonrpc": "2.0", "result": None}
    switch._session.post.return_value = response
    with pytest.raises(SwitchAPIError, match="unexpected NX-API response"):
        switch._run_text("show running-config interface Ethernet1/1")
