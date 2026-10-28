# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2026 Delos Data, Inc.

from ipaddress import ip_interface
from unittest.mock import MagicMock, call

import pytest

from production_test_framework.switch.exceptions import SwitchAPIError
from production_test_framework.switch.models import FecEncoding, NetworkSwitchConfig
from production_test_framework.switch.nvidia.nvidia_cumulus_switch import NvidiaCumulusSwitch

REVISION_ID = "changeset/cumulus/2026-01-01_00.00.00_ABCD"


def _switch(get: dict | SwitchAPIError | None = None) -> NvidiaCumulusSwitch:
    """A driver whose reads return get (or raise it) and whose writes are recorded in _run_api_write."""
    switch = NvidiaCumulusSwitch(
        NetworkSwitchConfig(host="10.0.0.1", username="admin", password="secret", verify_tls=False, port=8765)
    )
    switch._run_api_call = MagicMock(  # type: ignore[method-assign]
        side_effect=get if isinstance(get, Exception) else None, return_value=get or {}
    )
    switch._create_revision = MagicMock(return_value=REVISION_ID)  # type: ignore[method-assign]
    switch._apply_revision = MagicMock()  # type: ignore[method-assign]
    switch._run_api_write = MagicMock()  # type: ignore[method-assign]
    return switch


def _not_found() -> SwitchAPIError:
    return SwitchAPIError("API call failed: 404 not found", 404)


@pytest.mark.parametrize(
    ("link", "expected"),
    [
        ({}, None),
        ({"fec": "auto"}, None),
        ({"fec": "off"}, FecEncoding.OFF),
        ({"fec": "baser"}, FecEncoding.FIRE_CODE),
        ({"fec": "rs"}, FecEncoding.REED_SOLOMON),
    ],
)
def test_port_fec_reads_applied_link_fec(link: dict, expected: FecEncoding | None) -> None:
    switch = _switch({"link": link})
    assert switch.port_fec("swp1") == expected
    switch._run_api_call.assert_called_once_with("/interface/swp1", params={"rev": "applied"})


@pytest.mark.parametrize(
    ("fec", "value"),
    [(None, "auto"), (FecEncoding.OFF, "off"), (FecEncoding.FIRE_CODE, "baser"), (FecEncoding.REED_SOLOMON, "rs")],
)
def test_set_port_fec_patches_link_fec_in_a_changeset(fec: FecEncoding | None, value: str) -> None:
    switch = _switch()
    switch.set_port_fec("swp1", fec)
    switch._run_api_write.assert_called_once_with(
        "PATCH", "/interface/swp1", params={"rev": REVISION_ID}, json_body={"link": {"fec": value}}
    )
    switch._apply_revision.assert_called_once_with(REVISION_ID)


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ({"type": "svi"}, None),
        ({"type": "svi", "ip": {"address": {"10.0.0.1/24": {}}}}, "10.0.0.1/24"),
        ({"type": "svi", "ip": {"address": {"fe80::1/64": {}, "fd00::1/64": {}}}}, "fd00::1/64"),
    ],
)
def test_svi_address_reads_applied_svi(body: dict, expected: str | None) -> None:
    switch = _switch(body)
    assert switch.svi_address("100") == (ip_interface(expected) if expected else None)
    switch._run_api_call.assert_called_once_with("/interface/vlan100", params={"rev": "applied"})


def test_svi_address_without_svi_is_none() -> None:
    assert _switch(_not_found()).svi_address("100") is None


def test_set_svi_address_replaces_the_address_in_one_changeset() -> None:
    switch = _switch()
    switch.set_svi_address("4000", "fd00:dec0::1/64")
    assert switch._run_api_write.call_args_list == [
        call("DELETE", "/interface/vlan4000/ip/address", params={"rev": REVISION_ID}),
        call(
            "PATCH",
            "/interface",
            params={"rev": REVISION_ID},
            json_body={"vlan4000": {"type": "svi", "vlan": 4000, "ip": {"address": {"fd00:dec0::1/64": {}}}}},
        ),
    ]
    switch._apply_revision.assert_called_once_with(REVISION_ID)


def test_set_svi_address_creates_a_missing_svi() -> None:
    switch = _switch()
    switch._run_api_write.side_effect = [_not_found(), None]  # no addresses to delete yet
    switch.set_svi_address("4000", "10.0.0.1/24")
    assert switch._run_api_write.call_count == 2
    switch._apply_revision.assert_called_once_with(REVISION_ID)


def test_set_svi_address_none_removes_the_svi() -> None:
    switch = _switch()
    switch.set_svi_address("4000", None)
    switch._run_api_write.assert_called_once_with("DELETE", "/interface/vlan4000", params={"rev": REVISION_ID})
    switch._apply_revision.assert_called_once_with(REVISION_ID)


def test_set_svi_address_other_errors_raise() -> None:
    switch = _switch()
    switch._run_api_write.side_effect = SwitchAPIError("API call failed: 500", 500)
    with pytest.raises(SwitchAPIError):
        switch.set_svi_address("4000", "10.0.0.1/24")
    switch._apply_revision.assert_not_called()


@pytest.mark.parametrize(("vlan_id", "address"), [("4000", "10.0.0.1"), ("vlan4000", "10.0.0.1/24")])
def test_set_svi_address_rejects_bad_arguments_before_any_changeset(vlan_id: str, address: str) -> None:
    switch = _switch()
    with pytest.raises(ValueError):
        switch.set_svi_address(vlan_id, address)
    switch._create_revision.assert_not_called()


def test_write_errors_carry_the_status_code() -> None:
    switch = NvidiaCumulusSwitch(
        NetworkSwitchConfig(host="10.0.0.1", username="admin", password="secret", verify_tls=False, port=8765)
    )
    response = MagicMock(status_code=404, text="not found")
    switch._session.request = MagicMock(return_value=response)
    with pytest.raises(SwitchAPIError) as raised:
        switch._run_api_write("DELETE", "/interface/vlan1")
    assert raised.value.status_code == 404
