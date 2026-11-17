# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2026 Delos Data, Inc.

import pytest

from production_test_framework.switch.arista.arista_eos_switch import AristaEosSwitch
from production_test_framework.switch.nvidia.nvidia_cumulus_switch import NvidiaCumulusSwitch
from production_test_framework.switch.openconfig.openconfig_switch import OpenConfigSwitch
from production_test_framework.switch.switch_status import create_switch


def test_create_switch_builds_arista_driver() -> None:
    switch = create_switch("arista-eos", "h", "u", "p")
    assert isinstance(switch, AristaEosSwitch)


def test_create_switch_builds_nvidia_driver() -> None:
    switch = create_switch("nvidia-cumulus", "h", "u", "p")
    assert isinstance(switch, NvidiaCumulusSwitch)


@pytest.mark.parametrize(
    ("hostname", "host", "port"),
    [
        ("10.0.0.1", "10.0.0.1", 9339),
        ("10.0.0.1:50051", "10.0.0.1", 50051),
        ("fd00::1", "fd00::1", 9339),
        ("[fd00::1]", "fd00::1", 9339),
        ("[fd00::1]:50051", "fd00::1", 50051),
    ],
)
def test_create_switch_builds_openconfig_driver(hostname: str, host: str, port: int) -> None:
    switch = create_switch("openconfig", hostname, "u", "p")
    assert isinstance(switch, OpenConfigSwitch)
    assert (switch._config.host, switch._config.port) == (host, port)


def test_create_switch_rejects_unknown_type() -> None:
    with pytest.raises(SystemExit):
        create_switch("bogus", "h", "u", "p")
