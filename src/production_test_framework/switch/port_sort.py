# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2025 Delos Data, Inc.

from production_test_framework.switch.models import Port
from production_test_framework.switch.portname import NO_SUBPORT, parse, parse_cisco


def port_id_sort_key(port_id: str) -> tuple[str, int, int, int]:
    """Sort key ordering names the way the switch lays its ports out: swp6 before
    swp59, and swp1s0 before swp1s1 before swp2s0.

    Both halves need the numeric ordering: a lexicographic sort puts swp59 ahead of
    swp6, and swp10s0 ahead of swp1s0. Names in no recognized notation ("lo") keep
    their lexicographic place among the prefixes.
    """
    name = parse(port_id)
    if name is None:
        return (port_id, 0, 0, NO_SUBPORT)
    return name.sort_key


def cisco_port_id_sort_key(port_id: str) -> tuple[str, int, int, int]:
    """port_id_sort_key for Cisco NX-OS names: Ethernet1/2 before Ethernet1/10,
    and slot 2 after every port of slot 1. Other names ("mgmt0", "Vlan1") keep
    their lexicographic place among the prefixes.
    """
    name = parse_cisco(port_id)
    if name is None:
        return (port_id, 0, 0, NO_SUBPORT)
    return name.sort_key


def sort_ports(ports: list[Port]) -> list[Port]:
    return sorted(ports, key=lambda port: port_id_sort_key(port.id))
