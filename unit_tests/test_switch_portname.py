# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2025 Delos Data, Inc.

"""Port ID <-> interface name mapping.

Port IDs are a convention the services under test share, not one this library is
free to choose, so these cases pin the numbering itself: the order names sort in,
the IDs they are given, and what does and does not count as a breakout.
"""

import pytest

from production_test_framework.switch.portname import NO_SUBPORT, PortNameResolver, parse, parse_cisco


@pytest.mark.parametrize(
    ("interface_name", "expected"),
    [
        # Cumulus / Spectrum: plain cage and breakout subport.
        ("swp1", ("swp", 1, NO_SUBPORT)),
        ("swp64", ("swp", 64, NO_SUBPORT)),
        ("swp1s0", ("swp", 1, 0)),
        ("swp1s3", ("swp", 1, 3)),
        ("swp32s2", ("swp", 32, 2)),
        # Arista: "/" separator, 1-based subport.
        ("Ethernet49", ("Ethernet", 49, NO_SUBPORT)),
        ("Ethernet49/1", ("Ethernet", 49, 1)),
        ("Et49", ("Et", 49, NO_SUBPORT)),
        # Surrounding whitespace is tolerated; raw is the trimmed name.
        ("  swp7s1  ", ("swp", 7, 1)),
    ],
)
def test_parse(interface_name: str, expected: tuple[str, int, int]) -> None:
    name = parse(interface_name)
    assert name is not None
    assert (name.prefix, name.cage, name.subport) == expected
    assert name.raw == interface_name.strip()


@pytest.mark.parametrize(
    "interface_name",
    [
        "swp",  # no cage number
        "",
        "br_default",
        "lo",
        "swp1.100",  # dotted subinterface
        "swp1s0x",  # trailing junk
        "1swp",  # leading digit
    ],
)
def test_parse_rejects(interface_name: str) -> None:
    assert parse(interface_name) is None


@pytest.mark.parametrize("interface_name", ["eth0", "vlan101", "bond1"])
def test_parse_management_interfaces_match(interface_name: str) -> None:
    """Names like "eth0" DO parse: they are structurally identical to a data port
    name. Filtering them out is the vendor driver's job, since only the driver
    knows which prefix its data ports use.
    """
    name = parse(interface_name)
    assert name is not None
    assert not name.has_subport_suffix


@pytest.mark.parametrize(
    ("interface_name", "expected"),
    [
        ("swp3", False),
        ("swp3s0", True),
        ("Ethernet49", False),
        # A whole QSFP is still written "/1", so this is true while the cage is not
        # broken out.
        ("Ethernet49/1", True),
    ],
)
def test_has_subport_suffix(interface_name: str, expected: bool) -> None:
    """A property of the name alone: a suffix does not imply the cage is split.
    That question needs the whole name set - see has_breakout.
    """
    name = parse(interface_name)
    assert name is not None
    assert name.has_subport_suffix is expected


def test_learn_assigns_dense_ids_in_switch_port_order() -> None:
    port_names = PortNameResolver()
    # Deliberately unsorted input, and includes swp10 to catch lexical sorting
    # ("swp10" < "swp2" as strings, but cage 10 > cage 2).
    assert port_names.learn(["swp2", "swp1s1", "swp1s0", "swp10"]) == 4

    assert port_names.mapping() == "1=swp1s0 2=swp1s1 3=swp2 4=swp10"
    assert len(port_names) == 4
    assert port_names.discovered
    assert port_names.has_breakout


def test_learn_mixed_breakout_shifts_later_cages() -> None:
    port_names = PortNameResolver()
    # Cage 1 broken out 4 ways, cages 2 and 3 plain: every later cage shifts up.
    assert port_names.learn(["swp1s0", "swp1s1", "swp1s2", "swp1s3", "swp2", "swp3"]) == 6

    assert port_names.mapping() == "1=swp1s0 2=swp1s1 3=swp1s2 4=swp1s3 5=swp2 6=swp3"


def test_learn_no_breakout_is_identity() -> None:
    """The backward-compatibility guarantee: with no breakout, port ID N is swpN
    exactly as the old fixed format had it.
    """
    port_names = PortNameResolver()
    names = [f"swp{cage}" for cage in range(1, 33)]
    assert port_names.learn(names) == 32

    for cage in range(1, 33):
        assert port_names.name(cage) == f"swp{cage}"
        assert port_names.port_id(f"swp{cage}") == cage
    assert not port_names.has_breakout


def test_learn_real_breakout_switch_shifts_every_cage() -> None:
    """The 12-cage switch broken out two ways that this mapping was written for:
    every cage after the first shifts, and a subport with no LLDP neighbour or a
    down link still occupies its ID.
    """
    port_names = PortNameResolver()
    names = [f"swp{cage}s{subport}" for cage in range(1, 13) for subport in (0, 1)]
    assert port_names.learn(names) == 24

    assert port_names.port_id("swp1s0") == 1
    assert port_names.port_id("swp1s1") == 2
    assert port_names.port_id("swp2s0") == 3
    # swp7s1 and swp9s0/swp9s1 are down on that switch and still consume IDs.
    assert port_names.port_id("swp7s1") == 14
    assert port_names.port_id("swp8s0") == 15
    assert port_names.port_id("swp12s1") == 24
    assert port_names.has_breakout


@pytest.mark.parametrize(
    ("case", "interface_names", "expected"),
    [
        (
            "arista unbroken qsfp uses /1 naming",
            ["Ethernet1", "Ethernet48", "Ethernet49/1", "Ethernet50/1", "Ethernet54/1"],
            False,
        ),
        (
            "arista genuinely broken out qsfp",
            ["Ethernet48", "Ethernet49/1", "Ethernet49/2", "Ethernet49/3", "Ethernet49/4"],
            True,
        ),
        # The minimum evidence of an Arista split: "/2" exists next to "/1".
        ("arista 2x breakout", ["Ethernet49/1", "Ethernet49/2", "Ethernet50/1"], True),
        # Several whole QSFPs, each suffixed "/1" and none split.
        ("arista many whole qsfps", ["Ethernet49/1", "Ethernet50/1", "Ethernet51/1", "Ethernet52/1"], False),
        ("cumulus breakout", ["swp1s0", "swp1s1", "swp1s2", "swp1s3", "swp2"], True),
        ("cumulus no breakout", ["swp1", "swp2", "swp3"], False),
        ("lone cumulus subport without siblings", ["swp1s0", "swp2", "swp3"], False),
    ],
)
def test_has_breakout_requires_siblings(case: str, interface_names: list[str], expected: bool) -> None:
    """A subport suffix alone is not a breakout - only a cage holding more than one
    interface is. Observed on real hardware: an unbroken Arista QSFP is named
    "Ethernet49/1", so a suffix-only test reports a breakout on every port and warns
    about a port-ID shift that has not happened.
    """
    port_names = PortNameResolver()
    assert port_names.learn(interface_names) == len(interface_names)
    assert port_names.has_breakout is expected, case


def test_learn_round_trip() -> None:
    port_names = PortNameResolver()
    assert port_names.learn(["swp1s0", "swp1s1", "swp2"]) == 3

    for port_id in range(1, 4):
        name = port_names.name(port_id)
        assert name is not None
        assert port_names.port_id(name) == port_id


def test_learn_ignores_unparseable_and_duplicates() -> None:
    port_names = PortNameResolver()
    assert port_names.learn(["swp1", "br_default", "swp1", "lo", "swp2", "  swp2  "]) == 2
    assert port_names.mapping() == "1=swp1 2=swp2"


def test_learn_empty_keeps_existing_mapping() -> None:
    """A switch that answers with an empty or entirely unusable interface list must
    not discard a good mapping.
    """
    port_names = PortNameResolver()
    assert port_names.learn(["swp1s0", "swp1s1"]) == 2
    before = port_names.mapping()

    assert port_names.learn([]) == 0
    assert port_names.learn(["br_default", "lo"]) == 0

    assert port_names.mapping() == before
    assert port_names.discovered


def test_learn_replaces_previous_mapping() -> None:
    port_names = PortNameResolver()
    assert port_names.learn(["swp1", "swp2"]) == 2
    assert port_names.learn(["swp1s0", "swp1s1", "swp1s2"]) == 3

    assert port_names.mapping() == "1=swp1s0 2=swp1s1 3=swp1s2"
    # The stale name must no longer resolve.
    assert port_names.port_id("swp2") is None


def test_empty_resolver_reports_nothing() -> None:
    port_names = PortNameResolver()
    assert not port_names.discovered
    assert not port_names.has_breakout
    assert len(port_names) == 0
    assert port_names.mapping() == ""
    assert port_names.port_id("swp1") is None
    assert port_names.name(1) is None


@pytest.mark.parametrize(
    ("interface_name", "expected"),
    [
        # Cisco NX-OS: slot/port, plus /subport when broken out; "Eth" in some tables.
        ("Ethernet1/49", (1, 49, NO_SUBPORT)),
        ("Ethernet1/49/2", (1, 49, 2)),
        ("Eth1/1", (1, 1, NO_SUBPORT)),
        ("Ethernet2/3", (2, 3, NO_SUBPORT)),
    ],
)
def test_parse_cisco(interface_name: str, expected: tuple[int, int, int]) -> None:
    name = parse_cisco(interface_name)
    assert name is not None
    assert name.prefix == "Ethernet"
    assert (name.slot, name.cage, name.subport) == expected
    assert name.raw == interface_name


@pytest.mark.parametrize("interface_name", ["mgmt0", "Vlan1", "Ethernet49", "port-channel1", "sup-eth1(R)"])
def test_parse_cisco_rejects(interface_name: str) -> None:
    assert parse_cisco(interface_name) is None


def test_parse_reads_a_cisco_name_differently() -> None:
    """Why NX-OS needs its own parser: the default one reads "Ethernet1/49" the
    Arista way, as cage 1, subport 49."""
    name = parse("Ethernet1/49")
    assert name is not None
    assert (name.cage, name.subport) == (1, 49)


def test_learn_cisco_orders_by_slot_then_port() -> None:
    port_names = PortNameResolver(parser=parse_cisco)
    assert port_names.learn(["Ethernet2/1", "Ethernet1/10", "mgmt0", "Ethernet1/2", "Ethernet1/1"]) == 4
    assert port_names.mapping() == "1=Ethernet1/1 2=Ethernet1/2 3=Ethernet1/10 4=Ethernet2/1"
    assert not port_names.has_breakout


def test_learn_cisco_breakout_shifts_later_ports() -> None:
    port_names = PortNameResolver(parser=parse_cisco)
    port_names.learn(["Ethernet1/1/1", "Ethernet1/1/2", "Ethernet1/2"])
    assert port_names.mapping() == "1=Ethernet1/1/1 2=Ethernet1/1/2 3=Ethernet1/2"
    assert port_names.has_breakout
