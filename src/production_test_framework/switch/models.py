# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2025 Delos Data, Inc.

from dataclasses import dataclass, field
from enum import StrEnum
from ipaddress import IPv4Interface, IPv6Interface

# An SVI address: an IP address with its prefix length, e.g. 10.0.0.1/24.
SviAddress = IPv4Interface | IPv6Interface


@dataclass(frozen=True)
class NetworkSwitchConfig:
    host: str
    username: str
    password: str
    port: int = 443
    verify_tls: bool = True
    bridge_domain: str = "br_default"


@dataclass(frozen=True)
class Port:
    id: str
    admin_up: bool | None = None
    oper_up: bool | None = None
    description: str | None = None


@dataclass(frozen=True)
class Vlan:
    id: str
    ports: tuple[Port, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class LldpNeighbor:
    """An LLDP neighbor advertising a MAC-address chassis id on a switch port."""

    interface: str
    switch_port: int
    chassis_mac: str


@dataclass(frozen=True)
class MacEntry:
    """A MAC address-table (FDB) entry: a MAC reachable on a switch port in a VLAN."""

    mac: str
    port: str
    vlan: int | None = None
    static: bool = False  # True = statically configured, False = dynamically learned


@dataclass(frozen=True)
class PortBridge:
    """A port's bridging configuration. A field is None when the switch does not report it,
    for example on a port that is not bridged."""

    interface: str
    mode: str | None = None  # "trunk" or "access"; Arista may also report e.g. "dot1qTunnel" or "tap"
    native_vlan: int | None = None  # PVID: the VLAN untagged ingress frames are placed in
    learning: bool | None = None  # True = the port learns source MACs into the FDB


class FecEncoding(StrEnum):
    """A port's forward error correction. Both ends of a link must use the same one."""

    OFF = "off"
    FIRE_CODE = "fire-code"  # BASE-R, IEEE 802.3 Clause 74
    REED_SOLOMON = "reed-solomon"  # RS(528) or RS(544), as the port speed needs


@dataclass(frozen=True)
class FlowControl:
    """The switch-wide link-level flow control configuration."""

    roce_lossless: bool  # True = RoCE is on in lossless mode, the profile that carries PFC
    pfc_profile: bool  # True = a switch-wide PFC profile is configured


@dataclass(frozen=True)
class SwitchProcess:
    name: str
    pid: int
    status: str
    start_time: str
    end_time: str
    cpu_usage: float
    memory_usage: float


@dataclass(frozen=True)
class NetworkSwitchStatus:
    uptime: str
    hostname: str
    asic_model: str
    model: str
    serial_number: str
    firmware_version: str
    software_version: str
