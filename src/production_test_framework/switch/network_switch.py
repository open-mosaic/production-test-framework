# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2025 Delos Data, Inc.

from abc import ABC, abstractmethod
from ipaddress import ip_interface

from production_test_framework.switch.models import (
    FecEncoding,
    FlowControl,
    LldpNeighbor,
    MacEntry,
    NetworkSwitchConfig,
    NetworkSwitchStatus,
    Port,
    PortBridge,
    SviAddress,
    Vlan,
)


class NetworkSwitch(ABC):
    def __init__(self, config: NetworkSwitchConfig) -> None:
        self._config = config

    @property
    @abstractmethod
    def status(self) -> NetworkSwitchStatus:
        """Get the switch status"""
        ...

    @property
    @abstractmethod
    def ports(self) -> list[Port]:
        """Get the ports of the switch"""
        ...

    @property
    @abstractmethod
    def vlans(self) -> list[Vlan]:
        """Get the vlans of the switch"""
        ...

    @property
    @abstractmethod
    def lldp_neighbors(self) -> list[LldpNeighbor]:
        """Get the LLDP neighbors advertising a MAC chassis id, one per switch port."""
        ...

    @property
    @abstractmethod
    def mac_table(self) -> list[MacEntry]:
        """Get the switch MAC address-table (FDB) entries."""
        ...

    @property
    def mac_table_by_interface(self) -> dict[str, set[str]]:
        """The MAC table indexed by interface: {interface: {mac, ...}}."""
        table: dict[str, set[str]] = {}
        for entry in self.mac_table:
            table.setdefault(entry.port, set()).add(entry.mac)
        return table

    @property
    @abstractmethod
    def flow_control(self) -> FlowControl:
        """Get the switch-wide link-level flow control configuration.

        Raises NotImplementedError on a switch the driver cannot read it from.
        """
        ...

    @abstractmethod
    def port(self, port_id: str) -> Port:
        """Get configuration for a port of the switch."""
        ...

    @abstractmethod
    def port_bridge(self, port_id: str) -> PortBridge:
        """Get the bridging configuration of a port: mode, native VLAN and MAC learning."""
        ...

    @abstractmethod
    def vlan(self, vlan_id: str) -> Vlan:
        """Get configuration for a vlan of the switch."""
        ...

    @abstractmethod
    def set_port_admin_state(self, port_id: str, up: bool) -> None:
        """Administratively enable (up=True) or disable (up=False) a port."""
        ...

    @abstractmethod
    def delete_vlan(self, vlan_id: str) -> None:
        """Remove a VLAN from the bridge domain and from every member interface."""
        ...

    @abstractmethod
    def port_fec(self, port_id: str) -> FecEncoding | None:
        """The port's FEC encoding; None when the switch picks its own (its default, auto)."""
        ...

    @abstractmethod
    def set_port_fec(self, port_id: str, fec: FecEncoding | None) -> None:
        """Set the port's FEC encoding; None returns it to the switch's default."""
        ...

    @abstractmethod
    def svi_address(self, vlan_id: str) -> SviAddress | None:
        """The address of the VLAN's SVI, the switch's own L3 interface on the VLAN; None without one."""
        ...

    @abstractmethod
    def set_svi_address(self, vlan_id: str, address: SviAddress | str | None) -> None:
        """Make address the SVI's only address, creating the SVI if needed; None removes the SVI.

        A string must carry its prefix length, e.g. "fd00:dec0::1/64".
        """
        ...


def parse_svi_address(address: SviAddress | str) -> SviAddress:
    """An SVI address from a string such as "10.0.0.1/24", which must carry its prefix length."""
    if isinstance(address, str):
        if "/" not in address:
            raise ValueError(f"SVI address {address!r} has no prefix length")
        return ip_interface(address)
    return address


def svi_address_from_config(running_config: str) -> SviAddress | None:
    """The SVI's primary address from its CLI running config (EOS and NX-OS alike).

    Secondary and link-local addresses carry a fourth field and are left out, and so
    is a line that names no address, such as "ip address dhcp".
    """
    for line in running_config.splitlines():
        fields = line.split()
        if len(fields) == 3 and fields[0] in ("ip", "ipv6") and fields[1] == "address":
            try:
                return ip_interface(fields[2])
            except ValueError:
                continue
    return None
