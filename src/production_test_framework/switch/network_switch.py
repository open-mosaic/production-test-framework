# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2025 Delos Data, Inc.

from abc import ABC, abstractmethod

from production_test_framework.switch.models import (
    LldpNeighbor,
    MacEntry,
    NetworkSwitchConfig,
    NetworkSwitchStatus,
    Port,
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

    @abstractmethod
    def port(self, port_id: str) -> Port:
        """Get configuration for a port of the switch."""
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
