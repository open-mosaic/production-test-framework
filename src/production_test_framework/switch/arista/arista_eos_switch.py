# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2026 Delos Data, Inc.

"""Arista EOS switch driver via eAPI (pyeapi).

Only the commands backing the ``NetworkSwitch`` interface are wrapped:

    status                -> show version, show hostname
    ports / port          -> show interfaces status
    vlans / vlan          -> show vlan  (+ show interfaces status for members)
    lldp_neighbors        -> show lldp neighbors detail
    port_bridge           -> show interfaces <id> switchport
    flow_control          -> not supported yet, raises NotImplementedError
    set_port_admin_state  -> interface <id> ; [no] shutdown
    delete_vlan           -> no vlan <id>
    port_fec              -> show running-config interfaces <id>
    set_port_fec          -> interface <id> ; [no|default] error-correction encoding ...
    svi_address           -> show running-config interfaces Vlan<id>
    set_svi_address       -> interface Vlan<id> ; ip[v6] address ... / no interface Vlan<id>

Requires eAPI enabled on the switch (``management api http-commands``).
"""

import logging
import ssl
import time
from typing import Any

import pyeapi
from pyeapi.eapilib import CommandError
from pyeapi.eapilib import ConnectionError as EapiConnectionError

from production_test_framework.switch.exceptions import SwitchAPIError
from production_test_framework.switch.mac import normalize_mac
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
from production_test_framework.switch.network_switch import (
    NetworkSwitch,
    parse_svi_address,
    svi_address_from_config,
)
from production_test_framework.switch.port_sort import port_id_sort_key
from production_test_framework.switch.portname import PortNameResolver

# EOS names data ports "Ethernet<cage>" (abbreviated "Et<cage>"), plus a subport index
# when the cage is broken out. The interface status table also reports
# management, port-channel and VLAN interfaces, which are not data ports.
_ETHERNET_PREFIXES = ("Ethernet", "Et")


class AristaEosSwitch(NetworkSwitch):
    """eAPI client for Arista EOS switches."""

    def __init__(self, config: NetworkSwitchConfig) -> None:
        super().__init__(config)
        self._logger = logging.getLogger(__name__)
        self._logger.debug(f"Initializing AristaEosSwitch with config: {config}")
        # Port 80 is cleartext eAPI; anything else is TLS.
        if config.port == 80:
            connection = pyeapi.connect(
                transport="http",
                host=config.host,
                username=config.username,
                password=config.password,
                port=config.port,
            )
        else:
            # Explicit context so config.verify_tls is honoured (pyeapi defaults to unverified).
            if config.verify_tls:
                context = ssl.create_default_context()
            else:
                context = ssl.create_default_context()
                context.check_hostname = False
                context.verify_mode = ssl.CERT_NONE
            connection = pyeapi.connect(
                transport="https",
                host=config.host,
                username=config.username,
                password=config.password,
                port=config.port,
                context=context,
            )
        self._node = pyeapi.client.Node(connection)

    @property
    def status(self) -> NetworkSwitchStatus:
        version = self._run_show("show version")
        hostname = self._run_show("show hostname")
        return self._parse_system_status(version, hostname)

    @property
    def ports(self) -> list[Port]:
        statuses = self._interface_statuses()
        return self._parse_ports(statuses)

    @property
    def vlans(self) -> list[Vlan]:
        vlan_configs = self._run_show("show vlan").get("vlans", {})
        statuses = self._interface_statuses()
        return self._parse_vlans(vlan_configs, statuses)

    @property
    def lldp_neighbors(self) -> list[LldpNeighbor]:
        """LLDP neighbors whose advertised port id is a MAC (eAPI: show lldp neighbors detail)."""
        payload = self._run_show("show lldp neighbors detail").get("lldpNeighbors", {})
        # "show lldp neighbors detail" reports only the interfaces that have a
        # neighbour, so port IDs are learned from the full interface status table.
        return self._parse_lldp_neighbors(payload, self._port_names(self._interface_statuses()))

    @property
    def mac_table(self) -> list[MacEntry]:
        """MAC address-table entries (eAPI: show mac address-table)."""
        entries = self._run_show("show mac address-table").get("unicastTable", {}).get("tableEntries", [])
        return self._parse_mac_table(entries)

    def port(self, port_id: str) -> Port:
        statuses = self._run_show(f"show interfaces {port_id} status").get("interfaceStatuses", {})
        body = statuses.get(port_id)
        if not isinstance(body, dict):
            raise SwitchAPIError(f"port {port_id} not found")
        return self._parse_port(port_id, body)

    def port_bridge(self, port_id: str) -> PortBridge:
        """A port's switchport configuration (eAPI: show interfaces <id> switchport)."""
        switchports = self._run_show(f"show interfaces {port_id} switchport").get("switchports", {})
        body = switchports.get(port_id)
        if not isinstance(body, dict):
            raise SwitchAPIError(f"port {port_id} not found")
        return self._parse_port_bridge(port_id, body)

    @property
    def flow_control(self) -> FlowControl:
        raise NotImplementedError("flow control is not supported on Arista EOS yet")

    def vlan(self, vlan_id: str) -> Vlan:
        try:
            vlan_configs = self._run_show(f"show vlan {vlan_id}").get("vlans", {})
        except SwitchAPIError as exc:
            # EOS returns a command error for an unknown VLAN id.
            raise SwitchAPIError(f"VLAN {vlan_id} not found") from exc
        if vlan_id not in vlan_configs:
            raise SwitchAPIError(f"VLAN {vlan_id} not found")
        statuses = self._interface_statuses()
        return self._parse_vlan(vlan_id, vlan_configs[vlan_id], statuses)

    def set_port_admin_state(self, port_id: str, up: bool) -> None:
        """Administratively enable (up=True) or disable (up=False) a port."""
        command = "no shutdown" if up else "shutdown"
        self._run_config([f"interface {port_id}", command])

    def delete_vlan(self, vlan_id: str) -> None:
        """Remove a VLAN from the switch."""
        self._run_config([f"no vlan {vlan_id}"])

    def port_fec(self, port_id: str) -> FecEncoding | None:
        """The port's FEC from its running config; no error-correction line is the switch's default."""
        encodings = set()
        for line in self._run_text(f"show running-config interfaces {port_id}").splitlines():
            match line.strip():
                case "no error-correction encoding":
                    encodings.add(FecEncoding.OFF)
                case "error-correction encoding fire-code":
                    encodings.add(FecEncoding.FIRE_CODE)
                case "error-correction encoding reed-solomon":
                    encodings.add(FecEncoding.REED_SOLOMON)
        # Both encodings configured leaves the choice to the switch, as none does.
        return encodings.pop() if len(encodings) == 1 else None

    def set_port_fec(self, port_id: str, fec: FecEncoding | None) -> None:
        """Set the port's FEC. EOS adds an encoding to those the port has, so setting one removes the other."""
        match fec:
            case None:
                commands = ["default error-correction encoding"]
            case FecEncoding.OFF:
                commands = ["no error-correction encoding"]
            case FecEncoding.FIRE_CODE:
                commands = ["error-correction encoding fire-code", "no error-correction encoding reed-solomon"]
            case FecEncoding.REED_SOLOMON:
                commands = ["error-correction encoding reed-solomon", "no error-correction encoding fire-code"]
            case _:
                raise ValueError(f"unsupported FEC {fec!r}")
        self._run_config([f"interface {port_id}", *commands])

    def svi_address(self, vlan_id: str) -> SviAddress | None:
        """The SVI's address from its running config, which is empty when there is no SVI."""
        return svi_address_from_config(self._run_text(f"show running-config interfaces Vlan{vlan_id}"))

    def set_svi_address(self, vlan_id: str, address: SviAddress | str | None) -> None:
        """Make address the SVI's only address, or remove the SVI for None."""
        if address is None:
            self._run_config([f"no interface Vlan{vlan_id}"])
            return
        address = parse_svi_address(address)
        family = "ip" if address.version == 4 else "ipv6"
        self._run_config(
            [
                f"interface Vlan{vlan_id}",
                "no ip address",
                "no ipv6 address",
                f"{family} address {address.with_prefixlen}",
            ]
        )

    # --- eAPI plumbing -------------------------------------------------------

    def _run_show(self, command: str) -> dict[str, Any]:
        self._logger.debug(f"eAPI show: {command}")
        try:
            results = self._node.run_commands([command])
        except (CommandError, EapiConnectionError) as exc:
            self._logger.error(f"eAPI command failed: {command}: {exc}")
            raise SwitchAPIError(f"eAPI command failed: {command}: {exc}") from exc
        if not results or not isinstance(results[0], dict):
            raise SwitchAPIError(f"unexpected eAPI response for {command!r}: {results!r}")
        self._logger.debug(f"eAPI response: {results[0]}")
        return results[0]

    def _run_text(self, command: str) -> str:
        """A show command's text output, for what eAPI has no JSON form of (the running config)."""
        self._logger.debug(f"eAPI show (text): {command}")
        try:
            results = self._node.run_commands([command], encoding="text")
        except (CommandError, EapiConnectionError) as exc:
            self._logger.error(f"eAPI command failed: {command}: {exc}")
            raise SwitchAPIError(f"eAPI command failed: {command}: {exc}") from exc
        output = results[0].get("output") if results and isinstance(results[0], dict) else None
        if not isinstance(output, str):
            raise SwitchAPIError(f"unexpected eAPI response for {command!r}: {results!r}")
        return output

    def _run_config(self, commands: list[str]) -> None:
        self._logger.debug(f"eAPI config: {commands}")
        try:
            self._node.config(commands)
        except (CommandError, EapiConnectionError) as exc:
            self._logger.error(f"eAPI config failed: {commands}: {exc}")
            raise SwitchAPIError(f"eAPI config failed: {commands}: {exc}") from exc

    def _interface_statuses(self) -> dict[str, Any]:
        return self._run_show("show interfaces status").get("interfaceStatuses", {})

    # --- parsing -------------------------------------------------------------

    @staticmethod
    def _admin_up(link_status: Any) -> bool | None:
        if not isinstance(link_status, str):
            return None
        # linkStatus is one of: connected / notconnect / disabled / errdisabled
        return link_status != "disabled"

    @staticmethod
    def _oper_up(body: dict[str, Any]) -> bool | None:
        line_proto = body.get("lineProtocolStatus")
        if isinstance(line_proto, str):
            return line_proto == "up"
        link_status = body.get("linkStatus")
        if isinstance(link_status, str):
            return link_status == "connected"
        return None

    def _parse_port(self, port_id: str, body: dict[str, Any]) -> Port:
        description = body.get("description")
        if description == "":
            description = None
        return Port(
            id=port_id,
            admin_up=self._admin_up(body.get("linkStatus")),
            oper_up=self._oper_up(body),
            description=description if isinstance(description, str) else None,
        )

    def _parse_ports(self, statuses: dict[str, Any]) -> list[Port]:
        ports = [self._parse_port(port_id, body) for port_id, body in statuses.items() if isinstance(body, dict)]
        return sorted(ports, key=lambda port: port_id_sort_key(port.id))

    def _member_ports(self, member_ids: list[str], statuses: dict[str, Any]) -> tuple[Port, ...]:
        ordered = sorted(member_ids, key=port_id_sort_key)
        return tuple(self._parse_port(pid, statuses.get(pid, {})) for pid in ordered)

    def _parse_vlan(self, vlan_id: str, body: dict[str, Any], statuses: dict[str, Any]) -> Vlan:
        interfaces = body.get("interfaces", {})
        member_ids = list(interfaces.keys()) if isinstance(interfaces, dict) else []
        return Vlan(id=vlan_id, ports=self._member_ports(member_ids, statuses))

    def _parse_vlans(self, vlan_configs: dict[str, Any], statuses: dict[str, Any]) -> list[Vlan]:
        vlans = [self._parse_vlan(vid, body, statuses) for vid, body in vlan_configs.items()]
        return sorted(vlans, key=lambda vlan: int(vlan.id) if vlan.id.isdigit() else vlan.id)

    def _parse_lldp_neighbors(
        self,
        lldp_neighbors: dict[str, Any],
        port_names: PortNameResolver,
    ) -> list[LldpNeighbor]:
        neighbors: list[LldpNeighbor] = []
        for interface_id, body in lldp_neighbors.items():
            switch_port = port_names.port_id(interface_id)
            if switch_port is None or not isinstance(body, dict):
                continue
            for entry in body.get("lldpNeighborInfo", []):
                if not isinstance(entry, dict):
                    continue
                # chassis_mac carries the neighbor's port/NIC MAC, as the NVIDIA driver does.
                info = entry.get("neighborInterfaceInfo", {})
                if not isinstance(info, dict):
                    continue
                port_id = info.get("interfaceId", "")
                if not isinstance(port_id, str):
                    continue
                mac = normalize_mac(port_id.replace('"', ""))
                if not mac:
                    # A non-MAC port id (e.g. an interface name) is expected; skip it.
                    if info.get("interfaceIdType") == "macAddress":
                        self._logger.warning(f"lldp_neighbors: invalid MAC {port_id!r} on {interface_id}")
                    continue
                neighbors.append(LldpNeighbor(interface=interface_id, switch_port=switch_port, chassis_mac=mac))
        return sorted(neighbors, key=lambda neighbor: neighbor.switch_port)

    @staticmethod
    def _parse_port_bridge(port_id: str, body: dict[str, Any]) -> PortBridge:
        info = body.get("switchportInfo")
        if not body.get("enabled") or not isinstance(info, dict):  # a routed port does not bridge
            return PortBridge(interface=port_id)
        mode = info.get("mode")
        native_vlan = info.get("accessVlanId" if mode == "access" else "trunkingNativeVlanId")
        learning = info.get("macLearning")
        return PortBridge(
            interface=port_id,
            mode=mode if isinstance(mode, str) else None,
            native_vlan=native_vlan if isinstance(native_vlan, int) else None,
            learning=learning if isinstance(learning, bool) else None,
        )

    def _parse_mac_table(self, entries: list[dict[str, Any]]) -> list[MacEntry]:
        table: list[MacEntry] = []
        for entry in entries:
            interface = entry.get("interface", "")
            mac = normalize_mac(entry.get("macAddress", ""))
            if not interface or not mac:
                continue
            table.append(
                MacEntry(mac=mac, port=interface, vlan=entry.get("vlanId"), static=entry.get("entryType") == "static")
            )
        return table

    def _port_names(self, statuses: dict[str, Any]) -> PortNameResolver:
        """Learn the port ID of every data port from the switch's interface list.

        A port ID is an interface's position in that list, not its cage number, so
        it can only be learned from the whole list: a broken-out cage contributes
        one ID per subport and shifts every later cage up. "show interfaces status"
        is that list; only the Ethernet entries are data ports.
        """
        port_names = PortNameResolver()
        port_names.learn(name for name in statuses if name.startswith(_ETHERNET_PREFIXES))
        if port_names.has_breakout:
            self._logger.debug(f"port names: breakout shifts port ids from cage numbers: {port_names.mapping()}")
        return port_names

    def _parse_system_status(self, version: dict[str, Any], hostname: dict[str, Any]) -> NetworkSwitchStatus:
        uptime = version.get("uptime")
        if uptime is None and "bootupTimestamp" in version:
            # bootupTimestamp is epoch seconds; derive uptime in seconds.
            uptime = time.time() - version["bootupTimestamp"]
        return NetworkSwitchStatus(
            uptime=uptime,
            hostname=hostname.get("hostname") or hostname.get("fqdn"),
            model=version.get("modelName"),
            serial_number=version.get("serialNumber"),
            firmware_version=version.get("version"),
            # FLAG: no uniform eAPI field for ASIC model. Left None; may require
            # NetworkSwitchStatus.asic_model to be Optional, or a platform-family
            # lookup (`show platform ...`). See summary.
            asic_model=None,
            software_version=version.get("version"),
        )
