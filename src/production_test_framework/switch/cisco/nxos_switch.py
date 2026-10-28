# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2026 Delos Data, Inc.

"""Cisco NX-OS switch driver via NX-API (JSON-RPC over HTTPS).

Only the commands backing the ``NetworkSwitch`` interface are wrapped:

    status                -> show version, show hostname
    ports / port          -> show interface [<id>]
    vlans / vlan          -> show vlan brief / show vlan id <id> (+ show interface for members)
    lldp_neighbors        -> show lldp neighbors detail (+ show interface for port IDs)
    mac_table             -> show mac address-table
    port_bridge           -> show interface <id> switchport
    flow_control          -> not supported yet, raises NotImplementedError
    set_port_admin_state  -> interface <id> ; [no] shutdown
    delete_vlan           -> no vlan <id>
    port_fec              -> show running-config interface <id>
    set_port_fec          -> interface <id> ; fec <mode> / no fec
    svi_address           -> show running-config interface vlan<id>
    set_svi_address       -> interface vlan<id> ; ip[v6] address ... / no interface vlan<id>

Requires NX-API enabled on the switch (``feature nxapi``), ``feature lldp`` for
lldp_neighbors and ``feature interface-vlan`` for SVIs.
"""

import logging
from typing import Any

import requests
from requests.auth import HTTPBasicAuth

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
from production_test_framework.switch.port_sort import cisco_port_id_sort_key
from production_test_framework.switch.portname import PortNameResolver, parse_cisco

requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)

_TIMEOUT_S = 60

# NX-OS's FEC modes as its running config shows them. The CLI takes "rs-fec" for
# Reed-Solomon at any speed; "kp-fec" (RS(544)) is a DME value the CLI refuses.
_FEC_MODES = {
    "off": FecEncoding.OFF,
    "fc-fec": FecEncoding.FIRE_CODE,
    "rs-fec": FecEncoding.REED_SOLOMON,
    "kp-fec": FecEncoding.REED_SOLOMON,
    "rs-ieee": FecEncoding.REED_SOLOMON,
    "rs-cons16": FecEncoding.REED_SOLOMON,
}


class CiscoNxosSwitch(NetworkSwitch):
    """NX-API client for Cisco NX-OS switches."""

    def __init__(self, config: NetworkSwitchConfig) -> None:
        super().__init__(config)
        self._logger = logging.getLogger(__name__)
        self._logger.debug(f"Initializing CiscoNxosSwitch with config: {config}")
        self._url = f"https://{config.host}:{config.port}/ins"
        self._session = requests.Session()

    @property
    def status(self) -> NetworkSwitchStatus:
        version = self._run_show("show version")
        hostname = self._run_show("show hostname")
        return self._parse_system_status(version, hostname)

    @property
    def ports(self) -> list[Port]:
        return self._parse_ports(self._interfaces())

    @property
    def vlans(self) -> list[Vlan]:
        rows = _rows(self._run_show("show vlan brief"), "TABLE_vlanbriefxbrief", "ROW_vlanbriefxbrief")
        ports = {port.id: port for port in self._parse_ports(self._interfaces())}
        vlans = [self._parse_vlan(row, ports) for row in rows]
        return sorted(vlans, key=lambda vlan: int(vlan.id) if vlan.id.isdigit() else vlan.id)

    @property
    def lldp_neighbors(self) -> list[LldpNeighbor]:
        """LLDP neighbors whose advertised port id is a MAC (NX-API: show lldp neighbors detail)."""
        body = self._run_show("show lldp neighbors detail")
        rows = _rows(body, "TABLE_nbor_detail", "ROW_nbor_detail")
        # The neighbor table only lists ports that have a neighbor, so port IDs are
        # learned from the full interface list.
        return self._parse_lldp_neighbors(rows, self._port_names(self._interfaces()))

    @property
    def mac_table(self) -> list[MacEntry]:
        """MAC address-table entries (NX-API: show mac address-table)."""
        rows = _rows(self._run_show("show mac address-table"), "TABLE_mac_address", "ROW_mac_address")
        return self._parse_mac_table(rows)

    @property
    def flow_control(self) -> FlowControl:
        raise NotImplementedError("flow control is not supported on Cisco NX-OS yet")

    def port(self, port_id: str) -> Port:
        rows = _rows(self._run_show(f"show interface {port_id}"), "TABLE_interface", "ROW_interface")
        if not rows:
            raise SwitchAPIError(f"port {port_id} not found")
        return self._parse_port(rows[0])

    def port_bridge(self, port_id: str) -> PortBridge:
        """A port's switchport configuration (NX-API: show interface <id> switchport)."""
        rows = _rows(self._run_show(f"show interface {port_id} switchport"), "TABLE_interface", "ROW_interface")
        if not rows:
            raise SwitchAPIError(f"port {port_id} not found")
        return self._parse_port_bridge(port_id, rows[0])

    def vlan(self, vlan_id: str) -> Vlan:
        try:
            body = self._run_show(f"show vlan id {vlan_id}")
        except SwitchAPIError as exc:
            # NX-OS returns a command error for an out-of-range or reserved VLAN id. An
            # error without an HTTP status never reached the switch: report it as it is.
            if exc.status_code is None:
                raise
            raise SwitchAPIError(f"VLAN {vlan_id} not found", exc.status_code) from exc
        rows = _rows(body, "TABLE_vlanbriefid", "ROW_vlanbriefid")
        if not rows:
            raise SwitchAPIError(f"VLAN {vlan_id} not found")
        ports = {port.id: port for port in self._parse_ports(self._interfaces())}
        return self._parse_vlan(rows[0], ports)

    def set_port_admin_state(self, port_id: str, up: bool) -> None:
        """Administratively enable (up=True) or disable (up=False) a port."""
        self._run_config([f"interface {port_id}", "no shutdown" if up else "shutdown"])

    def delete_vlan(self, vlan_id: str) -> None:
        """Remove a VLAN from the switch."""
        self._run_config([f"no vlan {vlan_id}"])

    def port_fec(self, port_id: str) -> FecEncoding | None:
        """The port's FEC from its running config; no fec line (or fec auto) is the switch's choice."""
        for line in self._run_text(f"show running-config interface {port_id}").splitlines():
            fields = line.split()
            if len(fields) == 2 and fields[0] == "fec":
                if fields[1] == "auto":
                    return None
                if fields[1] not in _FEC_MODES:
                    raise SwitchAPIError(f"port {port_id}: unknown FEC mode {fields[1]!r}")
                return _FEC_MODES[fields[1]]
        return None

    def set_port_fec(self, port_id: str, fec: FecEncoding | None) -> None:
        """Set the port's FEC; None returns it to auto, the switch's choice."""
        match fec:
            case None:
                command = "no fec"
            case FecEncoding.OFF:
                command = "fec off"
            case FecEncoding.FIRE_CODE:
                command = "fec fc-fec"
            case FecEncoding.REED_SOLOMON:
                command = "fec rs-fec"
            case _:
                raise ValueError(f"unsupported FEC {fec!r}")
        self._run_config([f"interface {port_id}", command])

    def svi_address(self, vlan_id: str) -> SviAddress | None:
        """The SVI's address from its running config. NX-OS refuses to show an SVI that doesn't exist."""
        try:
            config = self._run_text(f"show running-config interface vlan{vlan_id}")
        except SwitchAPIError as exc:
            if "Invalid range" in str(exc):
                return None
            raise
        return svi_address_from_config(config)

    def set_svi_address(self, vlan_id: str, address: SviAddress | str | None) -> None:
        """Make address the SVI's only address, or remove the SVI for None."""
        if address is None:
            self._run_config([f"no interface vlan{vlan_id}"])
            return
        address = parse_svi_address(address)
        family = "ip" if address.version == 4 else "ipv6"
        self._run_config(
            [
                f"interface vlan{vlan_id}",
                "no ip address",
                "no ipv6 address",
                f"{family} address {address.with_prefixlen}",
            ]
        )

    # --- NX-API plumbing -----------------------------------------------------

    def _call(self, command: str, method: str = "cli") -> dict[str, Any]:
        """One JSON-RPC request: "cli" for structured output, "cli_ascii" for text.
        Several commands go in one request separated by " ; ", and the switch runs
        them in order."""
        payload = {"jsonrpc": "2.0", "method": method, "params": {"cmd": command, "version": 1}, "id": 1}
        self._logger.debug(f"NX-API: {command}")
        try:
            response = self._session.post(
                self._url,
                json=payload,
                headers={"Content-Type": "application/json-rpc"},
                auth=HTTPBasicAuth(self._config.username, self._config.password),
                verify=self._config.verify_tls,
                timeout=_TIMEOUT_S,
            )
        except requests.RequestException as exc:
            raise SwitchAPIError(f"NX-API request failed: {command}: {exc}") from exc
        try:
            reply = response.json()
        except ValueError as exc:
            raise SwitchAPIError(f"NX-API {command!r}: HTTP {response.status_code}: {response.text[:200]}") from exc
        error = reply.get("error") if isinstance(reply, dict) else None
        if error or response.status_code != 200:
            data = error.get("data") if isinstance(error, dict) else None
            msg = data.get("msg") if isinstance(data, dict) else None
            summary = error.get("message") if isinstance(error, dict) else None
            message = (
                (msg.strip() if isinstance(msg, str) else "")
                or (summary if isinstance(summary, str) else "")
                or response.text[:200]
            )
            self._logger.error(f"NX-API command failed: {command}: {message}")
            raise SwitchAPIError(f"NX-API command failed: {command}: {message}", response.status_code)
        self._logger.debug(f"NX-API response: {reply}")
        return reply

    def _run_show(self, command: str) -> dict[str, Any]:
        """A show command's body; empty when the switch has nothing to show."""
        result = self._call(command).get("result")
        body = result.get("body") if isinstance(result, dict) else None
        return body if isinstance(body, dict) else {}

    def _run_text(self, command: str) -> str:
        """A show command's text, for what NX-API has no structured form of (the running config)."""
        result = self._call(command, method="cli_ascii").get("result")
        msg = result.get("msg") if isinstance(result, dict) else None
        if not isinstance(msg, str):
            raise SwitchAPIError(f"unexpected NX-API response for {command!r}: {result!r}")
        return msg

    def _run_config(self, commands: list[str]) -> None:
        self._call(" ; ".join(["configure terminal", *commands]))

    def _interfaces(self) -> list[dict[str, Any]]:
        return _rows(self._run_show("show interface"), "TABLE_interface", "ROW_interface")

    # --- parsing -------------------------------------------------------------

    @staticmethod
    def _parse_port(row: dict[str, Any]) -> Port:
        admin = row.get("admin_state")
        oper = row.get("state")
        description = row.get("desc")
        return Port(
            id=row.get("interface", ""),
            admin_up=admin == "up" if isinstance(admin, str) else None,
            oper_up=oper == "up" if isinstance(oper, str) else None,
            description=description if isinstance(description, str) and description else None,
        )

    def _parse_ports(self, rows: list[dict[str, Any]]) -> list[Port]:
        ports = [self._parse_port(row) for row in rows if row.get("interface")]
        return sorted(ports, key=lambda port: cisco_port_id_sort_key(port.id))

    @staticmethod
    def _parse_vlan(row: dict[str, Any], ports: dict[str, Port]) -> Vlan:
        vlan_id = str(row.get("vlanshowbr-vlanid", ""))
        members = row.get("vlanshowplist-ifidx", "")
        member_ids = [name.strip() for name in members.split(",") if name.strip()] if isinstance(members, str) else []
        ordered = sorted((_full_name(name) for name in member_ids), key=cisco_port_id_sort_key)
        return Vlan(id=vlan_id, ports=tuple(ports.get(name, Port(id=name)) for name in ordered))

    def _parse_lldp_neighbors(self, rows: list[dict[str, Any]], port_names: PortNameResolver) -> list[LldpNeighbor]:
        neighbors: list[LldpNeighbor] = []
        for row in rows:
            interface = _full_name(row.get("l_port_id", ""))
            switch_port = port_names.port_id(interface)
            if switch_port is None:
                continue  # mgmt0 or another non-data port
            port_id = row.get("port_id", "")
            mac = normalize_mac(port_id) if isinstance(port_id, str) else ""
            if not mac:
                # A non-MAC port id (e.g. an interface name) is expected; skip it.
                if row.get("port_type") == "Mac Address":
                    self._logger.warning(f"lldp_neighbors: invalid MAC {port_id!r} on {interface}")
                continue
            # chassis_mac carries the neighbor's port/NIC MAC, as the other drivers do.
            neighbors.append(LldpNeighbor(interface=interface, switch_port=switch_port, chassis_mac=mac))
        return sorted(neighbors, key=lambda neighbor: neighbor.switch_port)

    @staticmethod
    def _parse_mac_table(rows: list[dict[str, Any]]) -> list[MacEntry]:
        table: list[MacEntry] = []
        for row in rows:
            interface = _full_name(row.get("disp_port", ""))
            mac = normalize_mac(row.get("disp_mac_addr", ""))
            if not interface or interface.startswith("sup-") or interface.endswith("(R)") or not mac:
                continue  # the supervisor's own and routed entries ("sup-eth1(R)") are not on a port
            vlan = row.get("disp_vlan")
            table.append(
                MacEntry(
                    mac=mac,
                    port=interface,
                    vlan=int(vlan) if isinstance(vlan, str) and vlan.isdigit() else None,
                    static=row.get("disp_type") == "static",
                )
            )
        return table

    @staticmethod
    def _parse_port_bridge(port_id: str, row: dict[str, Any]) -> PortBridge:
        if row.get("switchport") != "Enabled":  # a routed port does not bridge
            return PortBridge(interface=port_id)
        mode = row.get("oper_mode")
        native_vlan = row.get("access_vlan" if mode == "access" else "native_vlan")
        learning = row.get("mac_learning")
        return PortBridge(
            interface=port_id,
            mode=mode if isinstance(mode, str) else None,
            native_vlan=native_vlan if isinstance(native_vlan, int) else None,
            learning=learning == "Enabled" if learning in ("Enabled", "Disabled") else None,
        )

    def _port_names(self, rows: list[dict[str, Any]]) -> PortNameResolver:
        """Learn the port ID of every data port from the switch's interface list.

        A port ID is an interface's position in that list, not its port number, so
        it can only be learned from the whole list: a broken-out port contributes one
        ID per subport and shifts every later port up. Only the Ethernet entries are
        data ports; parse_cisco drops the rest.
        """
        port_names = PortNameResolver(parser=parse_cisco)
        port_names.learn(row.get("interface", "") for row in rows)
        if port_names.has_breakout:
            self._logger.debug(f"port names: breakout shifts port ids from port numbers: {port_names.mapping()}")
        return port_names

    @staticmethod
    def _parse_system_status(version: dict[str, Any], hostname: dict[str, Any]) -> NetworkSwitchStatus:
        days, hours, minutes, seconds = (
            int(version.get(f"kern_uptm_{unit}", 0) or 0) for unit in ("days", "hrs", "mins", "secs")
        )
        return NetworkSwitchStatus(
            uptime=str(((days * 24 + hours) * 60 + minutes) * 60 + seconds),
            hostname=hostname.get("hostname") or version.get("host_name"),
            model=version.get("chassis_id"),
            serial_number=version.get("proc_board_id"),
            firmware_version=version.get("bios_ver_str"),
            # NX-API has no uniform ASIC field; left None as on Arista.
            asic_model=None,
            software_version=version.get("nxos_ver_str"),
        )


def _rows(body: dict[str, Any], table: str, row: str) -> list[dict[str, Any]]:
    """The rows of an NX-API table. NX-API returns a single row as an object and
    several as a list, and leaves the table out when it is empty."""
    rows = body.get(table, {})
    rows = rows.get(row, []) if isinstance(rows, dict) else []
    if isinstance(rows, dict):
        rows = [rows]
    return [r for r in rows if isinstance(r, dict)]


def _full_name(name: str) -> str:
    """Expands NX-OS's abbreviated "Eth1/1" to "Ethernet1/1"; other names pass through."""
    name = name.strip() if isinstance(name, str) else ""
    if name.startswith("Eth") and not name.startswith("Ethernet"):
        return "Ethernet" + name[len("Eth") :]
    return name
