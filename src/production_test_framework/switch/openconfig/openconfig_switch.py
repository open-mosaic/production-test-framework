# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2026 Delos Data, Inc.

"""Vendor-neutral switch driver over gNMI, using the OpenConfig models.

Each part of the ``NetworkSwitch`` interface maps to these OpenConfig paths, where
NI is ``network-instances/network-instance[name=<network_instance>]``:

    status                -> system/state, components/component/state
    ports / port          -> interfaces/interface/state
    vlans / vlan          -> NI/vlans (+ each port's ethernet/switched-vlan/state for members)
    lldp_neighbors        -> lldp/interfaces/interface/neighbors
    mac_table             -> NI/fdb/mac-table/entries
    port_bridge           -> ethernet/switched-vlan/state, NI/fdb/state/mac-learning
    flow_control          -> no OpenConfig model, raises SwitchNotSupportedError
    set_port_admin_state  -> interfaces/interface/config/enabled
    delete_vlan           -> delete NI/vlans/vlan, and the VLAN from each port's switched-vlan
    port_fec              -> ethernet/config/fec-mode
    set_port_fec          -> ethernet/config/fec-mode
    svi_address           -> interfaces/interface[name=<svi>]/routed-vlan
    set_svi_address       -> interfaces/interface[name=<svi>]/routed-vlan

Whatever the switch does not implement raises SwitchNotSupportedError.
"""

import logging
import re
import time
from collections.abc import Callable
from ipaddress import ip_interface
from typing import Any

import grpc
from pygnmi.client import gNMIclient, gNMIException
from pygnmi.create_gnmi_path import gnmi_path_generator

from production_test_framework.switch.exceptions import SwitchAPIError, SwitchNotSupportedError
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
from production_test_framework.switch.network_switch import NetworkSwitch, parse_svi_address
from production_test_framework.switch.portname import NO_SUBPORT, InterfaceName, PortNameResolver, parse, parse_cisco

_TIMEOUT_S = 60

# The words a switch uses when it rejects a path it does not implement.
_UNSUPPORTED_HINTS = ("not supported", "unsupported", "not mapped", "not implemented")

_DATA_PORT = re.compile(r"^(swp|Ethernet|Eth)\d")

_FEC_MODES = {
    "FEC_DISABLED": FecEncoding.OFF,
    "FEC_FC": FecEncoding.FIRE_CODE,
    "FEC_RS528": FecEncoding.REED_SOLOMON,
    "FEC_RS544": FecEncoding.REED_SOLOMON,
    "FEC_RS544_LOW_LATENCY": FecEncoding.REED_SOLOMON,
    "FEC_RS544_2X_INTERLEAVE": FecEncoding.REED_SOLOMON,
    "FEC_RS544_2X_INTERLEAVE_CL172": FecEncoding.REED_SOLOMON,
    "FEC_RS544_2X_INTERLEAVE_ETC": FecEncoding.REED_SOLOMON,
}


class _PathNotFound(SwitchAPIError):
    """The switch has nothing at the path."""


class OpenConfigSwitch(NetworkSwitch):
    """gNMI client for switches that implement the OpenConfig models.

    network_instance is the instance holding the VLANs and the MAC table. svi_name
    formats a VLAN ID into its SVI's interface name, which OpenConfig leaves to the
    vendor ("vlan10" on NX-OS, "Vlan10" on EOS).
    """

    def __init__(
        self, config: NetworkSwitchConfig, network_instance: str = "default", svi_name: str = "vlan{}"
    ) -> None:
        super().__init__(config)
        self._logger = logging.getLogger(__name__)
        self._logger.debug(f"Initializing OpenConfigSwitch with host {config.host}:{config.port}")
        self._network_instance = network_instance
        self._svi_name = svi_name
        self._encoding: str | None = None

    @property
    def status(self) -> NetworkSwitchStatus:
        system = _dig(self._get("system/state"), "system", "state")
        try:
            components = _list(self._get("components/component/state"), "components", "component")
        except SwitchNotSupportedError:
            components = []
        chassis = _state_of_type(components, "CHASSIS")
        os_component = _state_of_type(components, "OPERATING_SYSTEM")
        boot_time = system.get("boot-time")
        return NetworkSwitchStatus(
            uptime=str(int(time.time() - int(boot_time) / 1e9)) if boot_time else None,
            hostname=system.get("hostname"),
            model=chassis.get("part-no") or chassis.get("description"),
            serial_number=chassis.get("serial-no"),
            firmware_version=chassis.get("firmware-version"),
            asic_model=None,
            software_version=system.get("software-version") or os_component.get("software-version"),
        )

    @property
    def ports(self) -> list[Port]:
        interfaces = _list(self._get("interfaces/interface/state"), "interfaces", "interface")
        ports = [_port(entry) for entry in interfaces if _is_ethernet(entry)]
        return sorted(ports, key=lambda port: _sort_key(port.id))

    @property
    def vlans(self) -> list[Vlan]:
        entries = _list(self._instance(self._get(self._ni("vlans"))), "vlans", "vlan")
        members = self._vlan_members()
        vlan_ids = sorted({vid for entry in entries if (vid := _vlan_id(entry)) is not None})
        return [Vlan(id=str(vid), ports=members.get(vid, ())) for vid in vlan_ids]

    @property
    def lldp_neighbors(self) -> list[LldpNeighbor]:
        """LLDP neighbors whose advertised port id is a MAC."""
        tree = self._get("lldp/interfaces/interface/neighbors", "interfaces/interface/state")
        port_names = _port_names(_list(tree, "interfaces", "interface"))
        neighbors: list[LldpNeighbor] = []
        for interface in _list(tree, "lldp", "interfaces", "interface"):
            name = _key(interface, "name")
            switch_port = port_names.port_id(name)
            if switch_port is None:
                continue  # a management or other non-data port
            for neighbor in _list(interface, "neighbors", "neighbor"):
                mac = normalize_mac(str(_dig(neighbor, "state").get("port-id", "")))
                if mac:
                    # chassis_mac carries the neighbor's port/NIC MAC, as the other drivers do.
                    neighbors.append(LldpNeighbor(interface=name, switch_port=switch_port, chassis_mac=mac))
        return sorted(neighbors, key=lambda neighbor: neighbor.switch_port)

    @property
    def mac_table(self) -> list[MacEntry]:
        path = self._ni("fdb/mac-table/entries")
        table: list[MacEntry] = []
        for entry in _list(self._instance(self._get(path)), "fdb", "mac-table", "entries", "entry"):
            state = _dig(entry, "state")
            mac = normalize_mac(str(_key(entry, "mac-address") or ""))
            port = _dig(entry, "interface", "interface-ref", "state").get("interface")
            if not mac or not port:
                continue
            vlan = _key(entry, "vlan")
            table.append(
                MacEntry(
                    mac=mac,
                    port=port,
                    vlan=int(vlan) if str(vlan).isdigit() else None,
                    static=_identity(state.get("entry-type")) == "STATIC",
                )
            )
        return table

    @property
    def flow_control(self) -> FlowControl:
        raise SwitchNotSupportedError("flow control (RoCE lossless, PFC) has no OpenConfig model")

    def port(self, port_id: str) -> Port:
        interfaces = _list(self._get(f"interfaces/interface[name={port_id}]/state"), "interfaces", "interface")
        if not interfaces:
            raise SwitchAPIError(f"port {port_id} not found")
        return _port(interfaces[0])

    def port_bridge(self, port_id: str) -> PortBridge:
        """A port's switched-VLAN state. OpenConfig sets MAC learning per network instance, not per
        port, so learning is the instance's setting."""
        path = f"interfaces/interface[name={port_id}]/ethernet/switched-vlan/state"
        interfaces = _list(self._get(path), "interfaces", "interface")
        state = _dig(interfaces[0], "ethernet", "switched-vlan", "state") if interfaces else {}
        if not state:  # a routed port does not bridge
            return PortBridge(interface=port_id)
        mode = _identity(state.get("interface-mode")).lower() or None
        native_vlan = state.get("access-vlan" if mode == "access" else "native-vlan")
        return PortBridge(
            interface=port_id,
            mode=mode,
            native_vlan=int(native_vlan) if native_vlan is not None else None,
            learning=self._mac_learning(),
        )

    def vlan(self, vlan_id: str) -> Vlan:
        entries = _list(self._instance(self._get(self._ni(f"vlans/vlan[vlan-id={vlan_id}]"))), "vlans", "vlan")
        if not entries:
            raise SwitchAPIError(f"VLAN {vlan_id} not found")
        return Vlan(id=str(vlan_id), ports=self._vlan_members().get(int(vlan_id), ()))

    def set_port_admin_state(self, port_id: str, up: bool) -> None:
        self._set(update=[(f"interfaces/interface[name={port_id}]/config/enabled", up)])

    def delete_vlan(self, vlan_id: str) -> None:
        """Remove the VLAN, and remove it from every port that carries it, in one transaction."""
        vid = int(vlan_id)
        deletes = [self._ni(f"vlans/vlan[vlan-id={vid}]")]
        replaces: list[tuple[str, Any]] = []
        tree = self._get("interfaces/interface/ethernet/switched-vlan/config")
        for interface in _list(tree, "interfaces", "interface"):
            config = _dig(interface, "ethernet", "switched-vlan", "config")
            base = f"interfaces/interface[name={_key(interface, 'name')}]/ethernet/switched-vlan/config"
            for leaf in ("access-vlan", "native-vlan"):
                if str(config.get(leaf)) == str(vid):
                    deletes.append(f"{base}/{leaf}")
            trunks = config.get("trunk-vlans") or []
            rest = _without(trunks, vid)
            if rest != trunks:
                if rest:
                    replaces.append((f"{base}/trunk-vlans", rest))
                else:
                    deletes.append(f"{base}/trunk-vlans")
        self._set(delete=deletes, replace=replaces)

    def port_fec(self, port_id: str) -> FecEncoding | None:
        """The port's configured FEC; None when unset or FEC_AUTO, the switch's choice."""
        path = f"interfaces/interface[name={port_id}]/ethernet/config/fec-mode"
        interfaces = _list(self._get(path), "interfaces", "interface")
        mode = _identity(_dig(interfaces[0], "ethernet", "config").get("fec-mode")) if interfaces else ""
        if not mode or mode == "FEC_AUTO":
            return None
        if mode not in _FEC_MODES:
            raise SwitchAPIError(f"port {port_id}: unknown FEC mode {mode!r}")
        return _FEC_MODES[mode]

    def set_port_fec(self, port_id: str, fec: FecEncoding | None) -> None:
        """Set the port's FEC; None removes the setting, leaving the switch's choice. Reed-Solomon is
        RS(544) at 200G and above and RS(528) below, from the port's speed."""
        path = f"interfaces/interface[name={port_id}]/ethernet/config/fec-mode"
        match fec:
            case None:
                self._set(delete=[path])
                return
            case FecEncoding.OFF:
                mode = "FEC_DISABLED"
            case FecEncoding.FIRE_CODE:
                mode = "FEC_FC"
            case FecEncoding.REED_SOLOMON:
                mode = "FEC_RS544" if self._port_speed_gbps(port_id) >= 200 else "FEC_RS528"
            case _:
                raise ValueError(f"unsupported FEC {fec!r}")
        self._set(update=[(path, self._identity_value("openconfig-if-ethernet", mode))])

    def svi_address(self, vlan_id: str) -> SviAddress | None:
        """The SVI's first IPv4 address, else its first non link-local IPv6 one; None without either."""
        path = f"interfaces/interface[name={self._svi_name.format(vlan_id)}]/routed-vlan"
        interfaces = _list(self._get(path), "interfaces", "interface")
        routed = _dig(interfaces[0], "routed-vlan") if interfaces else {}
        for family in ("ipv4", "ipv6"):
            for address in _list(routed, family, "addresses", "address"):
                ip = _key(address, "ip")
                length = _dig(address, "config").get("prefix-length", _dig(address, "state").get("prefix-length"))
                if ip and length is not None and not (parsed := ip_interface(f"{ip}/{length}")).is_link_local:
                    return parsed
        return None

    def set_svi_address(self, vlan_id: str, address: SviAddress | str | None) -> None:
        """Make address the SVI's only address, or remove the SVI for None."""
        base = f"interfaces/interface[name={self._svi_name.format(vlan_id)}]"
        if address is None:
            try:
                self._set(delete=[base])
            except _PathNotFound:
                pass  # no SVI to remove
            return
        address = parse_svi_address(address)
        family = "ipv4" if address.version == 4 else "ipv6"
        ip = str(address.ip)
        self._set(
            delete=[f"{base}/routed-vlan/ipv4/addresses", f"{base}/routed-vlan/ipv6/addresses"],
            update=[
                (
                    f"{base}/routed-vlan/{family}/addresses/address[ip={ip}]/config",
                    {"ip": ip, "prefix-length": address.network.prefixlen},
                )
            ],
        )

    # --- helpers over the switch ---------------------------------------------

    def _ni(self, rest: str) -> str:
        return f"network-instances/network-instance[name={self._network_instance}]/{rest}"

    def _instance(self, tree: dict[str, Any]) -> dict[str, Any]:
        """The configured network instance's entry in tree; empty without one."""
        entries = _list(tree, "network-instances", "network-instance")
        return next((e for e in entries if str(_key(e, "name")) == self._network_instance), {})

    def _vlan_members(self) -> dict[int, tuple[Port, ...]]:
        """{VLAN ID: the ports carrying it}, from each port's switched-VLAN state."""
        tree = self._get("interfaces/interface/state", "interfaces/interface/ethernet/switched-vlan/state")
        members: dict[int, list[Port]] = {}
        for interface in sorted(_list(tree, "interfaces", "interface"), key=lambda e: _sort_key(_key(e, "name"))):
            state = _dig(interface, "ethernet", "switched-vlan", "state")
            for vid in _vlans_carried(state):
                members.setdefault(vid, []).append(_port(interface))
        return {vid: tuple(ports) for vid, ports in members.items()}

    def _mac_learning(self) -> bool | None:
        try:
            tree = self._get(self._ni("fdb/state/mac-learning"))
        except SwitchNotSupportedError:
            return None
        learning = _dig(self._instance(tree), "fdb", "state").get("mac-learning")
        return learning if isinstance(learning, bool) else None

    def _port_speed_gbps(self, port_id: str) -> int:
        path = f"interfaces/interface[name={port_id}]/ethernet/state/port-speed"
        interfaces = _list(self._get(path), "interfaces", "interface")
        speed = _identity(_dig(interfaces[0], "ethernet", "state").get("port-speed")) if interfaces else ""
        match = re.search(r"(\d+)GB", speed)
        return int(match.group(1)) if match else 0

    def _identity_value(self, module: str, name: str) -> str:
        """An identity as the encoding in use writes it: module-qualified in JSON_IETF."""
        return f"{module}:{name}" if self._encoding == "json_ietf" else name

    # --- gNMI plumbing -------------------------------------------------------

    def _call(self, what: str, call: Callable[[gNMIclient], Any]) -> Any:
        """Run call on a fresh gNMI session, picking the JSON encoding on first use."""
        try:
            with gNMIclient(
                target=(self._config.host, self._config.port),
                username=self._config.username,
                password=self._config.password,
                skip_verify=not self._config.verify_tls,
                gnmi_timeout=_TIMEOUT_S,
            ) as client:
                if self._encoding is None:
                    self._encoding = _pick_encoding(client.capabilities())
                return call(client)
        except (gNMIException, grpc.RpcError, grpc.FutureTimeoutError, OSError) as exc:
            raise _switch_error(what, exc) from exc

    def _get(self, *paths: str) -> dict[str, Any]:
        """Everything the switch returns for paths, merged into one tree, with module prefixes
        dropped from names; empty when the switch has nothing there."""
        self._logger.debug(f"gNMI Get: {paths}")
        try:
            response = self._call(f"Get {list(paths)}", lambda c: c.get(path=list(paths), encoding=self._encoding))
        except _PathNotFound:
            return {}
        tree: dict[str, Any] = {}
        for notification in (response or {}).get("notification") or []:
            prefix = _elems(notification.get("prefix"))
            for update in notification.get("update") or []:
                _merge(tree, prefix + _elems(update.get("path")), _strip_modules(update.get("val")))
        return tree

    def _set(
        self,
        delete: list[str] | None = None,
        replace: list[tuple[str, Any]] | None = None,
        update: list[tuple[str, Any]] | None = None,
    ) -> None:
        """One gNMI Set transaction: the switch applies deletes, then replaces, then updates."""
        self._logger.debug(f"gNMI Set: delete={delete} replace={replace} update={update}")
        self._call(
            "Set",
            lambda c: c.set(
                delete=delete or None, replace=replace or None, update=update or None, encoding=self._encoding
            ),
        )


# --- the tree the switch returns ---------------------------------------------


def _elems(path: str | None) -> list[tuple[str, dict[str, str]]]:
    """A path string's elements as (name, keys)."""
    if not path:
        return []
    return [(element.name.split(":")[-1], dict(element.key)) for element in gnmi_path_generator(path).elem]


def _merge(tree: dict[str, Any], elems: list[tuple[str, dict[str, str]]], value: Any) -> None:
    """Put value at elems in tree. A keyed element is a list entry holding its keys, as in the
    JSON encoding, so a tree built from several updates reads the same as one JSON value."""
    node = tree
    for i, (name, keys) in enumerate(elems):
        last = i == len(elems) - 1
        if keys:
            entries = node.setdefault(name, [])
            entry = next((e for e in entries if all(str(e.get(k)) == v for k, v in keys.items())), None)
            if entry is None:
                entry = dict(keys)
                entries.append(entry)
            node = entry
        elif last and not isinstance(value, dict):
            node[name] = value
            return
        else:
            node = node.setdefault(name, {})
    if isinstance(value, dict):
        _deep_merge(node, value)


def _deep_merge(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        elif isinstance(value, list) and isinstance(target.get(key), list):
            _merge_entries(target[key], value)
        else:
            target[key] = value


def _merge_entries(target: list[Any], source: list[Any]) -> None:
    """Merge each list entry into the target entry with the same keys, else append it."""
    for item in source:
        match = next((e for e in target if _same_entry(e, item)), None) if isinstance(item, dict) else None
        if match is not None:
            _deep_merge(match, item)
        elif isinstance(item, dict) or item not in target:
            target.append(item)


def _same_entry(a: Any, b: dict[str, Any]) -> bool:
    """Whether two list entries agree on every scalar leaf both hold; their keys are among those."""
    if not isinstance(a, dict):
        return False
    scalars = {k for k, v in a.items() if not isinstance(v, dict | list)}
    shared = [k for k, v in b.items() if k in scalars and not isinstance(v, dict | list)]
    return bool(shared) and all(str(a[k]) == str(b[k]) for k in shared)


def _strip_modules(value: Any) -> Any:
    """value with the "module:" prefix dropped from every name."""
    if isinstance(value, dict):
        return {key.split(":")[-1]: _strip_modules(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_strip_modules(child) for child in value]
    return value


def _dig(node: dict[str, Any], *names: str) -> dict[str, Any]:
    """The container at names under node; empty when any is missing."""
    for name in names:
        child = node.get(name) if isinstance(node, dict) else None
        if not isinstance(child, dict):
            return {}
        node = child
    return node


def _list(node: dict[str, Any], *names: str) -> list[dict[str, Any]]:
    """The entries of the list at names (the last one) under node."""
    entries = _dig(node, *names[:-1]).get(names[-1])
    return [e for e in entries if isinstance(e, dict)] if isinstance(entries, list) else []


def _key(entry: dict[str, Any], name: str) -> Any:
    """A list entry's key leaf, which a switch may put at the entry, or only in config or state."""
    return entry.get(name, _dig(entry, "config").get(name, _dig(entry, "state").get(name)))


def _identity(value: Any) -> str:
    """An identity's name without its module ("openconfig-vlan-types:TRUNK" -> "TRUNK")."""
    return value.split(":")[-1] if isinstance(value, str) else ""


# --- reading the models ------------------------------------------------------


def _port(interface: dict[str, Any]) -> Port:
    state = _dig(interface, "state")
    admin, oper, description = state.get("admin-status"), state.get("oper-status"), state.get("description")
    return Port(
        id=_key(interface, "name"),
        admin_up=admin == "UP" if isinstance(admin, str) else None,
        oper_up=oper == "UP" if isinstance(oper, str) else None,
        description=description if isinstance(description, str) and description else None,
    )


def _is_ethernet(interface: dict[str, Any]) -> bool:
    """Whether the interface is a physical Ethernet port; one with no type reported counts."""
    kind = _identity(_dig(interface, "state").get("type") or _dig(interface, "config").get("type"))
    return not kind or kind == "ethernetCsmacd"


def _parse_name(name: str) -> InterfaceName | None:
    """An interface name in NX-OS notation, else the generic one. The two only disagree on a
    two-number name, and order those the same."""
    return parse_cisco(name) or parse(name)


def _sort_key(name: str) -> tuple[str, int, int, int]:
    parsed = _parse_name(name)
    return parsed.sort_key if parsed else (name, 0, 0, NO_SUBPORT)


def _port_names(interfaces: list[dict[str, Any]]) -> PortNameResolver:
    port_names = PortNameResolver(parser=_parse_name)
    port_names.learn(
        name
        for interface in interfaces
        if _is_ethernet(interface) and _DATA_PORT.match(name := _key(interface, "name"))
    )
    return port_names


def _vlan_id(entry: dict[str, Any]) -> int | None:
    vid = _key(entry, "vlan-id")
    return int(vid) if str(vid).isdigit() else None


def _trunk_ids(trunks: list[Any]) -> set[int]:
    """The VLAN IDs of a trunk-vlans leaf-list, whose items are IDs or "low..high" ranges."""
    ids: set[int] = set()
    for item in trunks:
        low, _, high = str(item).partition("..")
        if low.isdigit():
            ids.update(range(int(low), int(high or low) + 1))
    return ids


def _vlans_carried(state: dict[str, Any]) -> set[int]:
    """The VLANs a port's switched-VLAN state carries."""
    if not state:
        return set()
    mode = _identity(state.get("interface-mode"))
    if mode == "ACCESS":
        leaves = [state.get("access-vlan")]
    else:
        leaves = [state.get("native-vlan"), *(_trunk_ids(state.get("trunk-vlans") or []))]
        if mode != "TRUNK":
            leaves.append(state.get("access-vlan"))
    return {int(v) for v in leaves if v is not None and str(v).isdigit()}


def _without(trunks: list[Any], vid: int) -> list[Any]:
    """trunks without vid, splitting a range that holds it."""
    rest: list[Any] = []
    for item in trunks:
        low, _, high = str(item).partition("..")
        if not high:
            if not (low.isdigit() and int(low) == vid):
                rest.append(item)
            continue
        lo, hi = int(low), int(high)
        if not lo <= vid <= hi:
            rest.append(item)
            continue
        for a, b in ((lo, vid - 1), (vid + 1, hi)):
            if a == b:
                rest.append(a)
            elif a < b:
                rest.append(f"{a}..{b}")
    return rest


def _state_of_type(components: list[dict[str, Any]], kind: str) -> dict[str, Any]:
    """The state of the first component of type kind."""
    for component in components:
        state = _dig(component, "state")
        if _identity(state.get("type")) == kind:
            return state
    return {}


# --- errors ------------------------------------------------------------------


def _pick_encoding(capabilities: dict[str, Any] | None) -> str:
    supported = [str(e).lower() for e in (capabilities or {}).get("supported_encodings") or []]
    for encoding in ("json_ietf", "json"):
        if encoding in supported:
            return encoding
    raise SwitchNotSupportedError(f"the switch supports neither JSON encoding (it has {supported})")


def _rpc_error(exc: BaseException) -> grpc.RpcError | None:
    """The gRPC error behind exc; pygnmi wraps it in its own exception."""
    seen: BaseException | None = exc
    while seen is not None:
        if isinstance(seen, grpc.RpcError):
            return seen
        inner = getattr(seen, "orig_exc", None)
        seen = inner if isinstance(inner, BaseException) else seen.__cause__ or seen.__context__
    return None


def _switch_error(what: str, exc: BaseException) -> SwitchAPIError:
    rpc = _rpc_error(exc)
    code = rpc.code() if rpc is not None and hasattr(rpc, "code") else None
    details = (rpc.details() if rpc is not None and hasattr(rpc, "details") else None) or str(exc)
    if code == grpc.StatusCode.NOT_FOUND:
        return _PathNotFound(f"gNMI {what}: {details}")
    if code == grpc.StatusCode.UNIMPLEMENTED or any(hint in details.lower() for hint in _UNSUPPORTED_HINTS):
        return SwitchNotSupportedError(f"gNMI {what}: not supported by the switch: {details}")
    return SwitchAPIError(f"gNMI {what} failed: {details}")
