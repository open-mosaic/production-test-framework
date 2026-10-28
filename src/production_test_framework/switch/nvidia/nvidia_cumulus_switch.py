# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2025 Delos Data, Inc.

"""NVIDIA Spectrum switch driver via Cumulus Linux NVUE."""

import logging
import time
from ipaddress import ip_interface
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from requests.auth import HTTPBasicAuth
from urllib3.util.retry import Retry

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
from production_test_framework.switch.network_switch import NetworkSwitch, parse_svi_address
from production_test_framework.switch.nvidia.nvue_paths import (
    BRIDGE_DOMAIN,
    BRIDGE_DOMAIN_MAC_TABLE_PATH,
    BRIDGE_DOMAIN_VLANS_PATH,
    FIRMWARE_PATH,
    INTERFACES_PATH,
    PLATFORM_PATH,
    QOS_PFC_PROFILE_PATH,
    QOS_ROCE_PATH,
    REVISION_PATH,
    SYSTEM_PATH,
    bridge_domain_vlan_path,
    interface_bridge_domain_path,
    interface_bridge_vlan_path,
    interface_path,
    revision_path,
    svi_path,
)
from production_test_framework.switch.port_sort import port_id_sort_key
from production_test_framework.switch.portname import PortNameResolver

requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)

# OpenAPI getInterfaces view=description: admin-status, oper-status, description
VIEW_DESCRIPTION = "description"

# OpenAPI getInterfaces view=lldp-detail: per-interface LLDP neighbor detail.
VIEW_LLDP_DETAIL = "lldp-detail"

# NVUE's link fec values; "auto" (and none set) is the switch's own choice.
_FEC_VALUES = {FecEncoding.OFF: "off", FecEncoding.FIRE_CODE: "baser", FecEncoding.REED_SOLOMON: "rs"}

# NVUE names data ports "swp<cage>", plus a subport index when the cage is
# broken out. Management, loopback, bridge, bond and VLAN interfaces are
# reported alongside them and are not data ports.
_SWP_PREFIX = "swp"

# NVUE spells the bridge learning key "on"/"off" before 5.15 and "enabled"/"disabled" from it.
_LEARNING_VALUES = {"on": True, "enabled": True, "off": False, "disabled": False}

# Same release split for RoCE: "enable": "on" before 5.15, "state": "enabled" from it.
_ROCE_STATE_KEYS = ("state", "enable")
_ROCE_ON_VALUES = frozenset({"on", "enabled"})

# The RoCE mode that carries PFC. NVUE defaults to it, so a missing mode counts as lossless.
_ROCE_MODE_LOSSLESS = "lossless"

# NVUE applies config changes asynchronously once a changeset is applied.
_APPLY_TIMEOUT_S = 60.0
_APPLY_INTERVAL_S = 1.0

_APPLIED_STATES = frozenset({"applied", "applied_and_saved"})
_APPLY_ERROR_STATES = frozenset({"apply_error", "invalid", "rejected"})

_CONNECT_RETRIES = 5
_RETRY_BACKOFF_S = 0.3


def _vlan_id(value: Any) -> int | None:
    """A VLAN number from NVUE, which may send it as an int or a digit string; None otherwise."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


class NvidiaCumulusSwitch(NetworkSwitch):
    """NVUE client for Cumulus Linux (e.g. Spectrum-5610)."""

    def __init__(self, config: NetworkSwitchConfig) -> None:
        super().__init__(config)
        self._logger = logging.getLogger(__name__)
        self._logger.debug(f"Initializing NvidiaCumulusSwitch with config: {config}")
        self._api_root = "/nvue_v1"
        self._base_url = f"https://{config.host}:{config.port}{self._api_root}"
        self._interfaces_config_cache: dict[str, Any] | None = None
        self._interfaces_applied_cache: dict[str, Any] | None = None
        self._session = requests.Session()
        adapter = HTTPAdapter(
            max_retries=Retry(
                total=_CONNECT_RETRIES,
                connect=_CONNECT_RETRIES,
                backoff_factor=_RETRY_BACKOFF_S,
            )
        )
        self._session.mount("https://", adapter)

    @property
    def status(self) -> NetworkSwitchStatus:
        system = self._run_api_call(SYSTEM_PATH)
        platform = self._run_api_call(PLATFORM_PATH)
        firmware = self._run_api_call(FIRMWARE_PATH)
        return self._parse_system_status(system, platform, firmware)

    @property
    def ports(self) -> list[Port]:
        """All interfaces (OpenAPI operationId: getInterfaces, view=description)."""
        interfaces = self._run_api_call(
            INTERFACES_PATH,
            params={"view": VIEW_DESCRIPTION},
        )
        return self._parse_ports(interfaces)

    @property
    def vlans(self) -> list[Vlan]:
        """Configured VLANs on the bridge domain (OpenAPI operationId: getBridgeDomainVlans)."""
        self.refresh()
        vlan_configs = self._run_api_call(BRIDGE_DOMAIN_VLANS_PATH)
        membership = self._vlan_membership_by_id()
        return self._parse_vlans(vlan_configs, membership)

    @property
    def lldp_neighbors(self) -> list[LldpNeighbor]:
        """LLDP neighbors advertising a MAC chassis id (OpenAPI getInterfaces, view=lldp-detail)."""
        interfaces = self._run_api_call(INTERFACES_PATH, params={"view": VIEW_LLDP_DETAIL})
        return self._parse_lldp_neighbors(interfaces)

    @property
    def mac_table(self) -> list[MacEntry]:
        """MAC/FDB table for the bridge domain (NVUE: bridge domain <d> mac-table)."""
        entries = self._run_api_call(BRIDGE_DOMAIN_MAC_TABLE_PATH)
        return self._parse_mac_table(entries)

    def port(self, port_id: str) -> Port:
        """Single interface (OpenAPI operationId: getInterface)."""
        interface = self._run_api_call(interface_path(port_id))
        return self._parse_port(port_id, interface)

    def port_bridge(self, port_id: str) -> PortBridge:
        """An interface's bridge domain membership (NVUE: interface <id> bridge domain <d>)."""
        # rev=applied: the operational view omits the untagged (PVID) VLAN.
        try:
            body = self._run_api_call(interface_bridge_domain_path(port_id), params={"rev": "applied"})
        except SwitchAPIError as error:
            if error.status_code != 404:
                raise
            # NVUE answers 404 for an interface outside the bridge; raise only if the interface itself is missing.
            self._run_api_call(interface_path(port_id))
            return PortBridge(interface=port_id)
        return self._parse_port_bridge(port_id, body)

    @property
    def flow_control(self) -> FlowControl:
        """Switch-wide flow control (NVUE: qos roce, and qos pfc default-global)."""
        # rev=applied: a staged but unapplied revision must not show up as the switch's state.
        roce = self._applied_config_or_none(QOS_ROCE_PATH)
        pfc = self._applied_config_or_none(QOS_PFC_PROFILE_PATH)
        return FlowControl(roce_lossless=self._roce_lossless(roce or {}), pfc_profile=bool(pfc))

    def vlan(self, vlan_id: str) -> Vlan:
        """Single VLAN with member ports (OpenAPI operationId: getBridgeDomainVlan)."""
        self.refresh()
        vlan_configs = self._run_api_call(BRIDGE_DOMAIN_VLANS_PATH)
        if vlan_id not in vlan_configs:
            raise SwitchAPIError(f"VLAN {vlan_id} not found on bridge domain {BRIDGE_DOMAIN}")
        membership = self._vlan_membership_by_id()
        return self._parse_vlan(vlan_id, membership.get(vlan_id, []))

    def set_port_admin_state(self, port_id: str, up: bool) -> None:
        """Administratively enable (up=True) or disable (up=False) a port"""
        state = "up" if up else "down"
        revision_id = self._create_revision()
        self._run_api_write(
            "PATCH",
            interface_path(port_id),
            params={"rev": revision_id},
            json_body={"link": {"state": {state: {}}}},
        )
        self._apply_revision(revision_id)
        self.refresh()

    def delete_vlan(self, vlan_id: str) -> None:
        """Remove a VLAN from the bridge domain and from every member interface.

        Membership is carried per-interface, so the VLAN is unset both on the
        bridge domain VLAN list and on each interface that references it, in a
        single changeset.
        """
        self.refresh()
        member_ports = self._vlan_membership_by_id().get(vlan_id, [])
        revision_id = self._create_revision()
        for port_id in member_ports:
            self._run_api_write(
                "DELETE",
                interface_bridge_vlan_path(port_id, vlan_id),
                params={"rev": revision_id},
            )
        self._run_api_write(
            "DELETE",
            bridge_domain_vlan_path(vlan_id),
            params={"rev": revision_id},
        )
        self._apply_revision(revision_id)
        self.refresh()

    def port_fec(self, port_id: str) -> FecEncoding | None:
        """The port's link fec from its applied config (NVUE: interface <id> link fec)."""
        body = self._run_api_call(interface_path(port_id), params={"rev": "applied"})
        link = body.get("link")
        fec = link.get("fec") if isinstance(link, dict) else None
        return next((encoding for encoding, value in _FEC_VALUES.items() if value == fec), None)

    def set_port_fec(self, port_id: str, fec: FecEncoding | None) -> None:
        """Set the port's link fec; None sets auto, the switch's default."""
        value = "auto" if fec is None else _FEC_VALUES.get(fec)
        if value is None:
            raise ValueError(f"unsupported FEC {fec!r}")
        revision_id = self._create_revision()
        self._run_api_write(
            "PATCH", interface_path(port_id), params={"rev": revision_id}, json_body={"link": {"fec": value}}
        )
        self._apply_revision(revision_id)
        self.refresh()

    def svi_address(self, vlan_id: str) -> SviAddress | None:
        """The SVI's address from its applied config (NVUE: interface vlan<id> ip address)."""
        body = self._applied_config_or_none(svi_path(vlan_id))
        ip = body.get("ip") if body else None
        addresses = ip.get("address") if isinstance(ip, dict) else None
        if not isinstance(addresses, dict):
            return None
        # The kernel's link-local address is the switch's own, not one set here.
        parsed = sorted(
            (a for a in map(_parse_address, addresses) if a is not None and not a.ip.is_link_local),
            key=lambda a: (a.version, a),
        )
        return parsed[0] if parsed else None

    def set_svi_address(self, vlan_id: str, address: SviAddress | str | None) -> None:
        """Make address the SVI's only address, or remove the SVI for None, in one changeset."""
        # Everything that can fail on the arguments fails before the changeset exists.
        svi = None
        if address is not None:
            parsed = parse_svi_address(address)
            svi = {
                f"vlan{vlan_id}": {"type": "svi", "vlan": int(vlan_id), "ip": {"address": {parsed.with_prefixlen: {}}}}
            }
        revision_id = self._create_revision()
        if svi is None:
            self._delete_if_present(svi_path(vlan_id), revision_id)
        else:
            self._delete_if_present(f"{svi_path(vlan_id)}/ip/address", revision_id)
            self._run_api_write("PATCH", INTERFACES_PATH, params={"rev": revision_id}, json_body=svi)
        self._apply_revision(revision_id)
        self.refresh()

    def _delete_if_present(self, path: str, revision_id: str) -> None:
        try:
            self._run_api_write("DELETE", path, params={"rev": revision_id})
        except SwitchAPIError as error:
            if error.status_code != 404:
                raise

    def refresh(self) -> None:
        """Drop cached interface data so the next read re-fetches from the switch."""
        self._interfaces_config_cache = None
        self._interfaces_applied_cache = None

    def _create_revision(self) -> str:
        """Create a new changeset and return its id (POST /revision)."""
        body = self._run_api_write("POST", REVISION_PATH)
        if not isinstance(body, dict) or not body:
            raise SwitchAPIError(f"unexpected NVUE revision response: {body!r}")
        return next(iter(body))

    def _apply_revision(self, revision_id: str) -> None:
        """Apply a changeset and block until NVUE reports it applied."""
        self._run_api_write(
            "PATCH",
            revision_path(revision_id),
            json_body={"state": "apply", "auto-prompt": {"ays": "ays_yes"}},
        )
        deadline = time.monotonic() + _APPLY_TIMEOUT_S
        while True:
            body = self._run_api_call(revision_path(revision_id))
            state = body.get("state") if isinstance(body, dict) else None
            if state in _APPLIED_STATES:
                return
            if state in _APPLY_ERROR_STATES:
                raise SwitchAPIError(f"NVUE revision {revision_id} failed to apply: state={state!r}")
            if time.monotonic() >= deadline:
                raise SwitchAPIError(
                    f"NVUE revision {revision_id} not applied within {_APPLY_TIMEOUT_S}s (state={state!r})"
                )
            time.sleep(_APPLY_INTERVAL_S)

    def _interfaces_config(self) -> dict[str, Any]:
        """Operational interface state (oper-status, stats)."""
        if self._interfaces_config_cache is None:
            self._interfaces_config_cache = self._run_api_call(INTERFACES_PATH)
        return self._interfaces_config_cache

    def _interfaces_applied_config(self) -> dict[str, Any]:
        """Applied interface config carries bridge VLAN assignment."""
        if self._interfaces_applied_cache is None:
            self._interfaces_applied_cache = self._run_api_call(INTERFACES_PATH, params={"rev": "applied"})
        return self._interfaces_applied_cache

    def _applied_config_or_none(self, path: str) -> dict[str, Any] | None:
        """The applied config at path, or None when NVUE answers 404 because nothing is configured there."""
        try:
            return self._run_api_call(path, params={"rev": "applied"})
        except SwitchAPIError as error:
            if error.status_code != 404:
                raise
            return None

    def _vlan_membership_by_id(self) -> dict[str, list[str]]:
        """Map VLAN ID to interface names assigned on the configured bridge domain."""
        membership: dict[str, list[str]] = {}
        for interface_id, body in self._interfaces_applied_config().items():
            if not isinstance(body, dict):
                continue
            for vid in self._interface_vlan_ids(body, BRIDGE_DOMAIN):
                membership.setdefault(vid, []).append(interface_id)
        for vid in membership:
            membership[vid] = sorted(membership[vid])
        return membership

    @staticmethod
    def _interface_vlan_ids(interface_body: dict[str, Any], bridge_domain: str) -> list[str]:
        bridge = interface_body.get("bridge")
        if not isinstance(bridge, dict):
            return []
        domain_cfg = bridge.get("domain")
        if not isinstance(domain_cfg, dict):
            return []
        domain_body = domain_cfg.get(bridge_domain)
        if not isinstance(domain_body, dict):
            return []
        vlan_cfg = domain_body.get("vlan")
        if not isinstance(vlan_cfg, dict):
            return []
        return list(vlan_cfg.keys())

    def _member_ports(self, port_ids: list[str]) -> tuple[Port, ...]:
        interfaces = self._interfaces_config()
        ports: list[Port] = []
        for port_id in port_ids:
            body = interfaces.get(port_id)
            if isinstance(body, dict):
                ports.append(self._parse_port(port_id, body))
        return tuple(ports)

    def _parse_vlan(self, vlan_id: str, member_port_ids: list[str]) -> Vlan:
        return Vlan(id=vlan_id, ports=self._member_ports(member_port_ids))

    def _parse_vlans(self, vlan_configs: dict[str, Any], membership: dict[str, list[str]]) -> list[Vlan]:
        vlans = [self._parse_vlan(vid, membership.get(vid, [])) for vid in vlan_configs]
        return sorted(vlans, key=lambda vlan: int(vlan.id) if vlan.id.isdigit() else vlan.id)

    def _run_api_call(
        self,
        path: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{self._base_url}{path}"
        basic = HTTPBasicAuth(self._config.username, self._config.password)
        request_params = dict(params) if params else None

        self._logger.debug(f"NVUE GET {path} params={request_params}")
        response = self._session.get(
            url,
            auth=basic,
            params=request_params,
            verify=self._config.verify_tls,
            timeout=30,
        )

        if response.status_code != 200:
            self._logger.error(f"API call failed: {response.status_code} {response.text}")
            raise SwitchAPIError(f"API call failed: {response.status_code} {response.text}", response.status_code)

        result = response.json()
        if not isinstance(result, dict):
            raise SwitchAPIError(f"unexpected NVUE response type for {path}: {type(result).__name__}")
        self._logger.debug(f"NVUE response: {result}")
        return result

    def _run_api_write(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> Any:
        url = f"{self._base_url}{path}"
        basic = HTTPBasicAuth(self._config.username, self._config.password)
        request_params = dict(params) if params else None

        self._logger.debug(f"NVUE {method} {path} params={request_params} body={json_body}")
        response = self._session.request(
            method,
            url,
            auth=basic,
            params=request_params,
            json=json_body,
            verify=self._config.verify_tls,
            timeout=30,
        )

        if response.status_code not in (200, 201, 204):
            self._logger.error(f"API call failed: {response.status_code} {response.text}")
            raise SwitchAPIError(f"API call failed: {response.status_code} {response.text}", response.status_code)

        result = response.json() if response.content else None
        self._logger.debug(f"NVUE response: {result}")
        return result

    @staticmethod
    def _status_to_bool(status: str | None) -> bool | None:
        if status is None:
            return None
        if status == "up":
            return True
        if status == "down":
            return False
        return None

    def _parse_port(self, interface_id: str, body: dict[str, Any]) -> Port:
        link = body.get("link") if isinstance(body.get("link"), dict) else {}
        admin_status = link.get("admin-status")
        oper_status = link.get("oper-status")
        if admin_status is None and isinstance(link.get("state"), dict):
            state = link["state"]
            if "up" in state:
                admin_status = "up"
            elif "down" in state:
                admin_status = "down"
        description = body.get("description")
        if description == "":
            description = None
        return Port(
            id=interface_id,
            admin_up=self._status_to_bool(admin_status if isinstance(admin_status, str) else None),
            oper_up=self._status_to_bool(oper_status if isinstance(oper_status, str) else None),
            description=description if isinstance(description, str) else None,
        )

    def _parse_ports(self, interfaces: dict[str, Any]) -> list[Port]:
        ports = [self._parse_port(interface_id, body) for interface_id, body in interfaces.items()]
        return sorted(ports, key=lambda port: port_id_sort_key(port.id))

    @staticmethod
    def _parse_port_bridge(port_id: str, body: dict[str, Any]) -> PortBridge:
        if not body:  # not a member of the bridge domain
            return PortBridge(interface=port_id)
        # NVUE reports unset VLANs as the placeholder "none", so only a VLAN number counts as set.
        access = _vlan_id(body.get("access"))
        learning = body.get("learning")
        return PortBridge(
            interface=port_id,
            # NVUE has no trunk setting: a bridge member is a trunk unless given an access VLAN.
            mode="trunk" if access is None else "access",
            native_vlan=_vlan_id(body.get("untagged")) if access is None else access,
            learning=_LEARNING_VALUES.get(learning) if isinstance(learning, str) else None,
        )

    @staticmethod
    def _roce_lossless(body: dict[str, Any]) -> bool:
        """True when RoCE is on in lossless mode. RoCE in another mode brings no PFC, so it does not count."""
        state = next((body[key] for key in _ROCE_STATE_KEYS if key in body), None)
        if state not in _ROCE_ON_VALUES:
            return False
        mode = body.get("mode")
        return mode is None or mode == _ROCE_MODE_LOSSLESS

    def _parse_mac_table(self, entries: dict[str, Any]) -> list[MacEntry]:
        table: list[MacEntry] = []
        for entry in entries.values():
            if not isinstance(entry, dict):
                continue
            mac = entry.get("mac", "")
            interface = entry.get("interface", "")
            if not mac or not interface:
                continue
            table.append(
                MacEntry(
                    mac=mac.lower(),
                    port=interface,
                    vlan=entry.get("vlan"),
                    static=entry.get("entry-type") == "permanent",
                )
            )
        return table

    def _parse_lldp_neighbors(self, interfaces: dict[str, Any]) -> list[LldpNeighbor]:
        port_names = self._port_names(interfaces)
        neighbors: list[LldpNeighbor] = []
        for interface_id, body in interfaces.items():
            switch_port = port_names.port_id(interface_id)
            if switch_port is None or not isinstance(body, dict):
                continue
            lldp = body.get("lldp")
            if not isinstance(lldp, dict):
                continue
            per_neighbor = lldp.get("neighbor")
            if not isinstance(per_neighbor, dict):
                continue
            for neighbor in per_neighbor.values():
                if not isinstance(neighbor, dict):
                    continue
                # The lldp-detail view nests chassis info under "port"; keep only
                # neighbors advertising a MAC-address chassis id.
                port_obj = neighbor.get("port")
                if not isinstance(port_obj, dict) or port_obj.get("type") != "mac":
                    continue
                port_name = port_obj.get("name")
                if not isinstance(port_name, str):
                    continue
                mac = normalize_mac(port_name)
                if not mac:
                    self._logger.warning(f"lldp_neighbors: invalid MAC {port_name!r} on {interface_id}")
                    continue
                neighbors.append(LldpNeighbor(interface=interface_id, switch_port=switch_port, chassis_mac=mac))
        return sorted(neighbors, key=lambda neighbor: neighbor.switch_port)

    def _port_names(self, interfaces: dict[str, Any]) -> PortNameResolver:
        """Learn the port ID of every data port from the switch's interface list.

        A port ID is an interface's position in that list, not its cage number, so
        it can only be learned from the whole list: a broken-out cage contributes
        one ID per subport and shifts every later cage up. The lldp-detail view
        enumerates every interface, including those with no neighbor, so its keys
        are the full GET /interface list; only the swp entries are data ports.
        """
        port_names = PortNameResolver()
        port_names.learn(name for name in interfaces if name.startswith(_SWP_PREFIX))
        if port_names.has_breakout:
            self._logger.debug(f"port names: breakout shifts port ids from cage numbers: {port_names.mapping()}")
        return port_names

    def _parse_system_status(self, system: dict, platform: dict, firmware: dict) -> NetworkSwitchStatus:
        return NetworkSwitchStatus(
            uptime=system["uptime"],
            hostname=system["hostname"],
            model=platform["product-name"],
            serial_number=platform["serial-number"],
            firmware_version=firmware["Spectrum-4"]["actual-firmware"],
            asic_model=platform["asic-model"],
            software_version=system["version"]["image"],
        )


def _parse_address(value: str) -> SviAddress | None:
    try:
        return ip_interface(value)
    except ValueError:
        return None
