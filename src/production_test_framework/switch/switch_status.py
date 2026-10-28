# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2025 Delos Data, Inc.

"""Read-only switch status CLI.

Usage:
    switch-status --hostname=HOSTNAME --username=USERNAME [--password=PASSWORD] [--switch-type=TYPE] [--log-level=LEVEL]
    switch-status (-h | --help)

Options:
    -h --help                   Show this help message and exit
    --hostname=HOSTNAME         Switch management hostname or IP address
    --username=USERNAME         Management API username
    --password=PASSWORD         Management API password (prompted securely if omitted)
    --switch-type=TYPE          Switch driver to use [default: nvidia-cumulus]
    --log-level=LEVEL           Enable logging to stdout at LEVEL (e.g. DEBUG, INFO, WARNING)
"""

import logging
import os
import sys
from getpass import getpass
from typing import IO, TextIO

from docopt import docopt

from production_test_framework.switch.arista.arista_eos_switch import AristaEosSwitch
from production_test_framework.switch.cisco.nxos_switch import CiscoNxosSwitch
from production_test_framework.switch.models import NetworkSwitchConfig, NetworkSwitchStatus, Port, Vlan
from production_test_framework.switch.network_switch import NetworkSwitch
from production_test_framework.switch.nvidia.nvidia_cumulus_switch import NvidiaCumulusSwitch
from production_test_framework.switch.port_sort import sort_ports

NVIDIA_SWITCH_TYPE = "nvidia-cumulus"
ARISTA_SWITCH_TYPE = "arista-eos"
CISCO_SWITCH_TYPE = "cisco-nxos"

_SWITCH_TYPES = frozenset({NVIDIA_SWITCH_TYPE, ARISTA_SWITCH_TYPE, CISCO_SWITCH_TYPE})

logger = logging.getLogger(__name__)

_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_CYAN = "\033[36m"
_YELLOW = "\033[33m"
_GREEN = "\033[32m"
_RED = "\033[31m"


def _color_enabled(stream: IO[str] | None = None) -> bool:
    if os.environ.get("NO_COLOR", ""):
        return False
    if os.environ.get("FORCE_COLOR", ""):
        return True
    stream = stream or sys.stdout
    return hasattr(stream, "isatty") and stream.isatty()


def _paint(text: str, *codes: str, enabled: bool) -> str:
    if not enabled or not codes:
        return text
    prefix = "".join(codes)
    return f"{prefix}{text}{_RESET}"


def _section(title: str, rows: list[tuple[str, str]], *, enabled: bool) -> list[str]:
    lines = [
        _paint(title, _BOLD, _YELLOW, enabled=enabled),
    ]
    label_width = max((len(label) for label, _ in rows), default=0)
    for label, value in rows:
        label_col = _paint(f"{label:<{label_width}}", _DIM, enabled=enabled)
        value_col = _paint(value, _GREEN, enabled=enabled)
        lines.append(f"  {label_col}  {value_col}")
    return lines


def format_switch_status(status: NetworkSwitchStatus, *, use_color: bool | None = None) -> str:
    """Format switch status for terminal display."""
    enabled = _color_enabled() if use_color is None else use_color

    title = _paint(f"Switch status — {status.hostname}", _BOLD, _CYAN, enabled=enabled)
    rule = _paint("─" * max(len(status.hostname) + 16, 40), _DIM, enabled=enabled)

    blocks = [
        _section(
            "System",
            [
                ("Hostname", status.hostname),
                ("Uptime", status.uptime),
            ],
            enabled=enabled,
        ),
        _section(
            "Hardware",
            [
                ("Model", status.model),
                ("Serial", status.serial_number),
                ("ASIC", status.asic_model),
            ],
            enabled=enabled,
        ),
        _section(
            "Software",
            [
                ("OS image", status.software_version),
                ("Firmware", status.firmware_version),
            ],
            enabled=enabled,
        ),
    ]

    body = "\n\n".join("\n".join(block) for block in blocks)
    return f"\n\n{title}\n{rule}\n\n{body}\n\n\n"


def _link_status_label(value: bool | None) -> str:
    if value is True:
        return "up"
    if value is False:
        return "down"
    return "?"


def _paint_link_status(label: str, *, enabled: bool) -> str:
    if label == "up":
        return _paint(label, _GREEN, enabled=enabled)
    if label == "down":
        return _paint(label, _RED, enabled=enabled)
    return _paint(label, _DIM, enabled=enabled)


def _port_table_rows(ports: list[Port]) -> list[tuple[str, str, str, str]]:
    rows: list[tuple[str, str, str, str]] = []
    for port in ports:
        description = port.description if port.description else "—"
        rows.append(
            (
                port.id,
                _link_status_label(port.admin_up),
                _link_status_label(port.oper_up),
                description,
            )
        )
    return rows


def _format_port_table(
    rows: list[tuple[str, str, str, str]],
    *,
    enabled: bool,
    indent: str,
) -> list[str]:
    columns = ("Port", "Admin", "Oper", "Description")
    widths = [max(len(columns[i]), *(len(row[i]) for row in rows)) for i in range(len(columns))]

    def fmt_row(cells: tuple[str, ...], column_widths: list[int], *, header: bool = False) -> str:
        parts: list[str] = []
        for index, cell in enumerate(cells):
            if header:
                parts.append(_paint(cell.ljust(column_widths[index]), _BOLD, _DIM, enabled=enabled))
            elif index == 1 or index == 2:
                parts.append(_paint_link_status(cell.ljust(column_widths[index]), enabled=enabled))
            elif index == 3:
                parts.append(_paint(cell.ljust(column_widths[index]), _DIM, enabled=enabled))
            else:
                parts.append(_paint(cell.ljust(column_widths[index]), _GREEN, enabled=enabled))
        return indent + "  ".join(parts)

    table_lines = [
        fmt_row(columns, widths, header=True),
        fmt_row(tuple("─" * widths[i] for i in range(len(columns))), widths, header=True),
    ]
    table_lines.extend(fmt_row(row, widths) for row in rows)
    return table_lines


def format_ports(ports: list[Port], *, use_color: bool | None = None) -> str:
    """Format interface list as an aligned table."""
    enabled = _color_enabled() if use_color is None else use_color
    if not ports:
        header = _paint("Ports", _BOLD, _YELLOW, enabled=enabled)
        return f"{header}\n  (none)\n\n"

    lines = [_paint("Ports", _BOLD, _YELLOW, enabled=enabled)]
    lines.extend(_format_port_table(_port_table_rows(sort_ports(ports)), enabled=enabled, indent="  "))
    return "\n".join(lines) + "\n\n"


def format_vlans(vlans: list[Vlan], *, use_color: bool | None = None) -> str:
    """Format VLANs and their member ports for terminal display."""
    enabled = _color_enabled() if use_color is None else use_color
    if not vlans:
        header = _paint("VLANs", _BOLD, _YELLOW, enabled=enabled)
        return f"{header}\n  (none)\n\n"

    lines: list[str] = [_paint("VLANs", _BOLD, _YELLOW, enabled=enabled)]
    for vlan in vlans:
        lines.append(_paint(f"  VLAN {vlan.id}", _BOLD, _CYAN, enabled=enabled))
        if not vlan.ports:
            lines.append(_paint("    (no member ports)", _DIM, enabled=enabled))
            continue
        lines.extend(_format_port_table(_port_table_rows(sort_ports(list(vlan.ports))), enabled=enabled, indent="    "))

    return "\n".join(lines) + "\n\n"


def format_switch_report(
    status: NetworkSwitchStatus,
    ports: list[Port],
    vlans: list[Vlan],
    *,
    use_color: bool | None = None,
) -> str:
    """Format switch status, ports, and VLANs for terminal display."""
    enabled = _color_enabled() if use_color is None else use_color
    return (
        format_switch_status(status, use_color=enabled)
        + format_ports(ports, use_color=enabled)
        + format_vlans(vlans, use_color=enabled)
    )


def print_switch_status(
    status: NetworkSwitchStatus,
    ports: list[Port],
    vlans: list[Vlan],
    *,
    stream: TextIO | None = None,
    use_color: bool | None = None,
) -> None:
    stream = stream or sys.stdout
    if use_color is None:
        use_color = _color_enabled(stream)
    text = format_switch_report(status, ports, vlans, use_color=use_color)
    stream.write(text)
    if not text.endswith("\n"):
        stream.write("\n")


def _resolve_log_level(level_name: str) -> int:
    normalized = level_name.upper()
    level = logging.getLevelNamesMapping().get(normalized)
    if level is None:
        supported = ", ".join(sorted(logging.getLevelNamesMapping()))
        sys.exit(f"error: unknown log level {level_name!r} (supported: {supported})")
    return level


def _configure_logging(level_name: str) -> None:
    """Send log records for the switch package to stdout at the given level."""
    level = _resolve_log_level(level_name)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(
        logging.Formatter(
            fmt="{asctime} {levelname} {name}: {message}",
            datefmt="%H:%M:%S",
            style="{",
        )
    )

    switch_logger = logging.getLogger("production_test_framework.switch")
    switch_logger.handlers.clear()
    switch_logger.addHandler(handler)
    switch_logger.setLevel(level)
    switch_logger.propagate = False

    logger.log(level, f"logging enabled at {level_name.upper()} (production_test_framework.switch -> stdout)")


def resolve_password(args: dict) -> str:
    """Return password from argv or read it securely from stdin."""
    password = args.get("--password")
    if password:
        return password
    if sys.stdin.isatty():
        entered = getpass("Password: ")
        if not entered:
            sys.exit("error: empty password")
        return entered
    entered = sys.stdin.read().rstrip("\n")
    if not entered:
        sys.exit("error: empty password on stdin")
    return entered


def create_switch(
    switch_type: str,
    hostname: str,
    username: str,
    password: str,
) -> NetworkSwitch:
    """Instantiate a switch client for the given driver name."""
    if switch_type not in _SWITCH_TYPES:
        supported = ", ".join(sorted(_SWITCH_TYPES))
        sys.exit(f"error: unknown switch-type {switch_type!r} (supported: {supported})")

    if switch_type == NVIDIA_SWITCH_TYPE:
        config = NetworkSwitchConfig(
            host=hostname,
            username=username,
            password=password,
            verify_tls=False,
            port=8765,
        )
        return NvidiaCumulusSwitch(config)

    if switch_type == ARISTA_SWITCH_TYPE:
        config = NetworkSwitchConfig(
            host=hostname,
            username=username,
            password=password,
            verify_tls=False,
            port=80,
        )
        return AristaEosSwitch(config)

    if switch_type == CISCO_SWITCH_TYPE:
        config = NetworkSwitchConfig(
            host=hostname,
            username=username,
            password=password,
            verify_tls=False,
            port=443,
        )
        return CiscoNxosSwitch(config)

    raise AssertionError(f"unhandled switch-type: {switch_type}")


def main(argv: list[str] | None = None) -> int:
    args = docopt(__doc__, argv=argv)
    if args["--help"]:
        return 0

    log_level = args["--log-level"]
    if log_level:
        _configure_logging(log_level)

    switch_type = args["--switch-type"] or NVIDIA_SWITCH_TYPE
    if log_level:
        logger.debug(f"switch-type={switch_type} hostname={args['--hostname']}")

    switch = create_switch(switch_type, args["--hostname"], args["--username"], resolve_password(args))
    print_switch_status(switch.status, switch.ports, switch.vlans)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
