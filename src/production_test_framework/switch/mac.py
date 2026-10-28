# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2026 Delos Data, Inc.


def normalize_mac(value: str) -> str:
    """A MAC in lowercase colon form, from dotted ("001c.7300.abcd"), colon, dash or bare
    hex form; "" if it is not a MAC."""
    compact = value.replace(".", "").replace(":", "").replace("-", "").lower()
    if len(compact) != 12 or any(c not in "0123456789abcdef" for c in compact):
        return ""
    return ":".join(compact[i : i + 2] for i in range(0, 12, 2))
