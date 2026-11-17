# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2025 Delos Data, Inc.


class SwitchAPIError(Exception):
    """Raised when an API call to a switch fails. status_code is the HTTP status, when there was one."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class SwitchNotSupportedError(SwitchAPIError, NotImplementedError):
    """Raised when the switch, or the API the driver uses, cannot do what was asked."""
