"""Zeusd client library.

Provides `ZeusdConfig` and `ZeusdClient`, the entry points for
communicating with a Zeusd daemon.  Handles connection, discovery,
authentication, and exposes typed methods for every Zeusd endpoint.

Typical usage:

```python
from zeus.utils.zeusd import ZeusdConfig, ZeusdClient

client = ZeusdClient(ZeusdConfig.uds(socket_path="/var/run/zeusd.sock"))
print(client.gpu_ids)       # [0, 1, 2, 3]
print(client.can_read_gpu)  # True

snapshot = client.get_gpu_power()
print(snapshot.power_mw)    # {0: 75000, 1: 120000, ...}
```
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import httpx

from zeus.exception import ZeusBaseError

logger = logging.getLogger(__name__)


class ZeusdConnectionError(ZeusBaseError):
    """Cannot reach the Zeusd daemon."""


class ZeusdAuthError(ZeusBaseError):
    """Authentication or authorization failure."""


class ZeusdCapabilityError(ZeusBaseError):
    """Requested capabilities exceed what the daemon offers."""


@dataclass(frozen=True)
class GpuPowerSnapshot:
    """Instantaneous GPU power readings from the daemon.

    Attributes:
        timestamp_ms: Daemon-side Unix timestamp in milliseconds.
        power_mw: Mapping of GPU index to power draw in milliwatts.
    """

    timestamp_ms: int
    power_mw: dict[int, int]


@dataclass(frozen=True)
class CpuDramPower:
    """Power reading for a single CPU package.

    Attributes:
        cpu_mw: CPU package power in milliwatts.
        dram_mw: DRAM power in milliwatts, or None if unavailable.
    """

    cpu_mw: int
    dram_mw: int | None


@dataclass(frozen=True)
class CpuPowerSnapshot:
    """Instantaneous CPU power readings from the daemon.

    Attributes:
        timestamp_ms: Daemon-side Unix timestamp in milliseconds.
        power_mw: Mapping of CPU index to power readings.
    """

    timestamp_ms: int
    power_mw: dict[int, CpuDramPower]


@dataclass(frozen=True)
class CpuEnergyResult:
    """Cumulative energy for a single CPU package.

    Attributes:
        cpu_energy_uj: CPU package energy in microjoules, or None.
        dram_energy_uj: DRAM energy in microjoules, or None.
    """

    cpu_energy_uj: int | None
    dram_energy_uj: int | None


@dataclass(frozen=True)
class ZeusdConfig:
    """Connection configuration for a Zeusd daemon.

    Use the classmethods `tcp`, `uds`, or `from_env` to construct.

    Attributes:
        host_port: `host:port` string (TCP mode). None for UDS.
        socket_path: Unix domain socket path (UDS mode). None for TCP.
        token: JWT token. Falls back to `ZEUSD_TOKEN` env var.
        gpu_indices: GPU indices to stream (for `PowerStreamingClient`).
            None means all, empty list means skip. Ignored by `ZeusdClient`.
        cpu_indices: CPU indices to stream (for `PowerStreamingClient`).
            None means all, empty list means skip. Ignored by `ZeusdClient`.
    """

    host_port: str | None = None
    socket_path: str | None = None
    token: str | None = None
    gpu_indices: list[int] | None = None
    cpu_indices: list[int] | None = None

    @classmethod
    def tcp(
        cls,
        host: str,
        port: int,
        *,
        token: str | None = None,
        gpu_indices: list[int] | None = None,
        cpu_indices: list[int] | None = None,
    ) -> ZeusdConfig:
        """Create a TCP connection config.

        Args:
            host: Hostname or IP of the Zeusd instance.
            port: TCP port.
            token: JWT token. Falls back to `ZEUSD_TOKEN` env var.
            gpu_indices: GPU indices to stream (for `PowerStreamingClient`).
            cpu_indices: CPU indices to stream (for `PowerStreamingClient`).
        """
        pass

    @classmethod
    def uds(
        cls,
        socket_path: str,
        *,
        token: str | None = None,
        gpu_indices: list[int] | None = None,
        cpu_indices: list[int] | None = None,
    ) -> ZeusdConfig:
        """Create a Unix domain socket connection config.

        Args:
            socket_path: Path to the Zeusd Unix domain socket.
            token: JWT token. Falls back to `ZEUSD_TOKEN` env var.
            gpu_indices: GPU indices to stream (for `PowerStreamingClient`).
            cpu_indices: CPU indices to stream (for `PowerStreamingClient`).
        """
        raise NotImplementedError

    @classmethod
    def from_env(cls) -> ZeusdConfig | None:
        """Create from environment variables.

        Tries `ZEUSD_SOCK_PATH` (UDS) first, then `ZEUSD_HOST_PORT` (TCP).
        `ZEUSD_HOST_PORT` should be `host:port`. `ZEUSD_TOKEN` is read for
        JWT authentication.

        Returns None if neither env var is set.
        """
        raise NotImplementedError

    @property
    def _is_uds(self) -> bool:
        pass

    def make_client(self) -> httpx.Client:
        """Create an httpx.Client with the appropriate transport and auth."""
        raise NotImplementedError

    def url(self, path: str) -> str:
        """Build the full URL for the given path."""
        raise NotImplementedError

    @property
    def endpoint(self) -> str:
        """Human-readable identifier for this connection."""
        pass

    def _auth_headers(self) -> dict[str, str]:
        raise NotImplementedError


class ZeusdClient:
    """Authenticated client for a Zeusd daemon.

    Handles connection, service discovery, and JWT authentication in
    one place.  Provides typed methods for every Zeusd endpoint.

    Args:
        config: Connection configuration.  If None, tries environment
            variables: `ZEUSD_SOCK_PATH` (UDS) first, then
            `ZEUSD_HOST_PORT` (TCP).

    Raises:
        ZeusdConnectionError: If the daemon is unreachable.
    """

    def __init__(self, config: ZeusdConfig | None = None) -> None:
        """Initialize the client, run discovery, and attempt authentication."""
        raise NotImplementedError

    @property
    def endpoint(self) -> str:
        """Human-readable identifier for this connection."""
        pass

    @property
    def gpu_ids(self) -> list[int]:
        """GPU device indices available on this daemon."""
        pass

    @property
    def cpu_ids(self) -> list[int]:
        """CPU device indices available on this daemon."""
        pass

    @property
    def dram_available(self) -> list[bool]:
        """Per-CPU DRAM energy availability, aligned with `cpu_ids`."""
        pass

    @property
    def auth_required(self) -> bool:
        """Whether this daemon requires JWT authentication."""
        pass

    @property
    def auth_error(self) -> str | None:
        """Auth error message, or None if auth succeeded or is not required."""
        pass

    @property
    def granted_scopes(self) -> frozenset[str]:
        """Scopes granted by the current token (empty if auth is off or failed)."""
        pass

    def _can(self, api_group: str, scope: str) -> bool:
        pass

    @property
    def can_read_gpu(self) -> bool:
        """Whether GPU read endpoints are accessible."""
        pass

    @property
    def can_control_gpu(self) -> bool:
        """Whether GPU control endpoints are accessible."""
        pass

    @property
    def can_read_cpu(self) -> bool:
        """Whether CPU read endpoints are accessible."""
        pass

    def get_gpu_energy(self, gpu_ids: list[int]) -> dict[int, int]:
        """Get cumulative energy consumption per GPU.

        Args:
            gpu_ids: GPU indices to query.

        Returns:
            Mapping of GPU index to cumulative energy in millijoules.
        """
        pass

    def get_gpu_power(self, gpu_ids: list[int] | None = None) -> GpuPowerSnapshot:
        """Get instantaneous GPU power readings.

        Args:
            gpu_ids: GPU indices to query.  None means all.

        Returns:
            Snapshot with timestamp and per-GPU power in milliwatts.
        """
        pass

    def set_power_limit(self, gpu_ids: list[int], power_limit_mw: int, block: bool = True) -> None:
        """Set the power management limit for the given GPUs."""
        raise NotImplementedError

    def set_persistence_mode(self, gpu_ids: list[int], enabled: bool, block: bool = True) -> None:
        """Set persistence mode for the given GPUs."""
        raise NotImplementedError

    def set_gpu_locked_clocks(
        self,
        gpu_ids: list[int],
        min_clock_mhz: int,
        max_clock_mhz: int,
        block: bool = True,
    ) -> None:
        """Lock the GPU clock to a specified range (MHz)."""
        pass

    def reset_gpu_locked_clocks(self, gpu_ids: list[int], block: bool = True) -> None:
        """Reset locked GPU clocks to the default."""
        pass

    def set_mem_locked_clocks(
        self,
        gpu_ids: list[int],
        min_clock_mhz: int,
        max_clock_mhz: int,
        block: bool = True,
    ) -> None:
        """Lock the memory clock to a specified range (MHz)."""
        pass

    def reset_mem_locked_clocks(self, gpu_ids: list[int], block: bool = True) -> None:
        """Reset locked memory clocks to the default."""
        pass

    def get_cpu_energy(
        self,
        cpu_ids: list[int],
        cpu: bool = True,
        dram: bool = True,
    ) -> dict[int, CpuEnergyResult]:
        """Get cumulative energy consumption per CPU.

        Args:
            cpu_ids: CPU indices to query.
            cpu: Whether to include CPU package energy.
            dram: Whether to include DRAM energy.

        Returns:
            Mapping of CPU index to energy results.
        """
        raise NotImplementedError

    def get_cpu_power(self, cpu_ids: list[int] | None = None) -> CpuPowerSnapshot:
        """Get instantaneous CPU power readings.

        Args:
            cpu_ids: CPU indices to query.  None means all.

        Returns:
            Snapshot with timestamp and per-CPU power in milliwatts.
        """
        pass

    def get_time(self) -> float:
        """Get daemon timestamp in seconds."""
        raise NotImplementedError

    def make_client(self) -> httpx.Client:
        """Create a new httpx.Client with this client's transport and auth.

        Used by `PowerStreamingClient` for SSE streaming connections
        where a dedicated, long-lived httpx.Client is needed.
        """
        raise NotImplementedError

    def url(self, path: str) -> str:
        """Build the full URL for the given path.

        Used together with `make_client()` for streaming URLs.
        """
        raise NotImplementedError

    @staticmethod
    def _check(resp: httpx.Response, operation: str) -> None:
        """Raise ZeusdError if the response is not 200."""
        raise NotImplementedError


def require_capabilities(
    client: ZeusdClient,
    *,
    read_gpu: bool = False,
    control_gpu: bool = False,
    read_cpu: bool = False,
    gpu_ids: list[int] | None = None,
    cpu_ids: list[int] | None = None,
) -> None:
    """Fail-fast validation that the daemon supports what the caller needs.

    Checks that the required API groups are enabled, the required scopes
    are granted by the token, and that the requested device IDs are
    available on the daemon.

    Args:
        client: The ZeusdClient to validate against.
        read_gpu: Require the gpu-read capability.
        control_gpu: Require the gpu-control capability.
        read_cpu: Require the cpu-read capability.
        gpu_ids: GPU indices that must be available.
        cpu_ids: CPU indices that must be available.

    Raises:
        ZeusdCapabilityError: If any requirement is not met.
    """
    raise NotImplementedError


def _capability_reason(client: ZeusdClient, scope: str) -> str:
    """Build a human-readable reason why a capability is unavailable."""
    raise NotImplementedError
