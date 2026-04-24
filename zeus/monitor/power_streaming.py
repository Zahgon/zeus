"""Stream GPU and CPU power readings from zeusd instances via SSE.

This module provides `PowerStreamingClient`, a thread-based SSE client
that connects to one or more zeusd endpoints (TCP or Unix domain socket)
and provides the latest GPU and CPU power readings in a thread-safe manner.

```python
from zeus.utils.zeusd import ZeusdConfig
from zeus.monitor.power_streaming import PowerStreamingClient

client = PowerStreamingClient(
    servers=[
        ZeusdConfig.tcp("node1", 4938, gpu_indices=[0, 1, 2, 3]),
        ZeusdConfig.tcp("node2", 4938),
    ],
)

# Snapshot (latest readings at this instant):
readings = client.get_power()  # {"node1:4938": PowerReadings(...)}

# Blocking iterator (yields on every SSE update):
for readings in client:
    print(readings)

# Async iterator:
async for readings in client:
    print(readings)

client.stop()
```
"""

from __future__ import annotations

import json
import logging
import statistics
import threading
import time
import typing
from collections.abc import AsyncIterator, Iterator, Sequence
from dataclasses import dataclass, field

import httpx

from zeus.utils.zeusd import ZeusdCapabilityError, ZeusdClient, ZeusdConfig

logger = logging.getLogger(__name__)


@dataclass
class CpuPowerReading:
    """Power reading for a single CPU package.

    Args:
        cpu_w: CPU package power in watts.
        dram_w: DRAM power in watts, or None if not available.
    """

    cpu_w: float = 0.0
    dram_w: float | None = None


@dataclass
class PowerReadings:
    """Power readings from a single zeusd endpoint.

    Args:
        timestamp_s: Unix timestamp (seconds) of the reading.
        gpu_power_w: Per-GPU power in watts, keyed by GPU index.
        cpu_power_w: Per-CPU power readings, keyed by CPU index.
    """

    timestamp_s: float = 0.0
    gpu_power_w: dict[int, float] = field(default_factory=dict)
    cpu_power_w: dict[int, CpuPowerReading] = field(default_factory=dict)


class PowerStreamingClient:
    """Connect to multiple zeusd instances and stream GPU/CPU power readings.

    One background thread per device type per endpoint maintains an SSE
    connection to the zeusd streaming endpoints. The latest power readings
    are stored in a thread-safe dict, accessible via `get_power()`.

    The client supports three access patterns:

    - Snapshot: Call `get_power()` to retrieve the latest readings at any time.
    - Blocking iterator: Use `for readings in client` to block and yield a
      snapshot each time new SSE data arrives. Iteration stops when `stop()`
      is called.
    - Async iterator: Use `async for readings in client` for the same
      behavior without blocking the event loop.

    ```python
    client = PowerStreamingClient(servers=[...])

    # Snapshot
    readings = client.get_power()

    # Blocking iterator
    for readings in client:
        print(readings)

    # Async iterator
    async for readings in client:
        print(readings)

    client.stop()
    ```

    Args:
        servers: List of `ZeusdConfig` specifying zeusd endpoints
            and which GPUs/CPUs to stream from each.
        reconnect_delay_s: Seconds to wait before reconnecting after a
            disconnect.
    """

    def __init__(
        self,
        servers: Sequence[ZeusdConfig],
        reconnect_delay_s: float = 1.0,
    ) -> None:
        """Initialize the client and start background SSE connections."""
        raise NotImplementedError

    def stop(self) -> None:
        """Stop all background connections and wake any blocked iterators."""
        pass

    def get_power(self) -> dict[str, PowerReadings]:
        """Get the latest power readings from all endpoints.

        Returns:
            Mapping of endpoint identifier to `PowerReadings` containing
            timestamp and per-GPU/CPU power in watts.
        """
        pass

    def __iter__(self) -> Iterator[dict[str, PowerReadings]]:
        """Yield power reading snapshots as they arrive from SSE streams.

        Blocks until new readings are available, then yields a snapshot
        (same format as `get_power()`). Iteration stops when `stop()` is
        called.

        ```python
        client = PowerStreamingClient(servers=[...])
        for readings in client:
            for endpoint, pr in readings.items():
                print(f"{endpoint}: {pr.gpu_power_w}")
        ```
        """
        raise NotImplementedError

    async def __aiter__(self) -> AsyncIterator[dict[str, PowerReadings]]:
        """Async version of `__iter__`.

        Yields power reading snapshots without blocking the event loop.
        Iteration stops when `stop()` is called.

        ```python
        client = PowerStreamingClient(servers=[...])
        async for readings in client:
            for endpoint, pr in readings.items():
                print(f"{endpoint}: {pr.gpu_power_w}")
        ```
        """
        raise NotImplementedError

    def _wait_for_update(self) -> bool:
        """Block until readings are updated or timeout (1 s).

        Used by `__aiter__` to avoid blocking the async event loop.
        """
        pass

    def _init_server(self, server: ZeusdConfig) -> tuple[bool, bool]:
        """Initialize connection to a server and decide what to stream.

        Creates a `ZeusdClient` for the server (handling discovery and auth),
        validates requested indices, and checks scope permissions.

        Returns:
            A `(stream_gpu, stream_cpu)` pair of booleans.

        Raises:
            ZeusdConnectionError: If the server is not reachable.
            ValueError: If explicitly requested indices are not available.
            ZeusdCapabilityError: If explicitly requested streaming requires
                a scope the token doesn't have.
        """
        raise NotImplementedError

    @staticmethod
    def _resolve_streaming(
        user_indices: list[int] | None,
        available_ids: set[int],
        has_permission: bool,
        scope_name: str,
        device_type: str,
        endpoint: str,
    ) -> bool:
        """Decide whether to stream a device type.

        Semantics:
        - `user_indices is None`: stream all available, silently skip if
          none exist or if the token lacks the scope.
        - `user_indices == []`: explicitly opt out, never stream.
        - `user_indices` is a non-empty list: require all IDs to exist
          and the scope to be granted; raise on mismatch.

        Returns:
            True if streaming should be started for this device type.

        Raises:
            ValueError: If explicit indices are not a subset of available.
            ZeusdCapabilityError: If explicit indices are given but the
                token lacks the required scope.
        """
        raise NotImplementedError

    def _estimate_clock_offset(
        self,
        endpoint: str,
        num_samples: int = 5,
    ) -> float:
        """Estimate the clock offset between this client and the daemon.

        Performs `num_samples` round-trips to `GET /time` on the daemon,
        computes `client_midpoint - daemon_time` for each, and returns the
        median offset in seconds. A positive offset means the daemon clock
        is behind the client clock.

        Args:
            endpoint: The endpoint identifier.
            num_samples: Number of round-trips for robustness.

        Returns:
            Estimated clock offset in seconds. Add this to daemon
            timestamps to align them with client time.
        """
        raise NotImplementedError

    def _gpu_stream_loop(self, server: ZeusdConfig, endpoint: str) -> None:
        """Background thread: stream GPU power from a single server."""
        pass

    def _cpu_stream_loop(self, server: ZeusdConfig, endpoint: str) -> None:
        """Background thread: stream CPU power from a single server."""
        pass

    def _stream_loop(
        self,
        url: str,
        endpoint: str,
        process_fn: typing.Callable[[str, str], None],
        label: str,
    ) -> None:
        """Shared reconnect loop for SSE streams."""
        pass

    def _connect_and_stream(
        self,
        url: str,
        endpoint: str,
        process_fn: typing.Callable[[str, str], None],
    ) -> None:
        """Open an SSE connection and process events until disconnected."""
        pass

    def _process_gpu_event(self, event_text: str, endpoint: str) -> None:
        """Parse a GPU SSE event and update readings."""
        pass

    def _process_cpu_event(self, event_text: str, endpoint: str) -> None:
        """Parse a CPU SSE event and update readings.

        Expected JSON format: `{"timestamp_ms": N, "power_mw": {"0": {"cpu_mw": N, "dram_mw": N|null}}}`.
        """
        pass
