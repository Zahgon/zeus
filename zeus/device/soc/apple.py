"""Apple Silicon SoC's."""

from __future__ import annotations

import sys
import platform
from dataclasses import dataclass, asdict, fields
from functools import lru_cache

from zeus.device.soc.common import SoC, SoCMeasurement, ZeusSoCInitError

try:
    import zeus_apple_silicon  # type: ignore

    zeus_apple_available = True

except Exception:

    class MockZeusAppleSilicon:
        """Mock class for zeus-apple-silicon library."""

        def __getattr__(self, name):
            """Raise an error if any method is called.

            Since this class is only used when `zeus-apple-silicon` is not
            available, something has gone wrong if any method is called.
            """
            raise NotImplementedError

    zeus_apple_available = False
    zeus_apple_silicon = MockZeusAppleSilicon()


@lru_cache(maxsize=1)
def apple_silicon_is_available() -> bool:
    """Check if Apple silicon is available."""
    raise NotImplementedError


class ZeusAppleInitError(ZeusSoCInitError):
    """Import error for Apple SoC initialization failures."""

    def __init__(self, message: str) -> None:
        """Initialize Zeus Exception."""
        raise NotImplementedError


@dataclass
class AppleSiliconMeasurement(SoCMeasurement):
    """Represents energy consumption of various subsystems on an Apple processor.

    All measurements are in mJ. Fields that are unavailable on the current
    processor will be `None`.

    Attributes:
        cpu_total_mj: Total energy consumed by all CPU subsystems combined.
        efficiency_cores_mj: Per-core energy for each efficiency core.
        performance_cores_mj: Per-core energy for each performance core.
        efficiency_cluster_mj: Per-cluster energy for efficiency core clusters,
            including shared resources like L2 cache. More representative of
            end-to-end energy than the sum of individual cores.
        performance_cluster_mj: Per-cluster energy for performance core clusters,
            including shared resources like L2 cache. More representative of
            end-to-end energy than the sum of individual cores.
        efficiency_core_manager_mj: Energy for efficiency core cluster management.
        performance_core_manager_mj: Energy for performance core cluster management.
        dram_mj: Energy consumed by DRAM.
        gpu_mj: Energy consumed by the on-chip GPU.
        gpu_sram_mj: Energy consumed by GPU SRAM.
        ane_mj: Energy consumed by the Apple Neural Engine.
    """

    cpu_total_mj: int | None = None
    efficiency_cores_mj: list[int] | None = None
    performance_cores_mj: list[int] | None = None
    efficiency_cluster_mj: list[int] | None = None
    performance_cluster_mj: list[int] | None = None
    efficiency_core_manager_mj: int | None = None
    performance_core_manager_mj: int | None = None
    dram_mj: int | None = None
    gpu_mj: int | None = None
    gpu_sram_mj: int | None = None
    ane_mj: int | None = None

    def __sub__(self, other: AppleSiliconMeasurement) -> AppleSiliconMeasurement:
        """Produce a single measurement object containing differences across all fields."""
        raise NotImplementedError

    def zero_all_fields(self) -> None:
        """Set the value of all fields in the measurement object to zero."""
        raise NotImplementedError

    @classmethod
    def from_metrics(
        cls,
        metrics: zeus_apple_silicon.AppleEnergyMetrics,
    ) -> AppleSiliconMeasurement:
        """Return an AppleSiliconMeasurement object based on an AppleEnergyMetrics object."""
        raise NotImplementedError


class AppleSilicon(SoC):
    """An interface for obtaining energy metrics of an Apple processor."""

    def __init__(self) -> None:
        """Initialize an instance of an Apple Silicon energy monitor."""
        raise NotImplementedError

    def get_available_metrics(self) -> set[str]:
        """Return a set of all observable metrics on the current processor."""
        raise NotImplementedError

    def get_total_energy_consumption(self) -> AppleSiliconMeasurement:
        """Returns the total energy consumption of the SoC.

        The measurement should be cumulative; different calls to this function throughout
        the lifetime of a single `SoC` manager object should count from a fixed arbitrary
        point in time.

        Units: mJ.
        """
        raise NotImplementedError

    def begin_window(self, key: str, restart: bool = False) -> None:
        """Begin a measurement interval labeled with `key`.

        Args:
            key: Unique name of the measurement window.
            restart: If True and the window already exists, cancel the existing
                window and start a new one.
        """
        raise NotImplementedError

    def end_window(self, key: str) -> AppleSiliconMeasurement:
        """End a measurement window and return the energy consumption. Units: mJ."""
        raise NotImplementedError
