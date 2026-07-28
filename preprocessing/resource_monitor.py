"""
PixelSentinel Resource Monitor

Monitors CPU, RAM and Disk usage during preprocessing.

Features
--------
✓ CPU Usage Monitoring
✓ RAM Usage Monitoring
✓ Disk Space Monitoring
✓ Automatic Wait
✓ Emergency Stop
"""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import psutil

from preprocessing.logger import get_logger

logger = get_logger(__name__)


# ==========================================================
# CONFIGURATION
# ==========================================================

CPU_WARNING = 90.0
RAM_WARNING = 85.0

CPU_CRITICAL = 98.0
RAM_CRITICAL = 95.0

AVAILABLE_RAM_CRITICAL_GB = 2.0

CHECK_INTERVAL = 30


# ==========================================================
# RESOURCE SNAPSHOT
# ==========================================================

@dataclass(slots=True)
class ResourceSnapshot:

    cpu_percent: float

    ram_percent: float

    ram_available_gb: float

    disk_percent: float

    disk_free_gb: float


# ==========================================================
# RESOURCE MONITOR
# ==========================================================

class ResourceMonitor:

    def __init__(

        self,

        check_interval: int = CHECK_INTERVAL,

    ):

        self.check_interval = check_interval

    # ------------------------------------------------------

    def snapshot(

        self,

        path: Path | str = ".",

    ) -> ResourceSnapshot:

        cpu = psutil.cpu_percent(interval=1)

        memory = psutil.virtual_memory()
        path = Path(path).resolve()
        disk = shutil.disk_usage(path)

        total_gb = disk.total / (1024 ** 3)

        free_gb = disk.free / (1024 ** 3)

        used_percent = (

            (total_gb - free_gb)

            / total_gb

        ) * 100

        return ResourceSnapshot(

            cpu_percent=cpu,

            ram_percent=memory.percent,

            ram_available_gb=memory.available / (1024 ** 3),

            disk_percent=used_percent,

            disk_free_gb=free_gb,

        )

    # ------------------------------------------------------

    def print_snapshot(

        self,

        snapshot: ResourceSnapshot,

    ) -> None:

        logger.info("=" * 70)

        logger.info("SYSTEM RESOURCE STATUS")

        logger.info("=" * 70)

        logger.info(

            "CPU Usage           : %.1f%%",

            snapshot.cpu_percent,

        )

        logger.info(

            "RAM Usage           : %.1f%%",

            snapshot.ram_percent,

        )

        logger.info(

            "Available RAM       : %.2f GB",

            snapshot.ram_available_gb,

        )

        logger.info(

            "Disk Usage          : %.1f%%",

            snapshot.disk_percent,

        )

        logger.info(

            "Free Disk Space     : %.2f GB",

            snapshot.disk_free_gb,

        )

        logger.info("=" * 70)

# ==========================================================
# SAFETY CHECKS
# ==========================================================

    def is_safe(

        self,

        snapshot: ResourceSnapshot,

    ) -> bool:

        """
        Returns True if the system is safe to continue.
        """

        return (

            snapshot.cpu_percent < CPU_WARNING

            and

            snapshot.ram_percent < RAM_WARNING

            and

            snapshot.ram_available_gb >= AVAILABLE_RAM_CRITICAL_GB

        )

    # ------------------------------------------------------

    def is_critical(

        self,

        snapshot: ResourceSnapshot,

    ) -> bool:

        """
        Returns True if the system has reached
        emergency resource limits.
        """

        return (

            snapshot.cpu_percent >= CPU_CRITICAL

            or

            snapshot.ram_percent >= RAM_CRITICAL

            or

            snapshot.ram_available_gb < AVAILABLE_RAM_CRITICAL_GB

        )

    # ------------------------------------------------------

    def wait_until_safe(

        self,

        path: Path | str = ".",

    ) -> None:

        """
        Blocks pipeline execution until
        system resources return to safe limits.
        """

        logger.info("Checking system resources...")

        while True:

            snapshot = self.snapshot(path)

            self.print_snapshot(snapshot)

            if self.is_critical(snapshot):

                logger.warning("=" * 70)

                logger.warning("CRITICAL RESOURCE USAGE DETECTED")

                logger.warning("Pipeline paused to protect training.")

                logger.warning("=" * 70)

                time.sleep(self.check_interval)

                continue

            if self.is_safe(snapshot):

                logger.info("System resources are within safe limits.")

                return

            logger.warning(

                "High system load detected."

            )

            logger.warning(

                "Waiting %d seconds before rechecking...",

                self.check_interval,

            )

            time.sleep(self.check_interval)




# ==========================================================
# PIPELINE HOOKS
# ==========================================================

    def before_stage(

        self,

        stage_name: str,

        path: Path | str = ".",

    ) -> None:

        """
        Called before every pipeline stage.
        """

        logger.info("")

        logger.info("=" * 70)

        logger.info(

            "Preparing Stage : %s",

            stage_name,

        )

        logger.info("=" * 70)

        self.wait_until_safe(path)

    # ------------------------------------------------------

    def checkpoint(

        self,

        current: int,

        total: int,

        every: int = 100,

        path: Path | str = ".",

    ) -> None:

        """
        Resource check performed every N files.

        Example
        -------
        checkpoint(i, total)
        """

        if current == 1:

            logger.info(

                "Processing %d / %d",

                current,

                total,

            )

            return

        if current % every != 0 and current != total:

            return

        logger.info("")

        logger.info(

            "Progress : %d / %d",

            current,

            total,

        )

        snapshot = self.snapshot(path)

        self.print_snapshot(snapshot)

        if self.is_safe(snapshot):

            logger.info(

                "Resources healthy. Continuing..."

            )

            return

        logger.warning(

            "System load is high."

        )

        self.wait_until_safe(path)

    # ------------------------------------------------------

    def after_stage(

        self,

        stage_name: str,

        execution_time: float,

    ) -> None:

        """
        Called after a stage finishes.
        """

        logger.info("")

        logger.info("=" * 70)

        logger.info(

            "Completed Stage : %s",

            stage_name,

        )

        logger.info(

            "Execution Time : %.2f seconds",

            execution_time,

        )

        logger.info("=" * 70)

        snapshot = self.snapshot()

        self.print_snapshot(snapshot)


# ==========================================================
# GLOBAL MONITOR INSTANCE
# ==========================================================

resource_monitor = ResourceMonitor()

# ==========================================================
# CONVENIENCE FUNCTIONS
# ==========================================================

def wait_for_resources():

    resource_monitor.wait_until_safe()


def monitor_progress(

    current: int,

    total: int,

    every: int = 100,

):

    resource_monitor.checkpoint(

        current=current,

        total=total,

        every=every,

    )


def stage_start(

    stage_name: str,

):

    resource_monitor.before_stage(

        stage_name,

    )


def stage_end(

    stage_name: str,

    execution_time: float,

):

    resource_monitor.after_stage(

        stage_name,

        execution_time,

    )