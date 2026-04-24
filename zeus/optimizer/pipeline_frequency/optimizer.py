"""Pipeline frequency optimizer implementation.

The `PipelineFrequencyOptimizer` is to be integrated into the training framework.
It is responsible for communicating with the PFO server and managing
the `FrequencyController` instance, which is responsible for controlling
the frequency of the CPU of the current process.
"""

from __future__ import annotations

import httpx
import torch
import torch.distributed as dist

from zeus.callback import Callback
from zeus.device import get_gpus
from zeus.optimizer.pipeline_frequency.frequency_controller import FrequencyController
from zeus.optimizer.pipeline_frequency.common import (
    GET_FREQUENCY_SCHEDULE_URL,
    REGISTER_JOB_URL,
    REGISTER_RANK_URL,
    JobInfo,
    RankInfo,
    FrequencySchedule,
)
from zeus.utils.framework import sync_execution


class PipelineFrequencyOptimizer(Callback):
    """Pipeline frequency optimizer."""

    def __init__(
        self,
        rank: int,
        dp_rank: int,
        pp_rank: int,
        tp_rank: int,
        device_id: int,
        dp_degree: int,
        pp_degree: int,
        tp_degree: int,
        world_size: int,
        server_url: str,
        job_metadata: str | None = None,
    ) -> None:
        """Initialize the Pipeline frequency optimizer.

        Assumptions:
            - `torch.distributed` has been initialized.
            - `torch.cuda.set_device` has been called with `device_id`.
                This is needed to broadcast the job ID to all ranks.

        The master process (rank 0) will register the job with the Peresus
        server and retrieve the job ID of this job. Then, each rank will
        report itself to the PFO server with the job ID.

        Args:
            rank: Global rank of the current process.
            dp_rank: Rank in the data parallel group.
            pp_rank: Rank in the pipeline parallel group.
            tp_rank: Rank in the tensor parallel group.
            device_id: CUDA device ID that the current process manages.
            dp_degree: Size of the data parallel group.
            pp_degree: Size of the pipeline parallel group.
            tp_degree: Size of the tensor parallel group.
            world_size: Total number of ranks that participate in training.
            server_url: URL of the PFO server.
            job_metadata: An optional arbitrary string that describes the job. This will
                be appended to the job ID if given. Typically for logging purposes.
        """
        raise NotImplementedError

    def _get_frequency_schedule(self) -> list[tuple[str, int]]:
        """Get the frequency schedule from the PFO server."""
        raise NotImplementedError

    def on_step_begin(self) -> None:
        """Mark the beginning of a step.

        TODO(jaywonchung): InstructionProfiler iteration start mark.
        """
        pass

    def on_step_end(self) -> None:
        """Mark the end of a step.

        TODO(jaywonchung): InstructionProfiler iteration end mark.
        Also report the profiling result to the PFO server after N iterations.
        """
        pass

    def on_instruction_begin(self, name: str) -> None:
        """Mark the beginning of an instruction, like forward and backward.

        Retrieve the next frequency from the schedule, check whether the next
        expected instruction matches the name of the instruction, and set the
        frequency accordingly.
        """
        pass

    def on_instruction_end(self, name: str) -> None:
        """Mark the end of an instruction, like forward and backward."""
