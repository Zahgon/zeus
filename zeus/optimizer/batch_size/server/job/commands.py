"""Commands to use `JobStateRepository`."""

from __future__ import annotations

import json
from typing import Any, Optional

import numpy as np
from zeus.utils.pydantic_v1 import root_validator, validator, Field, BaseModel
from zeus.optimizer.batch_size.common import GpuConfig, JobSpecFromClient, JobParams
from zeus.optimizer.batch_size.server.database.schema import BatchSizeTable, JobTable
from zeus.optimizer.batch_size.server.job.models import Stage


class UpdateExpDefaultBs(BaseModel):
    """Parameters to update the exploration default batch size.

    Attributes:
        job_id: Job Id.
        exp_default_batch_size: new default batch size to use.
    """

    job_id: str
    exp_default_batch_size: int = Field(gt=0)


class UpdateJobStage(BaseModel):
    """Parameters to update the job stage.

    Attributes:
        job_id: Job Id.
        stage: Set it to MAB since we only go from Pruning to MAB.
    """

    job_id: str
    stage: Stage = Field(Stage.MAB, const=True)


class UpdateGeneratorState(BaseModel):
    """Parameters to update the generator state.

    Attributes:
        job_id: Job Id.
        state: Generator state.
    """

    job_id: str
    state: str

    @validator("state")
    def _validate_state(cls, state: str) -> str:
        """Validate the sanity of state."""
        pass


class UpdateJobMinCost(BaseModel):
    """Parameters to update the min training cost and corresponding batch size.

    Attributes:
        job_id: Job Id.
        min_cost: Min training cost.
        min_cost_batch_size: Corresponding batch size.
    """

    job_id: str
    min_cost: float = Field(ge=0)
    min_cost_batch_size: int = Field(gt=0)


class CreateJob(GpuConfig, JobParams):
    """Parameters to create a new job.

    Attributes:
        exp_default_batch_size: Exploration default batch size that is used during Pruning stage.
        min_cost: Min training cost observed. Initially, None.
        min_cost_batch_size: Batch size that has minimum training cost observed.
        stage: Stage of the job.
        mab_random_generator_state: Generator state if mab_seed is not None. Otherwise, None.

    For the rest of attributes, refer to `JobParams`[zeus.optimizer.batch_size.common.JobParams] and `GpuConfig`[zeus.optimizer.batch_size.common.GpuConfig]
    """

    exp_default_batch_size: int
    min_cost: None = Field(None, const=True)
    min_cost_batch_size: int
    stage: Stage = Field(Stage.Pruning, const=True)
    mab_random_generator_state: Optional[str] = None

    class Config:
        """Model configuration.

        Make it immutable after creation.
        """

        frozen = True

    @root_validator(skip_on_failure=True)
    def _validate_states(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Validate Job states.

        We are checking,
            - If mab seed and generator state is matching.
            - If default, exp_default, min batch sizes are correctly intialized.
            - If default batch size is in the list of batch sizes.
        """
        pass

    @classmethod
    def from_job_config(cls, js: JobSpecFromClient) -> "CreateJob":
        """From JobConfig, instantiate `CreateJob`.

        Initialize generator state, exp_default_batch_size, and min_cost_batch_size.
        """
        pass

    def to_orm(self) -> JobTable:
        """Convert pydantic model `CreateJob` to ORM object Job."""
        pass
