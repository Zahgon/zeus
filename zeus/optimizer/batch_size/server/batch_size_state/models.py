"""Pydantic models for Batch size/Trials/GaussianTsArmState."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from zeus.utils.pydantic_v1 import Field, root_validator, validator, BaseModel
from zeus.optimizer.batch_size.server.database.schema import (
    GaussianTsArmStateTable,
    TrialStatus,
    TrialType,
)


class BatchSizeBase(BaseModel):
    """Base model for representing batch size.

    Attributes:
        job_id (str): The ID of the job.
        batch_size (int): The size of the batch (greater than 0).
    """

    job_id: str
    batch_size: int = Field(gt=0)

    class Config:
        """Model configuration.

        Make it immutable after it's created.
        """

        frozen = True


class Trial(BatchSizeBase):
    """Pydantic model that represents Trial.

    Attributes:
        job_id (str): The ID of the job.
        batch_size (int): The size of the batch (greater than 0).
        trial_number (int): Number of trial.
        start_timestamp (datetime): Start time of trial.
        end_timestamp (datetime): End time of trial.
        type (TrialType): Type of this trial, which means in which stage this trial was executed.
        status (TrialStatus): Status of trial
        time (Optional[float]): Total time consumption of this trial.
        energy (Optional[float]): Total energy consumption of this trial.
        converged (Optional[bool]): Whether this trial is converged or not.
    """

    trial_number: int = Field(gt=0)
    start_timestamp: datetime
    end_timestamp: Optional[datetime] = Field(None)
    type: TrialType
    status: TrialStatus
    time: Optional[float] = Field(None, ge=0)
    energy: Optional[float] = Field(None, ge=0)
    converged: Optional[bool] = None

    class Config:
        """Model configuration.

        Enable instantiating model from an ORM object, and make it immutable after it's created.
        """

        orm_mode = True
        frozen = True

    @root_validator(skip_on_failure=True)
    def _validate_mab(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Validate trial.

        We are checking
            - start_timestamp <= end_timestamp
            - if status == dispatched | Failed, time/energy/converged = None
                else time/energy/converged != None
        """
        pass


class GaussianTsArmState(BatchSizeBase):
    """Model representing Gaussian Thompson Sampling arm state.

    Attributes:
        param_mean (float): Mean of the belief prior distribution.
        param_precision (float): Precision of the belief prior distribution.
        reward_precision (float): Precision (inverse variance) of the reward distribution.
        num_observations (int): How many observations we made.
    """

    param_mean: float
    param_precision: float
    reward_precision: float
    num_observations: int = Field(ge=0)

    class Config:
        """Model configuration.

        Enable instantiating model from an ORM object, and make it immutable after it's created.
        """

        orm_mode = True
        frozen = True

    def to_orm(self) -> GaussianTsArmStateTable:
        """Convert pydantic model to ORM object.

        Returns:
            GaussianTsArmState: The ORM object of Gaussian Arm State.
        """
        pass


# Helper models


class TrialResult(BatchSizeBase):
    """Model for reading the result of the trial.

    Refer to [`Trial`][zeus.optimizer.batch_size.server.batch_size_state.models.Trial] for attributes.
    """

    trial_number: int = Field(gt=0)
    status: TrialStatus
    time: float = Field(ge=0)
    energy: float = Field(ge=0)
    converged: bool

    class Config:
        """Model configuration.

        Enable instantiating model from an ORM object, and make it immutable after it's created.
        """

        orm_mode = True
        frozen = True

    @validator("status")
    def _check_state(cls, s: TrialStatus) -> TrialStatus:
        """Check if status is equal to succeeded."""
        pass


class TrialResultsPerBs(BatchSizeBase):
    """Model representing all succeeded results of trial for a given batch size.

    Attributes:
        results (list[TrialResult]): List of TrialResult per batch size.
    """

    results: list[TrialResult]

    @root_validator(skip_on_failure=True)
    def _check_explorations(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Validate if job_id and bs are consistent across all items in results."""
        pass


class ExplorationsPerJob(BaseModel):
    """Model representing all succeeded explorations we have done for a job. Immutable after it's created.

    Attributes:
        job_id (str): The ID of the job.
        explorations_per_bs (dict[int, list[Trial]]): Dictionary of "succeeded" explorations per batch size in trial_number ascending order.
    """

    job_id: str
    explorations_per_bs: dict[int, list[Trial]]  # BS -> Trials with exploration type

    class Config:
        """Model configuration.

        Make it immutable after it's created.
        """

        frozen = True

    @root_validator(skip_on_failure=True)
    def _check_explorations(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Check bs and job_id corresponds to explorations_per_bs and batch size is consistent."""
        pass
