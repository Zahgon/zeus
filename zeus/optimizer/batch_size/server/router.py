"""Zeus batch size optimizer server FAST API router."""

import asyncio
import logging
from collections import defaultdict

from fastapi import Depends, FastAPI, Response, status
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from zeus.optimizer.batch_size.common import (
    DELETE_JOB_URL,
    GET_NEXT_BATCH_SIZE_URL,
    REGISTER_JOB_URL,
    REPORT_END_URL,
    REPORT_RESULT_URL,
    JobSpecFromClient,
    TrialId,
    ReportResponse,
    TrainingResult,
)
from zeus.optimizer.batch_size.server.config import settings
from zeus.optimizer.batch_size.server.database.db_connection import get_db_session
from zeus.optimizer.batch_size.server.exceptions import ZeusBSOServerBaseError
from zeus.optimizer.batch_size.server.optimizer import ZeusBatchSizeOptimizer
from zeus.optimizer.batch_size.server.services.service import ZeusService

# Configure logging for the server application.
logging.basicConfig(level=logging.getLevelName(settings.log_level))
logger = logging.getLogger(__name__)

app = FastAPI()

# Per-job locking to prevent any concurrent operations on the same job.
# This is fine because it's very unlikely that the same job will be
# accessed concurrently in high frequency.
JOB_LOCKS = defaultdict(asyncio.Lock)
PREFIX_LOCKS = defaultdict(asyncio.Lock)


def get_job_locks() -> defaultdict[str, asyncio.Lock]:
    """Get global job locks."""
    pass


def get_prefix_locks() -> defaultdict[str, asyncio.Lock]:
    """Get global job Id prefix locks."""
    pass


@app.post(
    REGISTER_JOB_URL,
    responses={
        200: {"description": "Job is already registered"},
        201: {"description": "Job is successfully registered"},
    },
    response_model=JobSpecFromClient,
)
async def register_job(
    job: JobSpecFromClient,
    response: Response,
    db_session: AsyncSession = Depends(get_db_session),
    prefix_locks: defaultdict[str, asyncio.Lock] = Depends(get_prefix_locks),
):
    """Endpoint for users to register a job or check if the job is registered and configuration is identical."""
    pass


@app.delete(DELETE_JOB_URL)
async def delete_job(
    job_id: str,
    db_session: AsyncSession = Depends(get_db_session),
    job_locks: defaultdict[str, asyncio.Lock] = Depends(get_job_locks),
):
    """Endpoint for users to delete a job."""
    pass


@app.patch(REPORT_END_URL)
async def end_trial(
    trial: TrialId,
    db_session: AsyncSession = Depends(get_db_session),
    job_locks: defaultdict[str, asyncio.Lock] = Depends(get_job_locks),
):
    """Endpoint for users to end the trial."""
    pass


@app.get(GET_NEXT_BATCH_SIZE_URL, response_model=TrialId)
async def predict(
    job_id: str,
    db_session: AsyncSession = Depends(get_db_session),
    job_locks: defaultdict[str, asyncio.Lock] = Depends(get_job_locks),
):
    """Endpoint for users to receive a batch size."""
    pass


@app.post(REPORT_RESULT_URL, response_model=ReportResponse)
async def report(
    result: TrainingResult,
    db_session: AsyncSession = Depends(get_db_session),
    job_locks: defaultdict[str, asyncio.Lock] = Depends(get_job_locks),
):
    """Endpoint for users to report the training result."""
    pass
