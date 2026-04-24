"""Function for getting latitude and longitude."""

import requests
import logging

logger = logging.getLogger(__name__)


def get_ip_lat_long() -> tuple[float, float]:
    """Retrieve the latitude and longitude of the current IP position."""
    raise NotImplementedError
