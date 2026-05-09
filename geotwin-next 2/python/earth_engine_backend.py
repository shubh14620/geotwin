#!/usr/bin/env python3
"""Google Earth Engine backend bootstrap for GeoTwin.

This script is intentionally isolated from the Next.js frontend runtime.
Use it in a separate FastAPI/Cloud Run worker when production credentials are available.
"""

from __future__ import annotations

import os
from typing import Dict

try:
    import ee  # type: ignore
except Exception:  # pragma: no cover
    ee = None


def initialize_earth_engine() -> Dict[str, str]:
    if ee is None:
        return {"ready": "false", "reason": "earthengine-api package not installed"}

    project_id = os.getenv("EE_PROJECT_ID")
    service_account = os.getenv("EE_SERVICE_ACCOUNT_EMAIL")
    private_key_file = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

    if not (project_id and service_account and private_key_file):
        return {"ready": "false", "reason": "missing EE_PROJECT_ID / EE_SERVICE_ACCOUNT_EMAIL / GOOGLE_APPLICATION_CREDENTIALS"}

    credentials = ee.ServiceAccountCredentials(service_account, private_key_file)
    ee.Initialize(credentials, project=project_id)
    return {"ready": "true", "project": project_id, "service_account": service_account}


if __name__ == "__main__":
    print(initialize_earth_engine())
