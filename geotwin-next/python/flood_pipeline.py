#!/usr/bin/env python3
"""Simple flood-risk analysis pipeline for GeoTwin.

Usage:
  python flood_pipeline.py input.json
  cat input.json | python flood_pipeline.py

Input JSON example:
{
  "area": "Ganga Floodplain",
  "precipitation_mm": [0.2, 5.1, 11.4],
  "precipitation_probability": [20, 60, 85],
  "soil_surface": [0.21, 0.32, 0.41],
  "soil_root": [0.28, 0.37, 0.43]
}
"""

from __future__ import annotations

import json
import statistics
import sys
from dataclasses import dataclass, asdict
from typing import List


@dataclass
class FloodAssessment:
    area: str
    next_24h_rain_mm: float
    peak_hourly_rain_mm: float
    average_surface_moisture: float
    average_root_moisture: float
    flood_risk_index: float
    warning_level: str
    recommendation: str


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def classify(index: float) -> str:
    if index >= 75:
        return "Severe"
    if index >= 55:
        return "High"
    if index >= 35:
        return "Moderate"
    return "Low"


def recommend(level: str) -> str:
    mapping = {
        "Low": "Continue routine monitoring.",
        "Moderate": "Increase observation frequency and check pumps or embankments.",
        "High": "Notify district response teams and review evacuation readiness.",
        "Severe": "Issue high-priority alert and activate rapid response protocol.",
    }
    return mapping[level]


def analyze(payload: dict) -> FloodAssessment:
    precipitation = [float(x) for x in payload.get("precipitation_mm", [])]
    pop = [float(x) for x in payload.get("precipitation_probability", [])]
    soil_surface = [float(x) for x in payload.get("soil_surface", [])]
    soil_root = [float(x) for x in payload.get("soil_root", [])]

    rain_sum = sum(precipitation[:24]) if precipitation else 0.0
    peak_rain = max(precipitation) if precipitation else 0.0
    avg_surface = statistics.fmean(soil_surface) if soil_surface else 0.0
    avg_root = statistics.fmean(soil_root) if soil_root else 0.0
    avg_pop = statistics.fmean(pop[:24]) if pop else 0.0

    flood_risk_index = clamp(
        rain_sum * 2.2 + peak_rain * 4.5 + avg_surface * 35 + avg_root * 20 + avg_pop * 0.22,
        0,
        100,
    )
    warning_level = classify(flood_risk_index)

    return FloodAssessment(
        area=payload.get("area", "Unknown Area"),
        next_24h_rain_mm=round(rain_sum, 2),
        peak_hourly_rain_mm=round(peak_rain, 2),
        average_surface_moisture=round(avg_surface, 3),
        average_root_moisture=round(avg_root, 3),
        flood_risk_index=round(flood_risk_index, 1),
        warning_level=warning_level,
        recommendation=recommend(warning_level),
    )


def load_payload() -> dict:
    if len(sys.argv) > 1:
        with open(sys.argv[1], "r", encoding="utf-8") as handle:
            return json.load(handle)
    return json.load(sys.stdin)


if __name__ == "__main__":
    assessment = analyze(load_payload())
    print(json.dumps(asdict(assessment), indent=2))
