import numpy as np
import pandas as pd

from .constants import KEYS, RAW_FEATURES, MODEL_FEATURES


def clip01(values: np.ndarray) -> np.ndarray:
    return np.clip(values, 0.0, 1.0)


def engineer_features(result: pd.DataFrame) -> pd.DataFrame:
    result["drought_index"] = clip01(1.0 - result["rainfall_mm"] / 115.0)

    result["heat_stress"] = clip01(
        (result["temperature_c"] - 27.0) / 12.0
    )

    result["disease_pressure"] = clip01(
        result["disease_cases"] / 115.0
    )

    result["water_balance"] = clip01(
        0.55 * result["soil_moisture"]
        + 0.30 * result["irrigation_capacity"]
        + 0.15 * (1.0 - result["drought_index"])
    )

    angle = 2.0 * np.pi * (result["month"] - 1) / 12.0
    result["season_sin"] = np.sin(angle)
    result["season_cos"] = np.cos(angle)

    return result


def build_feature_row(measurements: dict) -> pd.DataFrame:
    row = {k: measurements.get(k, np.nan) for k in MODEL_FEATURES}
    for key in KEYS:
        if key in measurements:
            row[key] = measurements[key]
    row["month"] = measurements.get("month", 1)
    if "region_id" not in row:
        row["region_id"] = "R01"
    if "year" not in row:
        row["year"] = 2025
    temp = pd.DataFrame([row])
    temp = engineer_features(temp)
    return temp[MODEL_FEATURES]
