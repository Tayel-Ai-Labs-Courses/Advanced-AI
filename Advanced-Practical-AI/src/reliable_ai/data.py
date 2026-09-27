import numpy as np
import pandas as pd
from pathlib import Path

from .constants import (
    RANDOM_STATE,
    KEYS,
    RAW_FEATURES,
    MODEL_FEATURES,
    PROHIBITED_FEATURES,
    VALID_RANGES,
)
from .features import engineer_features


def _make_region_profiles(rng: np.random.Generator) -> pd.DataFrame:
    regions = pd.DataFrame()
    regions["region_id"] = [f"R{str(i).zfill(2)}" for i in range(1, 11)]
    n = len(regions)
    regions["region_name"] = [
        "Delta", "Nile Valley", "Coastal", "Upper Egypt", "Sinai",
        "Western Desert", "Eastern Desert", "Fayoum", "Red Sea", "Matrouh",
    ][:n]
    regions["irrigation_capacity"] = np.round(rng.uniform(0.3, 0.9, n), 4)
    regions["vulnerability_index"] = np.round(rng.uniform(0.1, 0.8, n), 4)
    climate_bands = ["arid", "semi_arid", "mediterranean", "arid", "arid",
                     "hyper_arid", "arid", "semi_arid", "arid", "mediterranean"]
    regions["climate_band"] = climate_bands[:n]
    return regions


def _make_weather(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for region_id_i in range(1, 11):
        for year in range(2019, 2026):
            for month in range(1, 13):
                # Embed some seasonal and regional signal
                seasonal_temp = 5.0 * np.sin(2.0 * np.pi * (month - 1) / 12.0)
                regional_offset = (region_id_i - 5) * 1.5
                temp = float(np.round(28.0 + seasonal_temp + regional_offset + rng.normal(0, 3), 1))
                rain = float(np.round(max(0, 35.0 + 20.0 * np.sin(2.0 * np.pi * (month + 3) / 12.0)
                                          + rng.normal(0, 15)), 1))
                humid = float(np.round(np.clip(50.0 + 20.0 * np.sin(2.0 * np.pi * (month + 2) / 12.0)
                                               + rng.normal(0, 10), 5, 95), 1))
                row = {
                    "region_id": f"R{str(region_id_i).zfill(2)}",
                    "year": year,
                    "month": month,
                    "temperature_c": temp,
                    "rainfall_mm": rain,
                    "humidity_pct": humid,
                }
                rows.append(row)
    df = pd.DataFrame(rows)

    # Inject controlled defects
    dup_row = df.iloc[[300]].copy()
    df = pd.concat([df, dup_row], ignore_index=True)
    df.loc[150, "humidity_pct"] = np.nan
    df.loc[400, "rainfall_mm"] = -18.0

    return df.sort_values(["region_id", "year", "month"]).reset_index(drop=True)


def _make_soil(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for region_id_i in range(1, 11):
        for year in range(2019, 2026):
            for month in range(1, 13):
                moisture = float(np.round(np.clip(
                    0.4 + 0.2 * np.sin(2.0 * np.pi * (month + 3) / 12.0)
                    + rng.normal(0, 0.12), 0.05, 0.9), 4))
                ph = float(np.round(np.clip(
                    7.2 + (region_id_i - 5) * 0.15 + rng.normal(0, 0.5), 5.5, 9.0), 2))
                veg = float(np.round(np.clip(
                    0.5 + 0.15 * np.sin(2.0 * np.pi * (month + 2) / 12.0)
                    + rng.normal(0, 0.1), 0.05, 0.95), 4))
                row = {
                    "region_id": f"R{str(region_id_i).zfill(2)}",
                    "year": year,
                    "month": month,
                    "soil_moisture": moisture,
                    "soil_ph": ph,
                    "vegetation_index": veg,
                }
                rows.append(row)
    df = pd.DataFrame(rows)

    # One missing soil_moisture
    df.loc[200, "soil_moisture"] = np.nan
    # One impossible pH
    df.loc[500, "soil_ph"] = 14.0

    return df.sort_values(["region_id", "year", "month"]).reset_index(drop=True)


def _make_disease(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for region_id_i in range(1, 11):
        for year in range(2019, 2026):
            for month in range(1, 13):
                cases = float(np.round(max(0, 50.0 + 15.0 * np.sin(2.0 * np.pi * month / 12.0)
                                           + rng.normal(0, 15)), 1))
                pests = float(np.round(max(0, 20.0 + 8.0 * np.sin(2.0 * np.pi * (month + 1) / 12.0)
                                           + rng.normal(0, 8)), 1))
                row = {
                    "region_id": f"R{str(region_id_i).zfill(2)}",
                    "year": year,
                    "month": month,
                    "disease_cases": cases,
                    "pest_reports": pests,
                }
                rows.append(row)
    df = pd.DataFrame(rows)

    # Duplicate row
    dup_row = df.iloc[[250]].copy()
    df = pd.concat([df, dup_row], ignore_index=True)
    # One missing pest_reports
    df.loc[100, "pest_reports"] = np.nan

    return df.sort_values(["region_id", "year", "month"]).reset_index(drop=True)


def _compute_risk_score(row: dict) -> float:
    drought = np.clip(1.0 - row["rainfall_mm"] / 115.0, 0, 1)
    heat = np.clip((row["temperature_c"] - 27.0) / 12.0, 0, 1)
    disease_p = np.clip(row["disease_cases"] / 115.0, 0, 1)
    water = np.clip(
        0.55 * row["soil_moisture"] + 0.30 * row["irrigation_capacity"]
        + 0.15 * (1.0 - drought),
        0, 1
    )
    score = (
        0.30 * drought + 0.20 * heat + 0.15 * disease_p
        - 0.15 * water + 0.20 * row["vulnerability_index"]
    )
    if row.get("year", 2020) >= 2023:
        score += 0.08
    return float(np.clip(score, 0, 1))


def _make_outcomes_from_features(
    rng: np.random.Generator,
    weather: pd.DataFrame,
    soil: pd.DataFrame,
    disease: pd.DataFrame,
    regions: pd.DataFrame,
) -> pd.DataFrame:
    # Deduplicate before merging to avoid cascade
    w = weather.drop_duplicates(subset=KEYS)
    s = soil.drop_duplicates(subset=KEYS)
    d = disease.drop_duplicates(subset=KEYS)
    merged = pd.merge(w, s, on=KEYS, how="inner")
    merged = pd.merge(merged, d, on=KEYS, how="inner")
    merged = pd.merge(merged, regions, on="region_id", how="inner")

    rows = []
    for _, row in merged.iterrows():
        risk_score = _compute_risk_score(row)
        noise = rng.uniform(-0.08, 0.08)
        final_score = np.clip(risk_score + noise, 0, 1)
        if final_score < 0.42:
            label = "LOW"
        elif final_score < 0.68:
            label = "MEDIUM"
        else:
            label = "HIGH"

        future_loss = float(np.clip(
            {"LOW": rng.uniform(0.0, 0.35), "MEDIUM": rng.uniform(0.3, 0.7),
             "HIGH": rng.uniform(0.6, 0.95)}[label],
            0.0, 1.0
        ))
        action_codes = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
        rows.append({
            "region_id": row["region_id"],
            "year": row["year"],
            "month": row["month"],
            "risk_label": label,
            "future_loss_index": round(future_loss, 4),
            "response_action_code": action_codes[label],
        })
    result = pd.DataFrame(rows)
    return result.sort_values(KEYS).reset_index(drop=True)


def generate_raw_sources(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(RANDOM_STATE)
    data_dir = Path(data_dir)
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    weather = _make_weather(rng)
    soil = _make_soil(rng)
    disease = _make_disease(rng)
    regions = _make_region_profiles(rng)
    # Rerun weather/soil/disease with same seed to get aligned data for outcomes
    rng_outcomes = np.random.default_rng(RANDOM_STATE + 1)
    outcomes = _make_outcomes_from_features(rng_outcomes, weather, soil, disease, regions)

    weather.to_csv(raw_dir / "weather.csv", index=False)
    soil.to_csv(raw_dir / "soil.csv", index=False)
    disease.to_csv(raw_dir / "disease.csv", index=False)
    regions.to_csv(raw_dir / "regions.csv", index=False)
    outcomes.to_csv(raw_dir / "outcomes.csv", index=False)

    return {"weather": weather, "soil": soil, "disease": disease,
            "regions": regions, "outcomes": outcomes}


def load_sources(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    data_dir = Path(data_dir)
    raw_dir = data_dir / "raw"
    sources = {}
    for name in ["weather", "soil", "disease", "regions", "outcomes"]:
        path = raw_dir / f"{name}.csv"
        if not path.exists():
            return {}
        sources[name] = pd.read_csv(path)
    return sources


def audit_sources(sources: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, frame in sources.items():
        dup_count = (
            frame.duplicated(subset=KEYS).sum()
            if all(k in frame.columns for k in KEYS)
            else 0
        )
        rows.append({
            "Source": name,
            "Rows": len(frame),
            "Missing cells": int(frame.isna().sum().sum()),
            "Duplicate composite keys": dup_count,
        })
    return pd.DataFrame(rows)


def aggregate_duplicates(frame: pd.DataFrame, keys: list[str] | None = None) -> pd.DataFrame:
    if keys is None:
        keys = KEYS
    dup_mask = frame.duplicated(subset=keys, keep=False)
    if not dup_mask.any():
        return frame
    numeric_cols = frame.select_dtypes(include=[np.number]).columns.tolist()
    id_cols = [c for c in frame.columns if c not in numeric_cols or c in keys]
    aggs = {c: "mean" for c in numeric_cols if c not in keys and c not in id_cols}
    if not aggs:
        return frame.drop_duplicates(subset=keys)
    deduped = frame.groupby(keys, as_index=False).agg({**aggs, **{c: "first" for c in id_cols if c not in keys}})
    return deduped


def invalidate_ranges(frame: pd.DataFrame, ranges: dict | None = None) -> pd.DataFrame:
    if ranges is None:
        ranges = VALID_RANGES
    result = frame.copy()
    for col, (lo, hi) in ranges.items():
        if col in result.columns:
            mask = (result[col] < lo) | (result[col] > hi)
            result.loc[mask, col] = np.nan
    return result


def temporal_impute(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    numeric_cols = result.select_dtypes(include=[np.number]).columns.tolist()
    sort_cols = [c for c in KEYS if c in result.columns]
    result = result.sort_values(sort_cols).reset_index(drop=True)
    for col in numeric_cols:
        if col in sort_cols:
            continue
        result[col] = result.groupby("region_id", group_keys=False)[col].transform(
            lambda s: s.interpolate(method="linear", limit_direction="both")
        )
        median_val = result[col].median()
        result[col] = result[col].fillna(median_val)
    return result


def build_feature_registry() -> pd.DataFrame:
    rows = []
    for col in MODEL_FEATURES:
        rows.append({"feature": col, "role": "approved_model_feature"})
    for col in PROHIBITED_FEATURES:
        rows.append({"feature": col, "role": "prohibited_post_outcome"})
    rows.append({"feature": "risk_label", "role": "target"})
    for col in KEYS:
        rows.append({"feature": col, "role": "identifier_or_context"})
    for col in ["region_name", "climate_band"]:
        rows.append({"feature": col, "role": "identifier_or_context"})
    return pd.DataFrame(rows)


def fuse_sources(data_dir: str | Path, generate_if_missing: bool = True) -> pd.DataFrame:
    data_dir = Path(data_dir)
    sources = load_sources(data_dir)
    if not sources and generate_if_missing:
        sources = generate_raw_sources(data_dir)

    weather = sources["weather"].copy()
    weather = aggregate_duplicates(weather)
    weather = invalidate_ranges(weather)
    weather = temporal_impute(weather)

    soil = sources["soil"].copy()
    soil = aggregate_duplicates(soil)
    soil = invalidate_ranges(soil)
    soil = temporal_impute(soil)

    disease = sources["disease"].copy()
    disease = aggregate_duplicates(disease)
    disease = invalidate_ranges(disease)
    disease = temporal_impute(disease)

    regions = sources["regions"].copy()
    outcomes = sources["outcomes"].copy()

    fused = pd.merge(weather, soil, on=KEYS, how="inner", validate="one_to_one",
                     suffixes=("_weather", "_soil"))
    fused = pd.merge(fused, disease, on=KEYS, how="inner", validate="one_to_one")
    fused = pd.merge(fused, regions, on="region_id", how="inner", validate="many_to_one")
    fused = pd.merge(fused, outcomes, on=KEYS, how="inner", validate="one_to_one")

    for col in ["rainfall_mm_weather", "rainfall_mm_soil"]:
        if col in fused.columns:
            new_name = col.replace("_weather", "").replace("_soil", "")
            if new_name not in fused.columns:
                fused.rename(columns={col: new_name}, inplace=True)

    fused = fused.sort_values(KEYS).reset_index(drop=True)
    fused = engineer_features(fused)

    assert fused.duplicated(subset=KEYS).sum() == 0, "Duplicate keys remain after fusion"
    assert fused[MODEL_FEATURES].isna().sum().sum() == 0, "Missing values remain in model features"

    return fused
