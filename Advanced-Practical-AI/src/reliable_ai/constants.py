import numpy as np

RANDOM_STATE = 42

KEYS = ["region_id", "year", "month"]

CLASS_NAMES = np.array(["LOW", "MEDIUM", "HIGH"])
LABEL_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}

RAW_FEATURES = [
    "temperature_c",
    "rainfall_mm",
    "humidity_pct",
    "soil_moisture",
    "soil_ph",
    "vegetation_index",
    "disease_cases",
    "pest_reports",
    "irrigation_capacity",
    "vulnerability_index",
]

ENGINEERED_FEATURES = [
    "drought_index",
    "heat_stress",
    "disease_pressure",
    "water_balance",
    "season_sin",
    "season_cos",
]

MODEL_FEATURES = RAW_FEATURES + ENGINEERED_FEATURES

MONITOR_FEATURES = list(RAW_FEATURES)

PROHIBITED_FEATURES = [
    "future_loss_index",
    "response_action_code",
]

VALID_RANGES = {
    "temperature_c": (-15.0, 60.0),
    "rainfall_mm": (0.0, 500.0),
    "humidity_pct": (0.0, 100.0),
    "soil_moisture": (0.0, 1.0),
    "soil_ph": (3.0, 10.0),
    "vegetation_index": (0.0, 1.0),
    "disease_cases": (0.0, 5000.0),
    "pest_reports": (0.0, 1000.0),
    "irrigation_capacity": (0.0, 1.0),
    "vulnerability_index": (0.0, 1.0),
}
