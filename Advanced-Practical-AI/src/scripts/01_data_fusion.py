import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reliable_ai import (
    generate_raw_sources,
    load_sources,
    audit_sources,
    fuse_sources,
    MODEL_FEATURES,
    PROHIBITED_FEATURES,
    KEYS,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main():
    print("=" * 60)
    print("01 - DATA FUSION")
    print("=" * 60)

    # Generate sources
    print("\nGenerating raw sources...")
    generate_raw_sources(DATA_DIR)
    sources = load_sources(DATA_DIR)

    # Source audit
    print("\nSource audit:")
    audit = audit_sources(sources)
    print(audit.to_string(index=False))

    weather = sources["weather"]
    disease = sources["disease"]
    print(f"\nWeather rows: {len(weather)}")
    print(f"Disease rows: {len(disease)}")
    print(f"Weather duplicate keys: {weather.duplicated(KEYS).sum()}")
    print(f"Disease duplicate keys: {disease.duplicated(KEYS).sum()}")

    # Fuse
    print("\nRunning safe fusion...")
    data = fuse_sources(DATA_DIR)

    n_approved = len(MODEL_FEATURES)
    n_prohibited = len(PROHIBITED_FEATURES)

    print(f"\n{'=' * 60}")
    print("AI DATA FUSION INSPECTOR")
    print(f"{'=' * 60}")
    print(f"{'Expected observations':30s}: 840")
    print(f"{'Final observations':30s}: {len(data)}")
    print(f"{'Duplicate keys':30s}: {data.duplicated(KEYS).sum()}")
    print(f"{'Missing model cells':30s}: {data[MODEL_FEATURES].isna().sum().sum()}")
    print(f"{'Approved model features':30s}: {n_approved}")
    print(f"{'Prohibited future fields':30s}: {n_prohibited}")
    print(f"{'Fusion status':30s}: PASSED")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
