import json
import shutil
import sys
from pathlib import Path

# --- Configuration ---
CANDIDATE_MODEL_PATH = Path("models/candidate_model.keras")
PRODUCTION_MODEL_PATH = Path("models/banana_mobilenet.keras")
CANDIDATE_METRICS_PATH = Path("reports/train_metrics.json")
PRODUCTION_METRICS_PATH = Path("reports/production_metrics.json")

# Sanity ceiling: historical runs cluster around 3.3-4.0 days MAE, but a
# diverged run once logged a MAE in the hundreds of millions. Anything
# past this is treated as a broken run, not a real candidate.
MAX_ACCEPTABLE_MAE = 10.0


def load_metrics(path):
    with open(path) as f:
        return json.load(f)


def promote():
    if not CANDIDATE_METRICS_PATH.exists() or not CANDIDATE_MODEL_PATH.exists():
        print(f"No candidate found at {CANDIDATE_MODEL_PATH} / {CANDIDATE_METRICS_PATH}.")
        print("Run src/train_v3.py first.")
        sys.exit(1)

    candidate_metrics = load_metrics(CANDIDATE_METRICS_PATH)
    candidate_mae = candidate_metrics["test_mae"]

    print(f"Candidate test MAE: {candidate_mae:.4f} days")

    if candidate_mae > MAX_ACCEPTABLE_MAE:
        print(f"REJECTED: candidate MAE {candidate_mae:.4f} exceeds sanity ceiling "
              f"({MAX_ACCEPTABLE_MAE} days). Likely a diverged/failed run.")
        write_result(promoted=False, candidate_mae=candidate_mae, production_mae=None,
                      reason="exceeds_sanity_ceiling")
        return

    if not PRODUCTION_METRICS_PATH.exists():
        print("No production model on record yet. Bootstrapping production with this candidate.")
        _promote_candidate(candidate_metrics)
        write_result(promoted=True, candidate_mae=candidate_mae, production_mae=None,
                      reason="bootstrap")
        return

    production_metrics = load_metrics(PRODUCTION_METRICS_PATH)
    production_mae = production_metrics["test_mae"]
    print(f"Current production test MAE: {production_mae:.4f} days")

    if candidate_mae < production_mae:
        print(f"PROMOTED: candidate ({candidate_mae:.4f}) beats production ({production_mae:.4f}).")
        _promote_candidate(candidate_metrics)
        write_result(promoted=True, candidate_mae=candidate_mae, production_mae=production_mae,
                      reason="improved")
    else:
        print(f"REJECTED: candidate ({candidate_mae:.4f}) does not beat production "
              f"({production_mae:.4f}). Production model unchanged.")
        write_result(promoted=False, candidate_mae=candidate_mae, production_mae=production_mae,
                      reason="no_improvement")


def _promote_candidate(candidate_metrics):
    shutil.copy2(CANDIDATE_MODEL_PATH, PRODUCTION_MODEL_PATH)
    with open(PRODUCTION_METRICS_PATH, "w") as f:
        json.dump(candidate_metrics, f, indent=2)
    print(f"{CANDIDATE_MODEL_PATH} -> {PRODUCTION_MODEL_PATH}")


def write_result(promoted, candidate_mae, production_mae, reason):
    Path("reports").mkdir(exist_ok=True)
    result = {
        "promoted": promoted,
        "candidate_mae": candidate_mae,
        "production_mae": production_mae,
        "reason": reason,
    }
    with open("reports/promotion_result.json", "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    promote()
