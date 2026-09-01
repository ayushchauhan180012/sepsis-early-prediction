"""Application configuration for the sepsis prediction backend.

Two distinct kinds of configuration live here:

1. TRACKED, deterministic model/training constants — the frozen values from
   ``docs/TRAINING_CONTRACT.md`` (vital medians, feature names, alert params).
   These are version-controlled and reproducible by design. They must NOT be
   moved to ``.env`` or treated as secrets.

2. Environment-specific settings — loaded via ``pydantic-settings`` from
   ``.env`` (git-ignored): ``DATABASE_URL``, model path override, credentials.
   No secrets are hardcoded in this module.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent
ROOT_DIR = BACKEND_DIR.parent

# ---------------------------------------------------------------------------
# Frozen training/inference contract (TRAINING_CONTRACT.md §3, §5)
# ---------------------------------------------------------------------------

VITALS = ["HR", "O2Sat", "SBP", "MAP", "Resp", "Temp"]
LABS = ["Lactate", "WBC", "Creatinine", "Platelets"]

# §3 Frozen vital medians (computed after patient-wise split and per-patient
# forward-fill on the training split only). Never recomputed during inference.
FROZEN_MEDIANS = {
    "HR": 84.0,
    "O2Sat": 98.0,
    "SBP": 118.0,
    "MAP": 77.0,
    "Resp": 18.0,
    "Temp": 36.94,
}

# §5 Feature-definition constants
BASELINE_WINDOW = 6
ROLL_WINDOW = 6
DELTA1_LOOKBACK = 1
DELTA6_LOOKBACK = 6
DENOMINATOR_EPSILON = 1

# §5 Clinical threshold flags (applied to imputed vitals)
TACHYCARDIA_HR = 100
HYPOTENSION_SBP = 90
TACHYPNEA_RESP = 22

# §4 The exact 50-feature order the model expects (model.feature_names_in_).
FEATURE_NAMES = (
    # Raw vitals + labs + demographics
    "HR", "O2Sat", "SBP", "MAP", "Resp", "Temp",
    "Lactate", "WBC", "Creatinine", "Platelets",
    "Age", "ICULOS",
    # Lab missing indicators
    "Lactate_missing", "WBC_missing", "Creatinine_missing", "Platelets_missing",
    # Temporal: delta6
    "HR_delta6", "O2Sat_delta6", "SBP_delta6", "MAP_delta6", "Resp_delta6", "Temp_delta6",
    # Temporal: delta1
    "HR_delta1", "O2Sat_delta1", "SBP_delta1", "MAP_delta1", "Resp_delta1", "Temp_delta1",
    # Temporal: roll6_std
    "HR_roll6_std", "O2Sat_roll6_std", "SBP_roll6_std", "MAP_roll6_std",
    "Resp_roll6_std", "Temp_roll6_std",
    # Temporal: baseline_dev
    "HR_baseline_dev", "O2Sat_baseline_dev", "SBP_baseline_dev", "MAP_baseline_dev",
    "Resp_baseline_dev", "Temp_baseline_dev",
    # Lab recent_test
    "Lactate_recent_test", "WBC_recent_test", "Creatinine_recent_test", "Platelets_recent_test",
    # Clinical composites / flags
    "shock_index", "resp_o2_ratio", "map_hr_ratio",
    "tachycardia", "hypotension", "tachypnea",
)

# §7 Frozen alert contract
ALERT_PARAMS = {
    "uncertain_low": 0.035,
    "uncertain_high": 0.055,
    "threshold": 0.045,
    "persistence": 2,        # consecutive high-risk observations required
    "cooldown_hours": 3,     # ICULOS - last_alert_time >= cooldown_hours
    "last_alert_time_init": -999,
}


# ---------------------------------------------------------------------------
# Environment-specific settings (from .env, git-ignored)
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    """Deployment-specific values read from the repo root ``.env`` file.

    Defaults are safe local-development placeholders; real deployments
    override them via environment variables or ``.env``.
    """

    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = "postgresql+psycopg://localhost:5432/sepsis_db"
    model_path: Path = BACKEND_DIR / "Model" / "hgb_sepsis_model.joblib"
    notification_channel: str = "noop"


settings = Settings()
