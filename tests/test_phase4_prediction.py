"""Phase 4 correctness tests — model loading + prediction pipeline.

Tests cover:
  - Model loads from config path and exposes predict_proba
  - predict_risk returns a float in [0, 1]
  - Full process_observation pipeline (uses in-memory SQLite — no PostgreSQL needed)
"""

from __future__ import annotations

import pytest
import joblib
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from Backend.config import settings, FEATURE_NAMES


# ── Model loading ────────────────────────────────────────────────────────────

class TestModelLoading:
    """The frozen HGB model loads from the config-based path."""

    def test_model_file_exists(self):
        """Model artifact exists at settings.model_path."""
        assert settings.model_path.exists(), (
            f"Model not found at {settings.model_path}"
        )

    def test_model_loads_via_joblib(self):
        """joblib.load produces a valid estimator."""
        model = joblib.load(settings.model_path)
        assert hasattr(model, "predict_proba")
        assert hasattr(model, "feature_names_in_")

    def test_model_feature_names_match_contract(self):
        """Loaded model's 50 features match the frozen contract."""
        model = joblib.load(settings.model_path)
        assert list(model.feature_names_in_) == list(FEATURE_NAMES)

    def test_model_classes(self):
        """Binary classifier: classes are [0, 1]."""
        model = joblib.load(settings.model_path)
        assert list(model.classes_) == [0, 1]


# ── predict_risk ─────────────────────────────────────────────────────────────

class TestPredictRisk:
    """predict_risk wraps model.predict_proba and returns a float."""

    @pytest.fixture(scope="class")
    def model(self):
        return joblib.load(settings.model_path)

    @pytest.fixture(scope="class")
    def valid_feature_row(self, model):
        """A synthetic 50-feature row (all zeros except ICULOS=1, Age=50)."""
        import pandas as pd
        row = pd.DataFrame(
            [[0.0] * 50],
            columns=list(model.feature_names_in_),
        )
        # Set raw features that make physiological sense
        row["HR"] = 84.0
        row["O2Sat"] = 98.0
        row["SBP"] = 118.0
        row["MAP"] = 77.0
        row["Resp"] = 18.0
        row["Temp"] = 36.94
        row["Age"] = 50
        row["ICULOS"] = 1
        return row

    def test_returns_float(self, model, valid_feature_row):
        """predict_risk returns a plain Python float."""
        from Backend.Services.pred_cache import predict_risk
        result = predict_risk(model, valid_feature_row)
        assert isinstance(result, float)

    def test_result_in_unit_interval(self, model, valid_feature_row):
        """Probability is in [0, 1]."""
        from Backend.Services.pred_cache import predict_risk
        result = predict_risk(model, valid_feature_row)
        assert 0.0 <= result <= 1.0

    def test_all_nan_labs_row(self, model):
        """A row with NaN labs (features 6-9) should still produce a valid probability."""
        import numpy as np
        import pandas as pd
        row = pd.DataFrame(
            [[np.nan] * 50],
            columns=list(model.feature_names_in_),
        )
        row["HR"] = 84.0
        row["O2Sat"] = 98.0
        row["SBP"] = 118.0
        row["MAP"] = 77.0
        row["Resp"] = 18.0
        row["Temp"] = 36.94
        row["Age"] = 50
        row["ICULOS"] = 1
        from Backend.Services.pred_cache import predict_risk
        result = predict_risk(model, row)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0


# ── process_observation — DB pipeline (in-memory SQLite) ─────────────────────
#
# The production upsert functions in operations.py use
# sqlalchemy.dialects.postgresql.insert().on_conflict_do_update(), which
# SQLAlchemy compiles to standard "INSERT ... ON CONFLICT ... DO UPDATE" SQL.
# SQLite 3.24+ supports this syntax natively, so all production operations
# work against an in-memory SQLite database with zero code changes.

class TestProcessObservation:
    """Full pipeline: ingest → features → inference → persist prediction.

    Uses an in-memory SQLite database — no PostgreSQL required.
    """

    @pytest.fixture(scope="class")
    def model(self):
        return joblib.load(settings.model_path)

    @pytest.fixture()
    def db_session(self):
        """Yield a session backed by an in-memory SQLite database."""
        from Backend.Database.schema import Base

        test_engine = create_engine("sqlite:///:memory:", echo=False)
        Base.metadata.create_all(bind=test_engine)

        TestSession = sessionmaker(bind=test_engine, autocommit=False, autoflush=False)
        session = TestSession()
        yield session
        session.rollback()
        session.close()
        test_engine.dispose()

    def _make_obs(self, patient_id: str, iculos: int) -> dict:
        """Synthetic observation dict."""
        return {
            "PatientID": patient_id,
            "Age": 65,
            "ICULOS": iculos,
            "HR": 84.0 + iculos,
            "O2Sat": 98.0,
            "SBP": 118.0,
            "MAP": 77.0,
            "Resp": 18.0,
            "Temp": 36.94,
            "Lactate": None,
            "WBC": None,
            "Creatinine": None,
            "Platelets": None,
        }

    def test_process_observation_returns_dict(self, db_session, model):
        """process_observation returns a dict with the expected keys."""
        from Backend.Services.pred_cache import process_observation
        result = process_observation(db_session, self._make_obs("P-T4-1", 1), model)
        assert isinstance(result, dict)
        assert set(result.keys()) == {
            "patient_id", "iculos", "raw_probability",
            "filtered_probability", "high_risk", "alert",
        }

    def test_raw_probability_is_float_in_range(self, db_session, model):
        """raw_probability is a float in [0, 1]."""
        from Backend.Services.pred_cache import process_observation
        result = process_observation(db_session, self._make_obs("P-T4-2", 1), model)
        assert isinstance(result["raw_probability"], float)
        assert 0.0 <= result["raw_probability"] <= 1.0

    def test_temporary_pass_through_fields(self, db_session, model):
        """Phase 5 placeholder fields: filtered = raw, alert = False."""
        from Backend.Services.pred_cache import process_observation
        result = process_observation(db_session, self._make_obs("P-T4-3", 1), model)
        assert result["filtered_probability"] == result["raw_probability"]
        assert result["alert"] is False

    def test_high_risk_uses_threshold(self, db_session, model):
        """high_risk = (filtered_probability >= 0.045) — temporary placeholder."""
        from Backend.Services.pred_cache import process_observation
        result = process_observation(db_session, self._make_obs("P-T4-4", 1), model)
        expected_high_risk = result["filtered_probability"] >= 0.045
        assert result["high_risk"] == expected_high_risk

    def test_prediction_persisted_in_db(self, db_session, model):
        """The prediction is persisted and retrievable from the DB."""
        from Backend.Services.pred_cache import process_observation
        from Backend.Database.operations import get_patient_predictions

        patient_id = "P-T4-5"
        result = process_observation(db_session, self._make_obs(patient_id, 1), model)

        preds = get_patient_predictions(db_session, patient_id)
        assert len(preds) == 1
        pred = preds[0]
        assert pred.patient_id == patient_id
        assert pred.iculos == 1
        assert pred.raw_probability == pytest.approx(result["raw_probability"])

    def test_persisted_raw_probability_matches_return(self, db_session, model):
        """Persisted raw_probability equals the returned raw_probability."""
        from Backend.Services.pred_cache import process_observation
        from Backend.Database.operations import get_patient_predictions

        patient_id = "P-T4-6"
        result = process_observation(db_session, self._make_obs(patient_id, 1), model)
        pred = get_patient_predictions(db_session, patient_id)[0]
        assert pred.raw_probability == result["raw_probability"]
        assert pred.filtered_probability == result["filtered_probability"]
        assert pred.high_risk == result["high_risk"]
        assert pred.alert == result["alert"]

    def test_multi_hour_patient(self, db_session, model):
        """Multiple hours for the same patient each produce a valid prediction."""
        from Backend.Services.pred_cache import process_observation
        from Backend.Database.operations import get_patient_predictions

        patient_id = "P-T4-7"
        for hour in range(1, 4):
            result = process_observation(
                db_session, self._make_obs(patient_id, hour), model,
            )
            assert 0.0 <= result["raw_probability"] <= 1.0

        preds = get_patient_predictions(db_session, patient_id)
        assert len(preds) == 3
        assert [p.iculos for p in preds] == [1, 2, 3]
