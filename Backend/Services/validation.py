from pydantic import BaseModel, Field, field_validator
from typing import Annotated, Optional


class PredictionResponse(BaseModel):
    """Response model for ``POST /predict``.

    Mirrors the keys returned by ``pred_cache.process_observation``.
    Probabilities are bounded to [0, 1] per the frozen training contract.
    """

    patient_id: str
    iculos: Annotated[int, Field(ge=1)]
    raw_probability: Annotated[float, Field(ge=0.0, le=1.0)]
    filtered_probability: Annotated[float, Field(ge=0.0, le=1.0)]
    high_risk: bool
    alert: bool


class Health(BaseModel):

    #Core Vitals

    HR: Annotated[int, Field(..., ge=20, le=250, description="Heart Rate")]
    O2Sat: Annotated[int, Field(..., ge=0, le=100, description="Oxygen Saturation")]
    SBP: Annotated[int, Field(..., ge=40, le=300, description="Systolic Blood Pressure")]
    MAP: Annotated[int, Field(..., ge=20, le=250, description="Mean Arterial Pressure")]
    Resp: Annotated[int, Field(..., ge=0, le=80, description="Respiration Rate")]
    Temp: Annotated[float, Field(..., ge=25, le=45, description="Body Temperature")]

    #Lab Values

    Lactate: Annotated[Optional[float], Field(default=None, ge=0)]
    WBC: Annotated[Optional[float], Field(default=None, ge=0)]
    Creatinine: Annotated[Optional[float], Field(default=None, ge=0)]
    Platelets: Annotated[Optional[float], Field(default=None, ge=0)]

    #Patient Info

    Age: Annotated[int, Field(..., ge=0, le=120)]
    ICULOS: Annotated[int, Field(..., ge=1)]
    PatientID: Annotated[str, Field(..., description="Patient ID")]

    #Validators

    @field_validator("HR")
    @classmethod
    def validate_hr(cls, value):
        if value == 0:
            raise ValueError("Heart Rate cannot be zero.")
        return value

    @field_validator("SBP")
    @classmethod
    def validate_sbp(cls, value):
        if value == 0:
            raise ValueError("SBP cannot be zero.")
        return value

    @field_validator("MAP")
    @classmethod
    def validate_map(cls, value):
        if value == 0:
            raise ValueError("MAP cannot be zero.")
        return value

    @field_validator("Resp")
    @classmethod
    def validate_resp(cls, value):
        if value == 0:
            raise ValueError("Respiration Rate cannot be zero.")
        return value

    @field_validator("Temp")
    @classmethod
    def validate_temp(cls, value):
        if value < 30 or value > 43:
            raise ValueError("Temperature is outside a realistic human range.")
        return value