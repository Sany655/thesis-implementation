import os
import joblib
import pandas as pd
import uuid
import logging
import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, selectinload
from sqlalchemy import String, Integer, Float, DateTime, ForeignKey, select
from datetime import datetime

import shap
import xgboost as xgb

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s │ %(name)s │ %(message)s")
logger = logging.getLogger("dengue_api")

# ─── DB Setup ─────────────────────────────────────────────────────────────────
DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    if os.environ.get("VERCEL") or os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
        DATABASE_URL = "sqlite+aiosqlite:////tmp/dengue_dashboard.db"
    else:
        DATABASE_URL = "sqlite+aiosqlite:///./dengue_dashboard.db"

engine = create_async_engine(DATABASE_URL, echo=False)
async_session = async_sessionmaker(engine, expire_on_commit=False)

class Base(DeclarativeBase):
    pass

class Patient(Base):
    __tablename__ = "patient"
    patient_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    age: Mapped[float] = mapped_column(Float)
    gender: Mapped[str] = mapped_column(String(10))
    assessments: Mapped[list["LabAssessment"]] = relationship("LabAssessment", back_populates="patient", cascade="all, delete-orphan")

class LabAssessment(Base):
    __tablename__ = "lab_assessment"
    assessment_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id: Mapped[str] = mapped_column(String(36), ForeignKey("patient.patient_id"))
    assessment_time: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    wbc: Mapped[float] = mapped_column(Float)
    hct: Mapped[float] = mapped_column(Float)
    rbc: Mapped[float] = mapped_column(Float)
    lymphocyte: Mapped[float] = mapped_column(Float)
    neutrophil: Mapped[float] = mapped_column(Float)
    ast: Mapped[float] = mapped_column(Float)
    alt: Mapped[float] = mapped_column(Float)
    plt: Mapped[float] = mapped_column(Float)
    patient: Mapped["Patient"] = relationship("Patient", back_populates="assessments")
    predictions: Mapped[list["ModelPrediction"]] = relationship("ModelPrediction", back_populates="assessment", cascade="all, delete-orphan")

class ModelPrediction(Base):
    __tablename__ = "model_prediction"
    prediction_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    assessment_id: Mapped[str] = mapped_column(String(36), ForeignKey("lab_assessment.assessment_id"))
    cohort: Mapped[str] = mapped_column(String(20))
    model: Mapped[str] = mapped_column(String(50))
    predicted_class: Mapped[str] = mapped_column(String(50))
    probability: Mapped[float] = mapped_column(Float)
    assessment: Mapped["LabAssessment"] = relationship("LabAssessment", back_populates="predictions")
    explanations: Mapped[list["ShapExplanation"]] = relationship("ShapExplanation", back_populates="prediction", cascade="all, delete-orphan")

class ShapExplanation(Base):
    __tablename__ = "shap_explanation"
    shap_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    prediction_id: Mapped[str] = mapped_column(String(36), ForeignKey("model_prediction.prediction_id"))
    feature: Mapped[str] = mapped_column(String(50))
    shap_value: Mapped[float] = mapped_column(Float)
    feature_val: Mapped[float] = mapped_column(Float)
    prediction: Mapped["ModelPrediction"] = relationship("ModelPrediction", back_populates="explanations")

models = {}
explainers = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables initialized.")
    except Exception as e:
        logger.error(f"DB init failed: {e}")
        raise

    custom_model_dir = os.environ.get("MODEL_DIR")
    search_dirs = [
        custom_model_dir,
        os.path.join(os.path.dirname(__file__), 'models'),
        os.path.join(os.path.dirname(__file__), '..', 'models'),
        os.path.join(os.path.dirname(__file__), '..', '..', 'models'),
        os.path.join(os.getcwd(), 'models'),
        os.path.join(os.getcwd(), 'backend', 'models'),
    ]

    ped_path = None
    adult_path = None

    for d in search_dirs:
        if not d:
            continue
        p_cand = os.path.join(d, 'pediatric_best.pkl')
        a_cand = os.path.join(d, 'adult_best.pkl')
        if ped_path is None and os.path.exists(p_cand):
            ped_path = p_cand
        if adult_path is None and os.path.exists(a_cand):
            adult_path = a_cand

    if ped_path and os.path.exists(ped_path):
        try:
            models['Pediatric'] = joblib.load(ped_path)
            explainers['Pediatric'] = shap.TreeExplainer(models['Pediatric'])
            logger.info(f"Pediatric model loaded from: {ped_path}")
        except Exception as e:
            logger.error(f"Failed to load Pediatric model ({ped_path}): {e}")
    else:
        logger.warning("Pediatric model not found in searched directories.")

    if adult_path and os.path.exists(adult_path):
        try:
            models['Adult'] = joblib.load(adult_path)
            explainers['Adult'] = shap.TreeExplainer(models['Adult'])
            logger.info(f"Adult model loaded from: {adult_path}")
        except Exception as e:
            logger.error(f"Failed to load Adult model ({adult_path}): {e}")
    else:
        logger.warning("Adult model not found in searched directories.")

    yield

app = FastAPI(title="Dengue Risk Assessment API", lifespan=lifespan)

# ─── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Global Exception Handler ─────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error on {request.method} {request.url}: {exc}\n{traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.warning(f"HTTP {exc.status_code} on {request.method} {request.url}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

# ─── PLT Clinical Thresholds ──────────────────────────────────────────────────
# Based on the thesis target proxy (Niazi & Momand):
#   Severe dengue:     PLT < 50,000 /µL
#   Moderate dengue:   PLT 50,000–100,000 /µL
#   Minor dengue:      PLT 100,000–150,000 /µL
#   Normal (non-dengue likely): PLT > 150,000 /µL

PLT_SEVERE_THRESHOLD   = 50_000   # < this → likely severe dengue
PLT_MODERATE_THRESHOLD = 100_000  # < this → moderate
PLT_NORMAL_THRESHOLD   = 150_000  # ≥ this → platelet not indicative of dengue

def classify_plt_context(plt_val: float) -> dict:
    """
    Returns a clinical context dict based on PLT value.
    PLT is in 10³/µL (thousands per µL) as entered in the UI.
    """
    plt_abs = plt_val * 1000  # convert from ×10³/µL to /µL

    if plt_abs >= PLT_NORMAL_THRESHOLD:
        return {
            "plt_warning": True,
            "plt_level": "normal",
            "plt_message": (
                f"Platelet count ({plt_val:.0f} ×10³/µL) is within the normal range (≥150 ×10³/µL). "
                "Platelet values at this level are NOT indicative of dengue thrombocytopenia. "
                "The model was trained on confirmed dengue (NS1+) patients — results may not be meaningful "
                "for patients without dengue infection."
            ),
            "skip_model": True,
        }
    elif plt_abs >= PLT_MODERATE_THRESHOLD:
        return {
            "plt_warning": True,
            "plt_level": "minor",
            "plt_message": (
                f"Platelet count ({plt_val:.0f} ×10³/µL) is in the Minor/non-severe range (100–150 ×10³/µL). "
                "This may indicate early or mild dengue, but does not meet the Severe dengue proxy threshold (PLT < 50 ×10³/µL). "
                "Model results are shown but interpret with caution."
            ),
            "skip_model": False,
        }
    elif plt_abs >= PLT_SEVERE_THRESHOLD:
        return {
            "plt_warning": True,
            "plt_level": "moderate",
            "plt_message": (
                f"Platelet count ({plt_val:.0f} ×10³/µL) is in the Moderate range (50–100 ×10³/µL). "
                "This is consistent with moderate dengue thrombocytopenia. Model is running normally."
            ),
            "skip_model": False,
        }
    else:
        return {
            "plt_warning": False,
            "plt_level": "severe",
            "plt_message": (
                f"Platelet count ({plt_val:.0f} ×10³/µL) is in the Severe range (< 50 ×10³/µL). "
                "This is consistent with the severe dengue proxy target. Model is running normally."
            ),
            "skip_model": False,
        }

# ─── Schema ───────────────────────────────────────────────────────────────────
class PatientData(BaseModel):
    patient_id: str = "P001"
    age: float
    gender: int
    wbc: float
    hct: float
    rbc: float
    lymph: float
    neut: float
    alt: float
    ast: float
    plt: float = 0.0
    hgb: float | None = None

    @field_validator('age')
    @classmethod
    def age_must_be_positive(cls, v):
        if v <= 0 or v > 120:
            raise ValueError('Age must be between 1 and 120.')
        return v

    @field_validator('plt')
    @classmethod
    def plt_must_be_non_negative(cls, v):
        if v < 0:
            raise ValueError('Platelet count cannot be negative.')
        return v

    @field_validator('wbc', 'hct', 'rbc', 'lymph', 'neut', 'alt', 'ast')
    @classmethod
    def fields_must_be_non_negative(cls, v):
        if v < 0:
            raise ValueError('Lab values cannot be negative.')
        return v

# ─── Feature Engineering ──────────────────────────────────────────────────────
def preprocess_features(data: PatientData) -> pd.DataFrame:
    # Construct exact DataFrame matching the Zero-Leakage Scikit-Learn Pipeline
    df = pd.DataFrame([{
        'Age': data.age,
        'Gender': data.gender,
        'WBC_k': data.wbc,
        'HCT_pct': data.hct,
        'RBC': data.rbc,
        'Lymph_pct': data.lymph,
        'Neut_pct': data.neut,
        'HGB': data.hgb if data.hgb is not None else float('nan'),
        'ALT': data.alt,
        'AST': data.ast
    }])
    
    expected_cols = [
        'Age', 'Gender', 'WBC_k', 'HCT_pct', 'RBC', 
        'Lymph_pct', 'Neut_pct', 'HGB', 'ALT', 'AST'
    ]
    return df[expected_cols]

# ─── Endpoints Router ─────────────────────────────────────────────────────────
router = APIRouter()

@router.post("/predict")
async def predict_risk(data: PatientData):
    try:
        if 'Pediatric' not in models or 'Adult' not in models:
            raise HTTPException(
                status_code=503,
                detail="Models are not loaded. Ensure pediatric_best.pkl and adult_best.pkl exist in the /models directory."
            )

        # ── PLT Clinical Pre-Check ────────────────────────────────────────────
        plt_context = classify_plt_context(data.plt)

        if plt_context["skip_model"]:
            logger.info(f"Skipping model for patient {data.patient_id}: PLT={data.plt} ×10³/µL (normal range).")
            return {
                "skipped": True,
                "reason": "plt_normal",
                "plt_context": plt_context,
                "patient_id": data.patient_id,
            }

        # ── Run Model ─────────────────────────────────────────────────────────
        df_features = preprocess_features(data)
        is_pediatric  = data.age <= 18
        active_cohort = "Pediatric" if is_pediatric else "Adult"

        ped_prob   = float(models['Pediatric'].predict_proba(df_features)[0][1] * 100)
        adult_prob = float(models['Adult'].predict_proba(df_features)[0][1] * 100)

        def extract_shap(explainer, X):
            import numpy as np
            raw = explainer.shap_values(X)
            if isinstance(raw, list):
                return np.array(raw[1][0])
            arr = np.array(raw)
            if arr.ndim == 3:
                return arr[0, :, 1]
            else:
                return arr[0]

        ped_shap_values   = extract_shap(explainers['Pediatric'], df_features)
        adult_shap_values = extract_shap(explainers['Adult'], df_features)

        features_list = df_features.columns.tolist()
        feature_vals  = df_features.iloc[0].tolist()

        def format_shap(shap_vals):
            return [
                {
                    'name': features_list[i],
                    'val': round(feature_vals[i], 2),
                    'contribution': round(float(val), 3),
                    'impact': 'positive' if float(val) > 0 else 'negative',
                    'threshold': 'Model Split'
                }
                for i, val in enumerate(shap_vals)
            ]

        ped_features   = format_shap(ped_shap_values)
        adult_features = format_shap(adult_shap_values)

        # ── Persist to DB ─────────────────────────────────────────────────────
        try:
            async with async_session() as session:
                result  = await session.execute(select(Patient).where(Patient.patient_id == data.patient_id))
                patient = result.scalars().first()
                if not patient:
                    patient = Patient(
                        patient_id=data.patient_id,
                        age=data.age,
                        gender="Male" if data.gender == 0 else "Female"
                    )
                    session.add(patient)

                assessment = LabAssessment(
                    patient_id=patient.patient_id,
                    wbc=data.wbc, hct=data.hct, rbc=data.rbc,
                    lymphocyte=data.lymph, neutrophil=data.neut,
                    ast=data.ast, alt=data.alt, plt=data.plt
                )
                session.add(assessment)
                await session.flush()

                for cohort_name, prob, shap_vals in [
                    ("Pediatric", ped_prob, ped_shap_values),
                    ("Adult",     adult_prob, adult_shap_values)
                ]:
                    pred_class = "High-risk pattern" if prob > 50 else "Low-risk pattern"
                    prediction = ModelPrediction(
                        assessment_id=assessment.assessment_id,
                        cohort=cohort_name,
                        model="XGBoost",
                        predicted_class=pred_class,
                        probability=prob
                    )
                    session.add(prediction)
                    await session.flush()

                    for i, val in enumerate(shap_vals):
                        session.add(ShapExplanation(
                            prediction_id=prediction.prediction_id,
                            feature=features_list[i],
                            shap_value=float(val),
                            feature_val=float(feature_vals[i])
                        ))

                await session.commit()
        except Exception as db_err:
            logger.error(f"DB write failed for patient {data.patient_id}: {db_err}")
            raise HTTPException(status_code=500, detail=f"Assessment ran but failed to save: {str(db_err)}")

        return {
            "skipped":     False,
            "plt_context": plt_context,
            "id":          assessment.assessment_id,
            "patient_id":  patient.patient_id,
            "timestamp":   assessment.assessment_time,
            "cohort":      active_cohort,
            "activeModel": active_cohort,
            "pediatric": {
                "probability": ped_prob,
                "prediction":  "High-risk pattern" if ped_prob > 50 else "Low-risk pattern",
                "features":    ped_features
            },
            "adult": {
                "probability": adult_prob,
                "prediction":  "High-risk pattern" if adult_prob > 50 else "Low-risk pattern",
                "features":    adult_features
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in /predict: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Assessment failed: {str(e)}")


@router.get("/assessments/{patient_id}")
async def get_assessments(patient_id: str):
    try:
        async with async_session() as session:
            result = await session.execute(
                select(LabAssessment)
                .where(LabAssessment.patient_id == patient_id)
                .options(
                    selectinload(LabAssessment.predictions)
                    .selectinload(ModelPrediction.explanations)
                )
                .order_by(LabAssessment.assessment_time.asc())
            )
            assessments = result.scalars().all()

            result  = await session.execute(select(Patient).where(Patient.patient_id == patient_id))
            patient = result.scalars().first()

        if not assessments:
            return []

        is_pediatric  = patient.age <= 18 if patient else True
        active_cohort = "Pediatric" if is_pediatric else "Adult"

        formatted_assessments = []
        for i, a in enumerate(assessments):
            ped_data   = {}
            adult_data = {}

            for pred in a.predictions:
                features = [
                    {
                        "name":         exp.feature,
                        "val":          round(exp.feature_val, 2),
                        "contribution": round(exp.shap_value, 3),
                        "impact":       'positive' if exp.shap_value > 0 else 'negative',
                        "threshold":    "Model Split"
                    }
                    for exp in pred.explanations
                ]
                model_obj = {
                    "probability": pred.probability,
                    "prediction":  pred.predicted_class,
                    "features":    features
                }
                if pred.cohort == "Pediatric":
                    ped_data = model_obj
                else:
                    adult_data = model_obj

            plt_context = classify_plt_context(a.plt)

            formatted_assessments.append({
                "skipped":      False,
                "plt_context":  plt_context,
                "id":           a.assessment_id,
                "patient_id":   a.patient_id,
                "timestamp":    a.assessment_time,
                "cohort":       active_cohort,
                "activeModel":  active_cohort,
                "pediatric":    ped_data,
                "adult":        adult_data,
                "input": {
                    "patient_id": a.patient_id,
                    "age":    patient.age if patient else 0,
                    "gender": 0 if patient and patient.gender == "Male" else 1,
                    "wbc": a.wbc, "hct": a.hct, "rbc": a.rbc,
                    "lymph": a.lymphocyte, "neut": a.neutrophil,
                    "ast": a.ast, "alt": a.alt, "plt": a.plt
                },
                "snapshotName": f"Assessment {i+1} ({a.assessment_time.strftime('%Y-%m-%d')})"
            })

        return formatted_assessments

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in /assessments/{patient_id}: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch assessments: {str(e)}")


@router.get("/patients")
async def get_patients():
    try:
        async with async_session() as session:
            result = await session.execute(
                select(Patient)
                .options(selectinload(Patient.assessments))
                .order_by(Patient.patient_id.asc())
            )
            patients = result.scalars().all()

            patient_list = []
            for p in patients:
                patient_list.append({
                    "patient_id": p.patient_id,
                    "age": p.age,
                    "gender": p.gender,
                    "cohort": "Pediatric" if p.age <= 18 else "Adult",
                    "assessments_count": len(p.assessments),
                })
            return patient_list
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in /patients: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch patients: {str(e)}")


@router.get("/health")
async def health_check():
    return {
        "status": "ok",
        "models_loaded": list(models.keys()),
        "db": "connected"
    }

# ─── Attach Router & Root Handlers ────────────────────────────────────────────
app.include_router(router)
app.include_router(router, prefix="/api")

@app.get("/")
@app.get("/api")
async def root():
    return {
        "name": "Dengue Risk Assessment API",
        "status": "online",
        "models_loaded": list(models.keys()),
        "endpoints": [
            "/predict", "/assessments/{patient_id}", "/patients", "/health",
            "/api/predict", "/api/assessments/{patient_id}", "/api/patients", "/api/health"
        ]
    }
