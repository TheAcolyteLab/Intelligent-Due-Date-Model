"""
api/main.py
-----------
FastAPI microservice serving the ForeSight due date prediction model.

Endpoints:
  GET  /health                  — liveness check
  POST /predict                 — single task prediction (raw ML, no constraints)
  POST /predict/batch           — multiple tasks, raw ML
  POST /suggest-workspace       — full inventory-aware scheduling (replaces Gemini)
  GET  /model/info              — model metadata and feature list

Run locally:
    uvicorn api.main:app --reload --port 8000
"""

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

load_dotenv()
api_key = os.getenv("API_KEY")

from api.constraints import (
    InventoryItem,
    TaskInput,
    schedule_tasks,
)

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "due_date_model.joblib"
FEATURES_PATH = MODELS_DIR / "feature_names.json"
METRICS_PATH = MODELS_DIR / "metrics.json"

# ── Load model at startup ────────────────────────────────────────────────────
print("Loading model...")
pipeline = joblib.load(MODEL_PATH)
with open(FEATURES_PATH) as f:
    FEATURE_NAMES: list[str] = json.load(f)
with open(METRICS_PATH) as f:
    METRICS: dict = json.load(f)
print(f"Model loaded. Features: {FEATURE_NAMES}")

# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="ForeSight Due Date Predictor",
    description="ML microservice for intelligent task due date suggestions.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://foresightsync.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schemas ──────────────────────────────────────────────────────────────────

class TaskInputSchema(BaseModel):
    priority: int = Field(..., ge=0, le=3)
    complexity: int = Field(..., ge=1, le=13)
    task_type: int = Field(..., ge=0, le=3)
    assignee_load: int = Field(..., ge=0, le=50)
    project_velocity: float = Field(..., ge=0.1, le=100)
    team_size: int = Field(..., ge=1, le=100)
    days_in_backlog: int = Field(default=0, ge=0, le=365)
    created_hour: Optional[int] = Field(default=9, ge=0, le=23)
    inventory_blocked: bool = False
    inventory_delay_days: float = 0
    num_dependencies: int = 0
    dependency_delay: float = 0
    day_of_week: int = 1
    sprint_day: int = 1

    @field_validator("complexity")
    @classmethod
    def validate_complexity(cls, v):
        if v not in {1, 2, 3, 5, 8, 13}:
            raise ValueError("complexity must be one of 1,2,3,5,8,13")
        return v


class PredictionResponse(BaseModel):
    predicted_days: float
    suggested_due_date: str
    confidence_range: dict
    model_mae_days: float


class InventoryItemSchema(BaseModel):
    status: str
    expected_date: Optional[str] = None


class WorkspaceTaskSchema(BaseModel):
    id: str
    title: str
    status: str
    story_points: Optional[int] = Field(default=None, alias="storyPoints")
    current_due_date: Optional[str] = Field(default=None, alias="currentDueDate")
    inventory_blocked: bool = Field(default=False, alias="inventoryBlocked")
    inventory_items: list[InventoryItemSchema] = Field(default_factory=list, alias="inventoryItems")
    assignee_load: int = Field(default=3, alias="assigneeLoad")
    project_velocity: float = Field(default=10.0, alias="projectVelocity")
    team_size: int = Field(default=5, alias="teamSize")
    num_dependencies: int = Field(default=0, alias="numDependencies")
    dependency_delay: float = Field(default=0.0, alias="dependencyDelay")
    inventory_delay_days: float = Field(default=0.0, alias="inventoryDelayDays")
    sprint_day: int = Field(default=0, alias="sprintDay")
    team_type: str = Field(default="structured", alias="teamType")

    model_config = {"populate_by_name": True}


class WorkspaceSuggestRequest(BaseModel):
    tasks: list[WorkspaceTaskSchema]


class TaskSuggestion(BaseModel):
    id: str
    title: str
    suggestedDate: str
    reason: str
    inventoryRisk: bool = False
    wasHardBlocked: bool = False


class WorkspaceSuggestResponse(BaseModel):
    suggestions: list[TaskSuggestion]
    model_mae_days: float
    generated_at: str


# ── Feature engineering ───────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "team_type" in df.columns:
        df["team_type"] = (df["team_type"] == "chaotic").astype(int)

    load = df["assignee_load"].fillna(0)
    complexity = df["complexity"].fillna(3) 
    inventory_delay_days = df.get("inventory_delay_days", 0)
    num_dependencies = df.get("num_dependencies", 0)
    dependency_delay = df.get("dependency_delay", 0)
    
    df["load_per_velocity"] = load / (df["project_velocity"] + 1e-6)
    df["complexity_per_teammate"] = complexity / (df["team_size"] + 1e-6)
    df["priority_x_complexity"] = df["priority"] * complexity
    df["inventory_x_delay"] = df["inventory_blocked"].astype(int) * inventory_delay_days
    df["deps_x_complexity"] = num_dependencies * complexity
    df["dependency_delay"] = dependency_delay

    if "day_of_week" not in df.columns:
        df["day_of_week"] = datetime.now().weekday()

    df["is_friday_start"] = (df["day_of_week"] == 4).astype(int)
    df["is_sprint_end"] = (df.get("sprint_day", 0) >= 8).astype(int)

    #  CRITICAL: Reindex to match the training feature order
    # This removes 'created_hour' (unseen) and adds any missing (seen)
    df = df.reindex(columns=FEATURE_NAMES, fill_value=0)
    return df


def story_points_to_complexity(sp: Optional[int]) -> int:
    if sp is None:
        return 3
    valid = [1, 2, 3, 5, 8, 13]
    return min(valid, key=lambda v: abs(v - sp))


def status_to_priority(status: str) -> int:
    return {"doing": 2, "todo": 1, "backlog": 0}.get(status, 1)


def predict_days_for_task(task: WorkspaceTaskSchema) -> float:
    row = pd.DataFrame([{
        "priority": status_to_priority(task.status),
        "complexity": story_points_to_complexity(task.story_points),
        "task_type": 1,
        "assignee_load": task.assignee_load,
        "project_velocity": task.project_velocity,
        "team_size": task.team_size,
        "days_in_backlog": 0,
        # "created_hour": datetime.now().hour,
        "inventory_blocked": int(task.inventory_blocked),
        "inventory_delay_days": task.inventory_delay_days,
        "num_dependencies": task.num_dependencies,
        "dependency_delay": task.dependency_delay,
        "sprint_day": task.sprint_day,
        "team_type": task.team_type,
    }])
    
    row = engineer_features(row)
    pred = pipeline.predict(row)[0]
    return float(np.clip(round(pred, 1), 0.5, 90))


def parse_date(date_str: Optional[str]) -> Optional[date]:
    if not date_str:
        return None
    try:
        return date.fromisoformat(date_str[:10])
    except ValueError:
        return None


# ── Generate Reasoning ───────────────────────────────────────────────────────────────────
def get_explanation(task: WorkspaceTaskSchema, pred_days: float) -> str:
    reasons = []
    if task.assignee_load > 10:
        reasons.append("high workload")
    if task.num_dependencies > 3:
        reasons.append("complex dependencies")
    if task.inventory_delay_days > 7:
        reasons.append("inventory lead time")
    
    if not reasons:
        return f"Standard estimate for {task.story_points} SP"
    return "Refinement due to " + " and ".join(reasons)


# ── Data Logging ───────────────────────────────────────────────────────────────────

def log_prediction(task_id: str, features: dict, prediction: float):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "task_id": task_id,
        "features": features,
        "prediction": prediction
    }
    # In production, this goes to a 'predictions' table in Convex or a logging DB
    with open("data/prediction_logs.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")

# ── Authentication for security ─────────────────────────────────────────────

def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != api_key:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": pipeline is not None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(task: TaskInputSchema):
    row = pd.DataFrame([task.model_dump()])
    row = engineer_features(row)
    raw_pred = float(np.clip(round(pipeline.predict(row)[0], 1), 0.5, 90))
    model_mae = METRICS["primary"]["mae"]
    due_date = datetime.now(timezone.utc) + timedelta(days=raw_pred)
    return PredictionResponse(
        predicted_days=raw_pred,
        suggested_due_date=due_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
        confidence_range={
            "low_days": max(0.5, round(raw_pred - model_mae, 1)),
            "high_days": round(raw_pred + model_mae, 1),
        },
        model_mae_days=round(model_mae, 2),
    )


@app.post("/suggest-workspace", response_model=WorkspaceSuggestResponse)
def suggest_workspace(request: WorkspaceSuggestRequest, _: str = Depends(verify_api_key)):
    """
    Full inventory-aware due date scheduling for an entire workspace.
    Drop-in replacement for the Gemini suggestDueDates action.

    Pipeline:
      1. ML model → raw duration estimate per task
      2. Constraint engine → apply inventory floors and hard blocks
      3. Scheduler → spread tasks, order by status (doing → todo → backlog)
      4. Response matches the Gemini JSON output shape exactly
    """
    if not request.tasks:
        return WorkspaceSuggestResponse(
            suggestions=[],
            model_mae_days=METRICS["primary"]["mae"],
            generated_at=datetime.now(timezone.utc).isoformat(),
        )

    today = date.today()

    task_inputs: list[TaskInput] = []
    for t in request.tasks:
        predicted_days = predict_days_for_task(t)
        inventory_items = [
            InventoryItem(
                status=item.status,
                expected_date=parse_date(item.expected_date),
            )
            for item in t.inventory_items
        ]
        task_inputs.append(TaskInput(
            id=t.id,
            title=t.title,
            status=t.status,
            story_points=t.story_points,
            current_due_date=parse_date(t.current_due_date),
            inventory_blocked=t.inventory_blocked,
            inventory_items=inventory_items,
            predicted_days=predicted_days,
        ))

    scheduled = schedule_tasks(task_inputs, today)

    suggestions = [
        TaskSuggestion(
            id=s.id,
            title=s.title,
            suggestedDate=s.suggested_date.isoformat(),
            reason=s.reason,
            inventoryRisk=s.inventory_risk,
            wasHardBlocked=s.was_hard_blocked,
        )
        for s in scheduled
    ]

    return WorkspaceSuggestResponse(
        suggestions=suggestions,
        model_mae_days=round(METRICS["primary"]["mae"], 2),
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


# @app.post("/model/retrain")
# def retrain_model():
    # 1. Pull data from Convex (actual vs predicted)
    # 2. Run the training script (subprocess or internal function)
    # 3. Reload the 'pipeline' global variable
    return {"status": "retraining triggered", "current_version": "1.0.1"}

@app.get("/model/info")
def model_info():
    return {"features": FEATURE_NAMES, "metrics": METRICS, "version": "1.0.0"}