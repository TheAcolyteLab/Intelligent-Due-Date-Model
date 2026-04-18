import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "tasks.csv"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

TARGET = "actual_days_to_complete"
RANDOM_STATE = 42

# All raw features the model receives (must match generate_synthetic.py SAFE_FEATURES)
FEATURES = [
    "priority",
    "complexity",           # has NaNs — imputed
    "task_type",
    "assignee_load",        # has NaNs — imputed
    "project_velocity",
    "team_size",
    "days_in_backlog",
    "inventory_blocked",
    "inventory_delay_days", # has NaNs — imputed
    "num_dependencies",
    "dependency_delay",
    "day_of_week",
    "sprint_day",
    "team_type",
]

# Engineered features (created from raw features, no leakage)
ENGINEERED = [
    "load_per_velocity",
    "complexity_per_teammate",
    "priority_x_complexity",
    "inventory_x_delay",        # NEW: interaction between blocked and delay
    "deps_x_complexity",        # NEW: dependency overhead scales with task size
    "is_friday_start",          # NEW: binary flag for day_of_week == 4
    "is_sprint_end",            # NEW: binary flag for sprint_day >= 8
]


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} rows from {path}")
    missing_pct = df.isna().mean() * 100
    missing = missing_pct[missing_pct > 0]
    if not missing.empty:
        print("  Missing values:")
        for col, pct in missing.items():
            print(f"    {col:<30} {pct:.1f}%")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create additional features from raw inputs.

    IMPORTANT: All operations here must use only features observable at
    PREDICTION time (before the task completes). Never use the target.
    """
    df = df.copy()

    # 1 for chaotic, 0 for structured (this converts team_type to numeric)
    if "team_type" in df.columns:
        df["team_type"] = (df["team_type"] == "chaotic").astype(int)
   
    # Handle NaNs before engineering (use 0 as default for ratio features)
    load   = df["assignee_load"].fillna(0)
    comp   = df["complexity"].fillna(df["complexity"].median())
    inv_d  = df["inventory_delay_days"].fillna(0)

    # Existing engineered features
    df["load_per_velocity"]      = load / (df["project_velocity"] + 1e-6)
    df["complexity_per_teammate"] = comp / (df["team_size"] + 1e-6)
    df["priority_x_complexity"]  = df["priority"] * comp

    # New: inventory block × delay length — captures the multiplicative risk
    # An unblocked task with delay_days=0 contributes nothing here.
    df["inventory_x_delay"]  = df["inventory_blocked"] * inv_d

    # New: how dependency overhead scales with task complexity
    df["deps_x_complexity"]  = df["num_dependencies"] * comp

    # New: temporal binary flags (cleaner signal than raw integers for these)
    df["is_friday_start"] = (df["day_of_week"] == 4).astype(int)
    df["is_sprint_end"]   = (df["sprint_day"] >= 8).astype(int)

    return df


def build_pipeline(model) -> Pipeline:
    """
    Build a preprocessing + model pipeline.

    The pipeline handles missing values via median imputation BEFORE scaling.
    This is done inside the pipeline so imputation parameters are fit only on
    training data and applied to test/inference data — no leakage.
    """
    all_features = FEATURES + ENGINEERED
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "process",
                Pipeline([
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale",  StandardScaler()),
                ]),
                all_features,
            )
        ],
        remainder="drop",
    )
    return Pipeline([
        ("engineer",     "passthrough"),   # engineering happens before pipeline
        ("preprocessor", preprocessor),
        ("model",        model),
    ])


def evaluate(name: str, pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    raw_pred = pipeline.predict(X_test)

    clipped_mask = (raw_pred <= 0.5) | (raw_pred >= 180)
    clipped_count = np.sum(clipped_mask)
   
    y_pred = np.clip(raw_pred, 0.5, 180)
    # y_pred = np.clip(pipeline.predict(X_test), 0.5, 365)

    mae    = mean_absolute_error(y_test, y_pred)
    mdae   = np.median(np.abs(y_test.values - y_pred))  # outlier-robust
    bias   = np.mean(y_pred - y_test)
    # rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
    # r2     = r2_score(y_test, y_pred)

    print(f"\n{'─'*60}")
    print(f" DEEP DIVE: {name}")
    print(f"{'─'*60}")
    print(f"  MAE: {mae: .2f}d | MdAE: {mdae:.2f}d | Bias: {bias:.2f}d")
    print(f"  Clipped Predictions: {clipped_count} ({clipped_count/len(y_test)*100:.1f}%)")
    
    inv_mask = X_test["inventory_blocked"] == 1
    if inv_mask.any():
        mae_inv = mean_absolute_error(y_test[inv_mask], y_pred[inv_mask])
        mae_no_inv = mean_absolute_error(y_test[~inv_mask], y_pred[~inv_mask])
        print(f"  MAE (Inventory Blocked): {mae_inv:.2f}d")
        print(f"  MAE (No Inventory): {mae_no_inv:.2f}d")

    for t_type in ["chaotic", "structured"]:
        t_mask = X_test["team_type"] == t_type
        if t_mask.any():
            t_mae = mean_absolute_error(y_test[t_mask], y_pred[t_mask])
            print(f"  MAE ({t_type:<10}):       {t_mae:.2f}d")
    return {"mae": mae,"mdae": mdae, "bias": bias}
    # print(f"\n{'─'*48}")
    # print(f"  {name}")
    # print(f"{'─'*48}")
    # print(f"  MAE    : {mae:.2f}d  (mean error — sensitive to outliers)")
    # print(f"  MdAE   : {mdae:.2f}d  (median error — robust to outliers) ← trust this")
    # print(f"  RMSE   : {rmse:.2f}d")
    # print(f"  R²     : {r2:.4f}")
    # return {"name": name, "mae": round(mae, 4), "mdae": round(mdae, 4),
    #         "rmse": round(rmse, 4), "r2": round(r2, 4)}


def print_feature_importance(pipeline, all_features: list):
    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return
    importances = model.feature_importances_
    pairs = sorted(zip(all_features, importances), key=lambda x: x[1], reverse=True)
    print("\n  Feature Importances (top 10):")
    print(f"  {'Feature':<32} {'Importance':>10}")
    print(f"  {'─'*44}")
    for feat, imp in pairs[:10]:
        bar = "█" * int(imp * 60)
        print(f"  {feat:<32} {imp:>8.4f}  {bar}")

def heuristic_baseline(X):
    # Mimicking a 'smart' project manager's rule of thumb
    base = X["complexity"].fillna(3) * 1.5
    # Add 1 day per dependency and the full inventory delay
    inventory = X["inventory_delay_days"].fillna(0)
    deps = X["num_dependencies"] * 1.2
    return base + inventory + deps


def main():
    print("=" * 55)
    print("  ForeSight Due Date Model — Training Pipeline v2")
    print("=" * 55)

    df = load_data(DATA_PATH)
    df = engineer_features(df)

    all_features = FEATURES + ENGINEERED
    X = df[all_features]
    y = df[TARGET]

    print(f"\n  Features ({len(all_features)}): {', '.join(all_features)}")
    print(f"  NaN counts in X:\n{X.isna().sum()[X.isna().sum() > 0].to_string()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=RANDOM_STATE
    )
    print(f"\n  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # Baseline: Ridge
    print("\n[1/2] Training Ridge baseline...")
    ridge_pl = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale",  StandardScaler()),
        ("model",  Ridge()),
    ])
    ridge_pl.fit(X_train, y_train)
    ridge_metrics = evaluate("Ridge Regression (baseline)", ridge_pl, X_test, y_test)

    # Primary: Gradient Boosting
    print("\n[2/2] Training Gradient Boosting Regressor...")
    gbr = GradientBoostingRegressor(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.04,
        subsample=0.8,
        min_samples_leaf=15,
        random_state=RANDOM_STATE,
    )
    gbr_pl = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        # ("scale",  StandardScaler()),
        ("model",  gbr),
    ])
    gbr_pl.fit(X_train, y_train)
    gbr_metrics = evaluate("Gradient Boosting Regressor (primary)", gbr_pl, X_test, y_test)

    # 5-fold CV on training set
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(gbr_pl, X_train, y_train,
                                cv=kf, scoring="neg_mean_absolute_error")
    cv_mae = -cv_scores
    print(f"\n  5-Fold CV MAE: {cv_mae.mean():.2f} ± {cv_mae.std():.2f}d")

    y_heuristic = heuristic_baseline(X_test)
    h_mae = mean_absolute_error(y_test, y_heuristic)
    print(f"HEURISTIC BASELINE MAE: {h_mae:.2f}d")

    # Feature importance
    # GBR inside the pipeline — need to reference .named_steps["model"]
    importances = gbr_pl.named_steps["model"].feature_importances_
    pairs = sorted(zip(all_features, importances), key=lambda x: x[1], reverse=True)
    print("\n  Feature Importances (top 10):")
    print(f"  {'Feature':<32} {'Importance':>10}")
    print(f"  {'─'*44}")
    for feat, imp in pairs[:10]:
        bar = "█" * int(imp * 60)
        print(f"  {feat:<32} {imp:>8.4f}  {bar}")

    # Check inventory features rank (they should be significant)
    inv_rank = next(i for i, (f, _) in enumerate(pairs) if "inventory" in f)
    print(f"\n  First inventory feature appears at rank #{inv_rank + 1}")
    dep_rank = next(i for i, (f, _) in enumerate(pairs) if "dep" in f)
    print(f"  First dependency feature appears at rank #{dep_rank + 1}")


    # Save
    model_path = MODELS_DIR / "due_date_model.joblib"
    joblib.dump(gbr_pl, model_path)
    print(f"\n  Model saved → {model_path}")

    feature_path = MODELS_DIR / "feature_names.json"
    with open(feature_path, "w") as f:
        json.dump(all_features, f, indent=2)

    metrics_path = MODELS_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({"baseline": ridge_metrics, "primary": gbr_metrics}, f, indent=2)

    print(f"  Features → {feature_path}")
    print(f"  Metrics  → {metrics_path}")
    print("\n" + "=" * 55)
    print("  Training complete.")
    print("=" * 55)

    # After GBR training
    residuals = y_test - gbr_pl.predict(X_test)
    X_test_with_res = X_test.copy()
    X_test_with_res["actual"] = y_test
    X_test_with_res["pred"] = gbr_pl.predict(X_test)
    X_test_with_res["error"] = np.abs(residuals)

    print("\n[WORST ERRORS - TOP 5]")
    print(X_test_with_res.sort_values("error", ascending=False).head(5))



if __name__ == "__main__":
    main()