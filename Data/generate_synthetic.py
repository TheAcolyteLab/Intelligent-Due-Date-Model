"""
    python data/generate_synthetic.py
Output:
    data/tasks.csv          — full dataset
    data/tasks_clean.csv    — NaNs imputed (for quick model testing)
"""

import os
import numpy as np
import pandas as pd

SEED = 42
N_SAMPLES = 8000   # larger dataset to cover the new feature space
rng = np.random.default_rng(SEED)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: Feature sampling
# Each feature is sampled from a distribution that reflects real project data.
# Where features are correlated in real life, we model that correlation.
# ─────────────────────────────────────────────────────────────────────────────

def sample_features(n: int) -> dict:

    # ── Priority: 0=low, 1=medium, 2=high, 3=critical ────────────────────────
    # Skewed toward medium/high — most teams don't mark things low priority
    priority = rng.choice([0, 1, 2, 3], size=n, p=[0.10, 0.40, 0.35, 0.15])

    # ── Complexity (story points) ─────────────────────────────────────────────
    # CORRELATED with priority: high priority tasks tend to be bigger
    # We model this by adding a priority-dependent offset to the SP probabilities
    complexity_choices = np.array([1, 2, 3, 5, 8, 13])
    complexity = np.empty(n, dtype=int)
    for i in range(n):
        if priority[i] <= 1:    # low/medium → smaller tasks
            w = np.array([0.20, 0.30, 0.25, 0.15, 0.07, 0.03])
        else:                   # high/critical → larger tasks
            w = np.array([0.05, 0.10, 0.20, 0.30, 0.25, 0.10])
        complexity[i] = rng.choice(complexity_choices, p=w)

    # ── Task type: 0=bug, 1=feature, 2=chore, 3=research ─────────────────────
    task_type = rng.choice([0, 1, 2, 3], size=n, p=[0.30, 0.40, 0.20, 0.10])

    # ── Assignee load ─────────────────────────────────────────────────────────
    # Real teams have some overloaded members and some underloaded ones
    assignee_load = rng.integers(0, 16, size=n)

    # ── Project velocity ──────────────────────────────────────────────────────
    # Persistent team characteristic — not uniform. Some teams are 2x faster.
    # Model as a mixture: slow teams (mean=5) and fast teams (mean=20)
    slow_mask = rng.random(n) < 0.35
    project_velocity = np.where(
        slow_mask,
        np.clip(rng.normal(5, 2, size=n), 1, 10),
        np.clip(rng.normal(18, 5, size=n), 5, 40),
    )

    # ── Team size ─────────────────────────────────────────────────────────────
    team_size = rng.integers(2, 16, size=n)

    # ── Days in backlog ───────────────────────────────────────────────────────
    # Right-skewed: most tasks sit <2 weeks, some sit for months
    days_in_backlog = np.clip(
        rng.exponential(scale=10, size=n).astype(int),
        0, 120
    )

    # ─────────────────────────────────────────────────────────────────────────
    # ISSUE 2 FIX: INVENTORY FEATURES
    # These are ForeSight's biggest differentiator. Tasks with inventory
    # dependencies have a hard floor — they literally cannot complete until
    # parts arrive. This is a STEP FUNCTION effect, not a smooth relationship,
    # which is why it needs to be an explicit feature rather than inferred.
    # ─────────────────────────────────────────────────────────────────────────

    # Was the task blocked on inventory at any point? (15% of tasks)
    inventory_blocked = rng.choice([0, 1], size=n, p=[0.85, 0.15])

    # If blocked, how many days did it add? (0 if not blocked)
    # Conditional: unblocked tasks get 0, blocked tasks get a real delay
    inventory_delay_days = np.where(
        inventory_blocked,
        rng.integers(3, 31, size=n),   # 3–30 day delays are realistic
        0,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # ISSUE 3 FIX: TASK DEPENDENCIES
    # Tasks with upstream blockers take longer because:
    #   1. You wait for the upstream task to finish
    #   2. Merging/integrating dependent work adds overhead
    # The delay per dependency is itself variable (some are trivial, some huge)
    # ─────────────────────────────────────────────────────────────────────────

    num_dependencies = rng.integers(0, 6, size=n)
    # Each dependency adds 0.5–2.0 days on average
    dependency_delay = np.power(num_dependencies, 1.3) * rng.uniform(0.5, 2.0, size=n)

    # ─────────────────────────────────────────────────────────────────────────
    # ISSUE 6 FIX: TEMPORAL CONTEXT
    # Real completion times vary by:
    #   - Day of week (tasks started Friday take longer — weekend gap)
    #   - Sprint position (tasks near sprint end get forced-done faster)
    # ─────────────────────────────────────────────────────────────────────────

    # Day of week task was started: 0=Mon … 4=Fri (we only work weekdays)
    day_of_week = rng.integers(0, 5, size=n)

    # Sprint day: which day of the sprint (0=day1 … 9=last day of a 2-week sprint)
    sprint_day = rng.integers(0, 10, size=n)
    team_type = rng.choice(["chaotic", "structured"], size = n, p = [0.3, 0.7])

    return dict(
        priority=priority,
        complexity=complexity,
        task_type=task_type,
        assignee_load=assignee_load,
        project_velocity=np.round(project_velocity, 2),
        team_size=team_size,
        days_in_backlog=days_in_backlog,
        inventory_blocked=inventory_blocked,
        inventory_delay_days=inventory_delay_days,
        num_dependencies=num_dependencies,
        dependency_delay=np.round(dependency_delay, 2),
        day_of_week=day_of_week,
        sprint_day=sprint_day,
        team_type=team_type,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: Target variable construction
#
# This is the "data generating process" — the ground truth formula we pretend
# produced the data. The model's job is to learn an approximation of this.
#
# WHY NOT JUST USE RANDOM TARGETS?
# Random targets would teach the model nothing. The formula creates statistical
# signal: features that matter in the formula will have predictive power in
# the model. Features that don't appear won't be learned as predictors.
# ─────────────────────────────────────────────────────────────────────────────

def compute_target(feats: dict, n: int) -> np.ndarray:
    priority             = feats["priority"]
    complexity           = feats["complexity"]
    task_type            = feats["task_type"]
    assignee_load        = feats["assignee_load"]
    project_velocity     = feats["project_velocity"]
    team_size            = feats["team_size"]
    days_in_backlog      = feats["days_in_backlog"]
    inventory_blocked    = feats["inventory_blocked"]
    inventory_delay_days = feats["inventory_delay_days"]
    dependency_delay     = feats["dependency_delay"]
    day_of_week          = feats["day_of_week"]
    sprint_day           = feats["sprint_day"]
    team_type            = feats["team_type"]

    # ── Base: story points → days (1 SP ≈ 1.5 working days) ─────────────────
    base = complexity * 1.5

    # ── Priority multiplier ───────────────────────────────────────────────────
    # Critical tasks get real focus; low priority tasks drift
    priority_mult =np.where(
        team_type == "chaotic",
        np.array([1.4, 1.1, 1.0, 0.85])[priority],
        np.array([1.35, 1.0, 0.80, 0.55])[priority]
    )


    # ── Task type multiplier ──────────────────────────────────────────────────
    # Chores are quickest (known scope), research is unpredictable
    type_mult = np.array([0.75, 1.3, 0.65, 1.7])[task_type]

    # ── Assignee load penalty ─────────────────────────────────────────────────
    # Context switching is costly. Nonlinear: first 5 tasks hurt most
    load_penalty = np.log1p(assignee_load) * 0.8

    # ── Team velocity reduction ───────────────────────────────────────────────
    velocity_factor = 1 - (project_velocity / 40) * 0.45

    # ── Team size: diminishing returns on parallelism ─────────────────────────
    team_factor = 1 - (np.log1p(team_size) / np.log1p(15)) * 0.25

    # ── Backlog staleness ─────────────────────────────────────────────────────
    # Tasks that sit get context rot — need re-familiarisation
    backlog_penalty = np.sqrt(days_in_backlog) * 0.2

    # ── INVENTORY DELAY (Issue 2) ─────────────────────────────────────────────
    # This is additive and unconditional — if parts are delayed, work stops.
    # The model must learn this is a HARD blocker, not a soft signal.
    # Note: we don't multiply — inventory delay adds directly to calendar time.
    eta_error = rng.normal(0, 3, size = n)
    inventory_impact = inventory_blocked * np.maximum(0, inventory_delay_days + eta_error)

    # ── DEPENDENCY DELAY (Issue 3) ────────────────────────────────────────────
    # Also additive. Each dependency is a potential wait + integration cost.
    dep_impact = dependency_delay

    # ── TEMPORAL EFFECTS (Issue 6) ────────────────────────────────────────────
    # Friday starts: task bleeds into next week (+1.5 days average)
    friday_penalty = np.where(day_of_week == 4, 1.5, 0.0)

    # Sprint end pressure: tasks started in last 2 days of sprint get forced
    # to "done" faster (managers close them). Real phenomenon.
    sprint_end_factor = np.where(sprint_day >= 8, 0.75, 1.0)

    # ── Assemble ──────────────────────────────────────────────────────────────
    days = (
        base
        * priority_mult
        * type_mult
        * team_factor
        * velocity_factor
        * sprint_end_factor
        + load_penalty
        + backlog_penalty
        + inventory_impact    # hard additive floor from inventory
        + dep_impact          # additive overhead from dependencies
        + friday_penalty
    )

    # Interaction: High load + High dependencies = Exponential friction
    # This models "The Wall" where developers get stuck in meeting hell.
    interaction = (feats["assignee_load"] > 10) & (feats["num_dependencies"] > 3)
    days[interaction] *= rng.uniform(1.2, 1.8, size=interaction.sum())

    # Failure Mode: Rare "Scope Creep Explosion"
    # This represents a task that fundamentally breaks or is redefined mid-way.
    failure_mask = rng.random(n) < 0.05
    days[failure_mask] *= rng.uniform(2, 5, size=failure_mask.sum())

    # ── Noise ─────────────────────────────────────────────────────────────────
    # Lognormal: right-skewed (tasks can run way over, rarely under)
    # sigma=0.25 means ~25% typical deviation
    noise_scale = np.where(task_type == 3, 0.4, 0.2)
    noise = rng.lognormal(mean=0, sigma=noise_scale)
    # noise = rng.lognormal(mean=0, sigma=0.25, size=n)
    days = days * noise

    return days


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: Inject realistic data quality issues
#
# ISSUE 4 FIX: DIRTY DATA
# Real data has missing values, outliers, and inconsistencies.
# Training on clean data and deploying on dirty data = silent model failure.
# The model must learn to handle these at training time.
# ─────────────────────────────────────────────────────────────────────────────

def inject_data_quality_issues(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    df = df.copy()
    n = len(df)

    # ── Missing values ────────────────────────────────────────────────────────
    # story_points (complexity) is often not set when tasks are created
    missing_complexity = rng.random(n) < 0.12   # 12% unestimated
    df.loc[missing_complexity, "complexity"] = np.nan

    # assignee_load: not always tracked (different integrations, unassigned tasks)
    missing_load = rng.random(n) < 0.08
    df.loc[missing_load, "assignee_load"] = np.nan

    # inventory_delay_days: if no inventory tracking integration, it's unknown
    missing_inv_delay = rng.random(n) < 0.05
    df.loc[missing_inv_delay, "inventory_delay_days"] = np.nan

    # ── Outliers ──────────────────────────────────────────────────────────────
    # ~2% of tasks are genuine outliers: sick leave, total scope creep,
    # external blockers, or waiting months for a vendor.
    # These should NOT be removed — the model must learn they exist.
    outlier_mask = rng.random(n) < 0.02
    outlier_multiplier = rng.uniform(3.0, 8.0, size=n)
    df.loc[outlier_mask, "actual_days_to_complete"] *= outlier_multiplier[outlier_mask]

    # ── Inconsistent patterns ─────────────────────────────────────────────────
    # Sometimes "critical" priority tasks sit in the backlog for weeks anyway
    # (priority inflation: teams mark everything critical)
    priority_inflation = (df["priority"] == 3) & (rng.random(n) < 0.20)
    df.loc[priority_inflation, "actual_days_to_complete"] *= rng.uniform(1.5, 3.0, size=priority_inflation.sum())

    return df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: Leakage audit
#
# ISSUE 5 FIX: FEATURE LEAKAGE
# Leakage = using information at training time that you wouldn't have at
# prediction time. It makes the model look great in eval but fail in prod.
#
# What would leak:
#   ✗  actual_completion_date  (that IS the target, derived from it)
#   ✗  days_remaining          (computed from due date — circular)
#   ✗  was_completed_on_time   (you don't know this at prediction time)
#   ✗  final_story_points      (re-estimated AFTER the work — use original)
#
# What is safe (all features we use):
#   ✓  complexity/story_points — set BEFORE work starts
#   ✓  priority — set at task creation
#   ✓  assignee_load — observable at assignment time
#   ✓  inventory_blocked — observable at task creation
#   ✓  inventory_delay_days — known from the PO/supplier at creation time
#   ✓  num_dependencies — set at sprint planning
#   ✓  day_of_week — when the task was started (observable)
#   ✓  sprint_day — known from sprint calendar
# ─────────────────────────────────────────────────────────────────────────────

SAFE_FEATURES = [
    "priority",
    "complexity",
    "task_type",
    "assignee_load",
    "project_velocity",
    "team_size",
    "days_in_backlog",
    "inventory_blocked",
    "inventory_delay_days",
    "num_dependencies",
    "dependency_delay",
    "day_of_week",
    "sprint_day",
    "team_type"
]

LEAKED_COLUMNS = [
    # Add any column names here that should NEVER appear in training data.
    # The audit function below will raise if they're found.
    "days_remaining",
    "was_on_time",
    "actual_completion_date",
    "final_story_points",
]


def audit_leakage(df: pd.DataFrame):
    """Raise immediately if any leaked column made it into the dataset."""
    found = [c for c in LEAKED_COLUMNS if c in df.columns]
    if found:
        raise ValueError(
            f"LEAKAGE DETECTED — these columns must not be in training data: {found}\n"
            "These are only observable AFTER the task completes."
        )
    print("  Leakage audit passed ✓ — no future-information columns found")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def generate_dataset(n: int) -> pd.DataFrame:
    feats = sample_features(n)
    days  = compute_target(feats, n)
    days  = np.clip(days, 0.5, 180)   # max 6 months — real projects can drag
    days  = np.round(days, 1)

    df = pd.DataFrame({**feats, "actual_days_to_complete": days})
    df = inject_data_quality_issues(df, rng)

    return df


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    print("=" * 55)
    print("  ForeSight Synthetic Data Generator v2")
    print("=" * 55)

    df = generate_dataset(N_SAMPLES)

    # ── Leakage audit ─────────────────────────────────────────────────────────
    audit_leakage(df)

    # ── Stats ─────────────────────────────────────────────────────────────────
    print(f"\n  Rows generated : {len(df):,}")
    print(f"  Features       : {len(SAFE_FEATURES)}")

    print("\n  Missing value rates:")
    for col in df.columns:
        pct = df[col].isna().mean() * 100
        if pct > 0:
            print(f"    {col:<30} {pct:.1f}%")

    print("\n  Target distribution (actual_days_to_complete):")
    desc = df["actual_days_to_complete"].describe()
    print(f"    mean   {desc['mean']:.1f}d")
    print(f"    median {desc['50%']:.1f}d")
    print(f"    p95    {df['actual_days_to_complete'].quantile(0.95):.1f}d")
    print(f"    max    {desc['max']:.1f}d")

    print("\n  Inventory block rate:")
    print(f"    {df['inventory_blocked'].mean()*100:.1f}% of tasks had inventory delays")
    print(f"    avg delay when blocked: {df.loc[df['inventory_blocked']==1,'inventory_delay_days'].mean():.1f}d")

    print("\n  Dependency distribution:")
    print(f"    {(df['num_dependencies']==0).mean()*100:.1f}% tasks have no dependencies")
    print(f"    avg dependencies: {df['num_dependencies'].mean():.1f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = "data/tasks.csv"
    df.to_csv(out_path, index=False)
    print(f"\n  Saved → {out_path}")

    # Also save a clean version (NaNs imputed with median) for quick testing
    df_clean = df.copy()
    for col in df_clean.columns:
        if df_clean[col].isna().any():
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    clean_path = "data/tasks_clean.csv"
    df_clean.to_csv(clean_path, index=False)
    print(f"  Saved → {clean_path}  (NaNs imputed, for quick model testing)")

    print("\n" + "=" * 55)