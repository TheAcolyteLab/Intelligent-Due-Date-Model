# ForeSight — Intelligent Due Date Predictor

A standalone ML microservice that replaces the Gemini API call for due date suggestions in ForeSight. Built as a learning-first AI engineering project.

## What's Inside

```
foresight-due-date-ml/
├── data/
│   └── generate_synthetic.py   # Generates 5,000 realistic training samples
├── models/                     # Auto-created by train.py
│   ├── due_date_model.joblib   # Trained GBR pipeline
│   ├── feature_names.json      # Ordered feature list
│   └── metrics.json            # MAE, RMSE, R² scores
├── api/
│   └── main.py                 # FastAPI microservice (3 endpoints)
├── convex/
│   └── suggestDueDate.ts       # Convex HTTP action (drop-in replacement)
├── train.py                    # Full training pipeline with evaluation
├── test_model.py               # Smoke test for local inference
├── requirements.txt
├── railway.toml                # Deploy config for Railway.app
└── render.yaml                 # Deploy config for Render.com
```

## Model

**Algorithm:** Gradient Boosting Regressor (scikit-learn)  
**Task:** Regression — predicts days to task completion  
**Training data:** 5,000 synthetic samples with realistic domain signal  

| Metric | Value |
|--------|-------|
| MAE    | ~1.10 days |
| RMSE   | ~1.67 days |
| R²     | ~0.83 |

### Features

| Feature | Description |
|---------|-------------|
| `priority` | 0=low, 1=medium, 2=high, 3=critical |
| `complexity` | Story points: 1, 2, 3, 5, 8, 13 |
| `task_type` | 0=bug, 1=feature, 2=chore, 3=research |
| `assignee_load` | Open tasks for the assignee |
| `project_velocity` | Team avg tasks/week |
| `team_size` | Active contributors |
| `days_in_backlog` | Time in backlog before pickup |
| `load_per_velocity` | Engineered: load relative to team speed |
| `complexity_per_teammate` | Engineered: work per person |
| `priority_x_complexity` | Engineered: interaction term |

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate training data

```bash
python data/generate_synthetic.py
# → data/tasks.csv (5,000 rows)
```

### 3. Train the model

```bash
python train.py
# → models/due_date_model.joblib
# → models/feature_names.json
# → models/metrics.json
```

### 4. Smoke test

```bash
python test_model.py
```

### 5. Run the API locally

```bash
uvicorn api.main:app --reload --port 8000
```

### 6. Test the endpoint

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "priority": 2,
    "complexity": 5,
    "task_type": 1,
    "assignee_load": 4,
    "project_velocity": 12.5,
    "team_size": 5,
    "days_in_backlog": 3
  }'
```

Expected response:

```json
{
  "predicted_days": 7.1,
  "suggested_due_date": "2025-09-04T10:00:00Z",
  "confidence_range": { "low_days": 6.0, "high_days": 8.2 },
  "model_mae_days": 1.10
}
```

## Deploying to Railway

1. Push this folder to a GitHub repo
2. Create a new project at [railway.app](https://railway.app)
3. Connect the repo — Railway auto-detects `railway.toml`
4. Set environment variable in Railway dashboard: none required for the model itself
5. Copy the generated URL (e.g. `https://foresight-ml.up.railway.app`)

## Connecting to ForeSight (Convex)

1. Copy `convex/suggestDueDate.ts` into your ForeSight `convex/actions/` directory
2. In your Convex dashboard, add environment variable:

   ```
   FORESIGHT_ML_URL=https://your-service.railway.app
   ```

3. Call the action from any mutation or the client:

   ```typescript
   const result = await ctx.runAction(api.actions.suggestDueDate.suggest, {
     taskId: task._id,
     priority: task.priority,
     complexity: task.storyPoints,
     taskType: task.type,
     assigneeLoad: openTaskCount,
     projectVelocity: weeklyVelocity,
     teamSize: activeMembers,
     daysInBacklog: daysSinceCreation,
   });
   ```

## Retraining on Real Data

Once ForeSight has real task history, you can retrain on actual completions:

1. Export completed tasks from Convex to a CSV matching the schema in `data/tasks.csv`
2. Replace `data/tasks.csv` with your real data
3. Run `python train.py` — the pipeline is identical
4. Redeploy the `.joblib` artifact

The model will automatically improve as it learns from your users' actual patterns.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness check |
| POST | `/predict` | Single task prediction |
| POST | `/predict/batch` | Up to 100 tasks at once |
| GET | `/model/info` | Feature list and metrics |

Interactive docs: `http://localhost:8000/docs`
