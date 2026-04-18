import pandas as pd
from api.main import engineer_features, FEATURE_NAMES

def test_feature_generation_complete():
    df = pd.DataFrame([{
        "priority": 2,
        "complexity": 5,
        "task_type": 1,
        "assignee_load": 10,
        "project_velocity": 8,
        "team_size": 4,
        "days_in_backlog": 3,
        "inventory_blocked": 1,
        "inventory_delay_days": 10,
        "num_dependencies": 2,
        "dependency_delay": 3.0,
        "sprint_day": 5,
        "team_type": "chaotic",
    }])

    out = engineer_features(df)

    # ✅ Ensure all expected features exist
    assert set(out.columns) == set(FEATURE_NAMES)

    # ✅ No NaNs
    assert out.isnull().sum().sum() == 0

    # ✅ Check critical derived feature
    assert out["inventory_x_delay"].iloc[0] == 10

    # ✅ Binary encoding works
    assert out["team_type"].iloc[0] == 1