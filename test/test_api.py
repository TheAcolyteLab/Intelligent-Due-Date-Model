from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_suggest_workspace_end_to_end():
    payload = {
        "tasks": [
            {
                "id": "1",
                "title": "Build API",
                "status": "doing",
                "storyPoints": 5,
                "inventoryBlocked": False,
                "inventoryItems": [],
                "assigneeLoad": 5,
                "projectVelocity": 10,
                "teamSize": 4,
            },
            {
                "id": "2",
                "title": "Install hardware",
                "status": "todo",
                "storyPoints": 3,
                "inventoryBlocked": True,
                "inventoryItems": [],
                "assigneeLoad": 2,
                "projectVelocity": 10,
                "teamSize": 4,
            }
        ]
    }

    res = client.post("/suggest-workspace", json=payload)

    assert res.status_code == 200

    data = res.json()
    assert "suggestions" in data
    assert len(data["suggestions"]) == 2

    # Ensure blocked task pushed out
    blocked_task = next(t for t in data["suggestions"] if t["id"] == "2")
    assert blocked_task["wasHardBlocked"] is True