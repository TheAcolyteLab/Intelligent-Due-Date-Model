from datetime import date, timedelta
from api.constraints import TaskInput, InventoryItem, schedule_tasks

def test_hard_blocked_task():
    today = date.today()

    task = TaskInput(
        id="1",
        title="Blocked Task",
        status="todo",
        story_points=5,
        current_due_date=None,
        inventory_blocked=True,
        inventory_items=[],
        predicted_days=3,
    )

    result = schedule_tasks([task], today)[0]

    assert result.suggested_date >= today + timedelta(days=14)
    assert result.was_hard_blocked is True


def test_inventory_arrival_floor():
    today = date.today()

    task = TaskInput(
        id="2",
        title="Waiting for parts",
        status="todo",
        story_points=3,
        current_due_date=None,
        inventory_blocked=False,
        inventory_items=[
            InventoryItem(
                status="ordered",
                expected_date=today + timedelta(days=5),
            )
        ],
        predicted_days=2,
    )

    result = schedule_tasks([task], today)[0]

    # Must be AFTER arrival + buffer (2 days)
    assert result.suggested_date >= today + timedelta(days=7)


def test_low_stock_does_not_delay():
    today = date.today()

    task = TaskInput(
        id="3",
        title="Low stock",
        status="todo",
        story_points=3,
        current_due_date=None,
        inventory_blocked=False,
        inventory_items=[
            InventoryItem(status="low_stock", expected_date=None)
        ],
        predicted_days=3,
    )

    result = schedule_tasks([task], today)[0]

    # No forced delay
    assert result.inventory_risk is True


def test_no_duplicate_dates():
    today = date.today()

    tasks = [
        TaskInput(
            id=str(i),
            title=f"Task {i}",
            status="todo",
            story_points=3,
            current_due_date=None,
            inventory_blocked=False,
            inventory_items=[],
            predicted_days=2,
        )
        for i in range(5)
    ]

    results = schedule_tasks(tasks, today)

    dates = [r.suggested_date for r in results]
    assert len(dates) == len(set(dates))  # no duplicates