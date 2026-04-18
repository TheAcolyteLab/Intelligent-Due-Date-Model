"""
api/constraints.py
------------------
Deterministic constraint engine for inventory-aware due date scheduling.

This is intentionally NOT ML — these are hard business rules that must
always hold regardless of what the model predicts:

  1. Hard-blocked tasks (inventoryBlocked=true) → floor = today + BLOCKED_BUFFER_DAYS
  2. Ordered/in-transit items → floor = latest arrival date + effort buffer
  3. Low stock → no date change, but flag risk in reason
  4. "doing" tasks → scheduled before "todo" tasks
  5. No two tasks share the same due date (spread them out)

The ML model provides the duration estimate. This layer applies constraints
on top of that estimate to produce the final suggestion.
"""

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Literal, Optional

# ── Constants ─────────────────────────────────────────────────────────────────

# Days added beyond the latest arrival date to account for actual work effort
ARRIVAL_EFFORT_BUFFER_DAYS = 2

# Minimum days out for a hard-blocked task
BLOCKED_BUFFER_DAYS = 14

# Statuses that represent inventory not yet available
BLOCKING_INVENTORY_STATUSES = {"ordered", "in_transit"}

# Status priority for scheduling order (lower = sooner)
STATUS_PRIORITY = {"doing": 0, "todo": 1, "backlog": 2}


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class InventoryItem:
    status: str                    # "available", "ordered", "in_transit", "low_stock"
    expected_date: Optional[date]  # arrival date (None if unknown or already available)


@dataclass
class TaskInput:
    id: str
    title: str
    status: str                           # "doing", "todo", "backlog", "done"
    story_points: Optional[int]           # 1,2,3,5,8,13 or None
    current_due_date: Optional[date]      # existing due date, if any
    inventory_blocked: bool               # hard block flag from Convex
    inventory_items: list[InventoryItem]  # all inventory items for this task
    predicted_days: float                 # raw ML model output


@dataclass
class ScheduledTask:
    id: str
    title: str
    suggested_date: date
    reason: str
    inventory_risk: bool = False
    was_hard_blocked: bool = False
    floor_date: Optional[date] = None    # the constraint floor, if any was applied


# ── Core constraint functions ─────────────────────────────────────────────────

def compute_inventory_floor(task: TaskInput, today: date) -> tuple[Optional[date], str]:
    """
    Returns (floor_date, reason_fragment) based on inventory state.

    floor_date: the earliest possible due date given inventory constraints.
                None if no inventory constraint applies.
    reason_fragment: a string to incorporate into the final reason.
    """
    if task.inventory_blocked:
        floor = today + timedelta(days=BLOCKED_BUFFER_DAYS)
        return floor, f"hard-blocked on inventory — cannot start until parts arrive"

    blocking_items = [
        item for item in task.inventory_items
        if item.status in BLOCKING_INVENTORY_STATUSES and item.expected_date is not None
    ]

    if blocking_items:
        latest_arrival = max(item.expected_date for item in blocking_items)
        floor = latest_arrival + timedelta(days=ARRIVAL_EFFORT_BUFFER_DAYS)
        return floor, f"parts arrive {latest_arrival.isoformat()}, work can start after"

    low_stock_items = [
        item for item in task.inventory_items
        if item.status == "low_stock"
    ]
    if low_stock_items:
        # Low stock: don't delay the date, but flag the risk
        return None, "low stock risk — monitor availability"

    return None, ""


def apply_constraints(task: TaskInput, today: date) -> tuple[date, str, bool, bool]:
    """
    Compute the final suggested date and reason for one task.

    Returns: (suggested_date, reason, inventory_risk, was_hard_blocked)
    """
    floor_date, inventory_fragment = compute_inventory_floor(task, today)
    was_hard_blocked = task.inventory_blocked
    inventory_risk = "low stock" in inventory_fragment

    # Base date: today + ML prediction
    base_date = today + timedelta(days=max(1, round(task.predicted_days)))

    # Apply floor constraint
    if floor_date and base_date < floor_date:
        final_date = floor_date
        reason_core = f"Scheduled for {final_date.isoformat()} — {inventory_fragment}."
    else:
        final_date = base_date
        if inventory_fragment and not inventory_risk:
            reason_core = f"Scheduled for {final_date.isoformat()} — {inventory_fragment}."
        elif inventory_risk:
            reason_core = f"Estimated {task.predicted_days:.0f}d effort — ⚠ {inventory_fragment}."
        else:
            reason_core = f"Based on {task.story_points or '?'} story points ({task.predicted_days:.1f}d estimated)."

    # If existing due date is reasonable (within 2 days of suggestion), prefer it
    if task.current_due_date and not was_hard_blocked:
        delta = abs((task.current_due_date - final_date).days)
        if delta <= 2 and task.current_due_date >= today:
            final_date = task.current_due_date
            reason_core = f"Keeping existing due date — {reason_core.lower()}"

    return final_date, reason_core, inventory_risk, was_hard_blocked


# ── Scheduler: spread tasks across dates ─────────────────────────────────────

def schedule_tasks(tasks: list[TaskInput], today: date) -> list[ScheduledTask]:
    """
    Apply constraints to all tasks and spread them to avoid clustering.

    Ordering: "doing" tasks are scheduled soonest, then "todo", then "backlog".
    No two tasks are assigned the same date — each gets bumped by 1 day if needed.
    """
    # Sort by status priority so "doing" tasks get first pick of dates
    ordered = sorted(
        tasks,
        key=lambda t: (STATUS_PRIORITY.get(t.status, 99), t.predicted_days),
    )

    used_dates: set[date] = set()
    results: list[ScheduledTask] = []

    for task in ordered:
        suggested, reason, risk, blocked = apply_constraints(task, today)

        # Spread: if date already taken, push forward 1 day at a time
        # (skip weekends if you want — left as an extension point)
        original_suggestion = suggested
        while suggested in used_dates:
            suggested += timedelta(days=1)

        if suggested != original_suggestion:
            reason += f" (shifted by {(suggested - original_suggestion).days}d to avoid clustering)"

        used_dates.add(suggested)
        results.append(ScheduledTask(
            id=task.id,
            title=task.title,
            suggested_date=suggested,
            reason=reason,
            inventory_risk=risk,
            was_hard_blocked=blocked,
            floor_date=None,  # could surface this to the UI if useful
        ))

    return results