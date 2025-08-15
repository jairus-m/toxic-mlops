"""
Moderation logic and queue management for toxic content moderation
"""

import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional
from dataclasses import dataclass
from src.core import logger


@dataclass
class ModerationDecision:
    """Data class for moderation decision results"""

    action: str  # "allow", "block", "human_review"
    confidence: float
    reasoning: str
    queue_id: Optional[str] = None


# In-memory moderation queue (for MVP - can be replaced with database later)
MODERATION_QUEUE: Dict[str, dict] = {}


def apply_moderation_rules(
    binary_predictions: Dict[str, bool],
    probabilities: Dict[str, float],
    context: Optional[str] = None,
) -> ModerationDecision:
    """
    Apply moderation rules based on toxicity predictions to make automated decisions.

    Args:
        binary_predictions: Binary toxicity predictions from model
        probabilities: Probability scores for each toxicity type
        context: Optional context about where content appears

    Returns:
        ModerationDecision with action, confidence, and reasoning
    """

    # Get max probability and count of toxic categories
    max_prob = max(probabilities.values())
    toxic_count = sum(binary_predictions.values())

    # Decision logic
    if max_prob >= 0.9:
        return ModerationDecision(
            action="reject",
            confidence=max_prob,
            reasoning=f"High confidence toxicity detected (max: {max_prob:.2f})",
        )

    elif max_prob <= 0.1:
        return ModerationDecision(
            action="approve",
            confidence=1.0 - max_prob,
            reasoning=f"Low toxicity probability (max: {max_prob:.2f})",
        )

    elif toxic_count >= 3:
        return ModerationDecision(
            action="human_review",
            confidence=max_prob,
            reasoning=f"Multiple toxicity categories detected ({toxic_count} types)",
        )

    elif max_prob >= 0.7:
        return ModerationDecision(
            action="human_review",
            confidence=max_prob,
            reasoning=f"High confidence toxicity requiring review (max: {max_prob:.2f})",
        )

    elif max_prob >= 0.3:
        return ModerationDecision(
            action="human_review",
            confidence=max_prob,
            reasoning=f"Medium confidence toxicity requiring review (max: {max_prob:.2f})",
        )

    else:
        return ModerationDecision(
            action="approve",
            confidence=1.0 - max_prob,
            reasoning=f"Low confidence toxicity (max: {max_prob:.2f})",
        )


def queue_for_review(
    text: str,
    binary_predictions: Dict[str, bool],
    probabilities: Dict[str, float],
    context: Optional[str] = None,
    user_id: Optional[str] = None,
) -> str:
    """
    Add content to the human review queue.

    Args:
        text: Original text content
        binary_predictions: Binary toxicity predictions
        probabilities: Toxicity probability scores
        context: Optional context information
        user_id: Optional user identifier

    Returns:
        queue_id: Unique identifier for the queued item
    """

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    queue_id = f"mod_{timestamp}_{str(uuid.uuid4())[:8]}"

    max_prob = max(probabilities.values())
    if max_prob >= 0.8:
        priority = "high"
    elif max_prob >= 0.5:
        priority = "medium"
    else:
        priority = "low"

    # Create queue item
    queue_item = {
        "queue_id": queue_id,
        "text": text,
        "toxicity_predictions": binary_predictions,
        "toxicity_probabilities": probabilities,
        "context": context,
        "user_id": user_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "priority": priority,
        "status": "pending",
    }

    MODERATION_QUEUE[queue_id] = queue_item

    logger.info(f"Added item to moderation queue: {queue_id} (priority: {priority})")

    return queue_id


def get_moderation_queue(
    status: str = "pending", priority: Optional[str] = None, limit: Optional[int] = None
) -> List[dict]:
    """
    Retrieve items from the moderation queue.

    Args:
        status: Filter by status ("pending", "reviewed", "all")
        priority: Filter by priority ("high", "medium", "low")
        limit: Maximum number of items to return

    Returns:
        List of queue items matching the criteria
    """

    items = list(MODERATION_QUEUE.values())

    if status != "all":
        items = [item for item in items if item.get("status") == status]

    if priority:
        items = [item for item in items if item.get("priority") == priority]

    priority_order = {"high": 0, "medium": 1, "low": 2}
    items.sort(
        key=lambda x: (
            priority_order.get(x.get("priority", "medium"), 1),
            x.get("created_at", ""),
        )
    )

    if limit:
        items = items[:limit]

    return items


def process_review_decision(
    queue_id: str, action: str, moderator_notes: Optional[str] = None
) -> bool:
    """
    Process a human moderator's decision on a queued item.

    Args:
        queue_id: ID of the item being reviewed
        action: Moderator's decision ("allow", "block", "warn")
        moderator_notes: Optional notes from moderator

    Returns:
        bool: True if processing was successful
    """

    if queue_id not in MODERATION_QUEUE:
        logger.error(f"Queue ID not found: {queue_id}")
        return False

    MODERATION_QUEUE[queue_id].update(
        {
            "status": "reviewed",
            "final_action": action,
            "moderator_notes": moderator_notes,
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
        }
    )

    logger.info(f"Processed review decision for {queue_id}: {action}")

    return True


def get_queue_stats() -> Dict[str, int]:
    """
    Get statistics about the moderation queue.

    Returns:
        Dict with queue statistics
    """

    total_items = len(MODERATION_QUEUE)
    pending_items = len(
        [item for item in MODERATION_QUEUE.values() if item.get("status") == "pending"]
    )
    reviewed_items = len(
        [item for item in MODERATION_QUEUE.values() if item.get("status") == "reviewed"]
    )

    high_priority = len(
        [
            item
            for item in MODERATION_QUEUE.values()
            if item.get("priority") == "high" and item.get("status") == "pending"
        ]
    )
    medium_priority = len(
        [
            item
            for item in MODERATION_QUEUE.values()
            if item.get("priority") == "medium" and item.get("status") == "pending"
        ]
    )
    low_priority = len(
        [
            item
            for item in MODERATION_QUEUE.values()
            if item.get("priority") == "low" and item.get("status") == "pending"
        ]
    )

    return {
        "total_items": total_items,
        "pending_items": pending_items,
        "reviewed_items": reviewed_items,
        "high_priority_pending": high_priority,
        "medium_priority_pending": medium_priority,
        "low_priority_pending": low_priority,
    }


def clear_old_queue_items(days_old: int = 30) -> int:
    """
    Clear old items from the queue to prevent memory bloat.

    Args:
        days_old: Remove items older than this many days

    Returns:
        Number of items removed
    """

    from datetime import timedelta

    cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_old)
    cutoff_str = cutoff_date.isoformat()

    items_to_remove = []
    for queue_id, item in MODERATION_QUEUE.items():
        if item.get("created_at", "") < cutoff_str:
            items_to_remove.append(queue_id)

    for queue_id in items_to_remove:
        del MODERATION_QUEUE[queue_id]

    logger.info(f"Removed {len(items_to_remove)} old items from moderation queue")

    return len(items_to_remove)
