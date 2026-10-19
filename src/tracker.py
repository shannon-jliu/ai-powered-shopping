"""
Temporal Voting and Checklist State Tracker
Distinguishes between items SPOTTED ON SHELF vs. physically IN CART,
with toggle and uncheck support.
Assigned: Paige Yun (phy6) & Shannon Liu (sjl356)
"""

from typing import List, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class ChecklistItem:
    name: str
    raw_text: str
    quantity: Optional[str] = None
    unit: Optional[str] = None
    found: bool = False              # In Cart
    in_view: bool = False            # Spotted on shelf
    consecutive_frames: int = 0
    confirmed_at: Optional[float] = None
    last_seen_at: Optional[float] = None
    aliases: List[str] = field(default_factory=list)


class ChecklistTracker:
    """
    Checklist state machine with temporal aggregation.
    """

    def __init__(self, target_items: List[dict], confirmation_threshold: int = 3):
        self.confirmation_threshold = confirmation_threshold
        pass

    def update(self, detected_names: List[str]) -> List[str]:
        """Updates in-view status and returns newly spotted items."""
        pass

    def toggle_item(self, index_1_based: int) -> Optional[str]:
        """Toggles in-cart status of item."""
        pass

    def confirm_spotted_to_cart(self) -> List[str]:
        """Confirms spotted item to cart."""
        pass

    def get_summary(self) -> Dict:
        """Returns checklist summary."""
        pass
