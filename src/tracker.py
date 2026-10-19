"""
Temporal Voting and Checklist State Tracker
Suppresses transient false positives (motion blur, camera jitter, packaging glare)
by requiring sustained consecutive detections before confirming an ingredient.
"""

import time
from typing import List, Dict, Optional
from dataclasses import dataclass, field


from src.recipe_parser import get_visual_aliases


@dataclass
class ChecklistItem:
    name: str
    raw_text: str
    quantity: Optional[str] = None
    unit: Optional[str] = None
    found: bool = False              # True when user has placed item in cart
    in_view: bool = False            # True when camera currently sees item on shelf
    consecutive_frames: int = 0
    confirmed_at: Optional[float] = None
    last_seen_at: Optional[float] = None
    aliases: List[str] = field(default_factory=list)


class ChecklistTracker:
    """
    State machine maintaining the active shopping checklist and distinguishing
    between items SPOTTED ON SHELF vs. physically IN CART.
    Supports user toggling / unchecking via keyboard controls.
    """

    def __init__(self, target_items: List[dict], confirmation_threshold: int = 3):
        self.confirmation_threshold = confirmation_threshold
        self.items: Dict[str, ChecklistItem] = {}
        self.item_list: List[ChecklistItem] = []

        for item in target_items:
            if isinstance(item, dict):
                name = item.get("name")
                raw = item.get("raw_text", name)
                qty = item.get("quantity")
                unit = item.get("unit")
            else:
                name = getattr(item, "name", None)
                raw = getattr(item, "raw_text", name)
                qty = getattr(item, "quantity", None)
                unit = getattr(item, "unit", None)

            if not name:
                continue

            key = name.lower().strip()
            aliases = get_visual_aliases(key)
            c_item = ChecklistItem(
                name=key,
                raw_text=raw,
                quantity=qty,
                unit=unit,
                aliases=aliases
            )
            self.items[key] = c_item
            self.item_list.append(c_item)

    def update(self, detected_names: List[str]) -> List[str]:
        """
        Updates the in-view status of items based on visual detections in the frame.
        Returns names of newly spotted items.
        """
        now = time.time()
        detected_set = {n.lower().strip() for n in detected_names}
        newly_spotted = []

        for key, item in self.items.items():
            matched_alias = None
            for d in detected_set:
                if key in d or d in key:
                    matched_alias = d
                    break
                for alias in item.aliases:
                    if alias in d or d in alias:
                        matched_alias = d
                        break
                if matched_alias:
                    break

            if matched_alias:
                item.consecutive_frames += 1
                item.last_seen_at = now
                if item.consecutive_frames >= self.confirmation_threshold:
                    if not item.in_view:
                        item.in_view = True
                        newly_spotted.append(item.name)
                        print(f" ⚡ [SHELF ALERT] '{item.name.upper()}' spotted in view! (Press SPACE to add to cart)")
            else:
                item.consecutive_frames = max(0, item.consecutive_frames - 1)
                if item.consecutive_frames == 0:
                    item.in_view = False

        return newly_spotted

    def toggle_item(self, index_1_based: int) -> Optional[str]:
        """Toggles the in-cart status of the item at the given 1-based index."""
        if 1 <= index_1_based <= len(self.item_list):
            item = self.item_list[index_1_based - 1]
            item.found = not item.found
            status = "IN CART [X]" if item.found else "UNCHECKED [ ]"
            print(f" [User Action] Item #{index_1_based} '{item.name}' set to: {status}")
            return item.name
        return None

    def confirm_spotted_to_cart(self) -> List[str]:
        """Adds any item currently in view on the shelf into the cart."""
        confirmed = []
        for item in self.item_list:
            if item.in_view and not item.found:
                item.found = True
                confirmed.append(item.name)
                print(f" >>> [CART UPDATE] '{item.name.upper()}' added to cart! <<<")
        return confirmed

    def get_summary(self) -> Dict:
        """Returns overall checklist progress."""
        total = len(self.items)
        found = sum(1 for item in self.items.values() if item.found)
        in_view_count = sum(1 for item in self.items.values() if item.in_view and not item.found)
        return {
            "total": total,
            "found": found,
            "in_view": in_view_count,
            "remaining": total - found,
            "is_complete": (found == total) and total > 0
        }

    def get_items(self) -> List[ChecklistItem]:
        """Returns the list of all checklist items."""
        return list(self.items.values())

