"""
Unit tests for temporal voting state tracker
"""

import unittest
from src.tracker import ChecklistTracker


class TestChecklistTracker(unittest.TestCase):

    def setUp(self):
        self.targets = [
            {"name": "onion", "raw_text": "1 yellow onion"},
            {"name": "tomatoes", "raw_text": "1 can crushed tomatoes"}
        ]
        self.tracker = ChecklistTracker(self.targets, confirmation_threshold=3)

    def test_initial_state(self):
        summary = self.tracker.get_summary()
        self.assertEqual(summary["total"], 2)
        self.assertEqual(summary["found"], 0)
        self.assertFalse(summary["is_complete"])

    def test_temporal_voting_confirmation(self):
        # Frame 1: detect onion
        newly = self.tracker.update(["onion"])
        self.assertEqual(len(newly), 0)
        self.assertFalse(self.tracker.items["onion"].found)
        self.assertEqual(self.tracker.items["onion"].consecutive_frames, 1)

        # Frame 2: detect onion
        newly = self.tracker.update(["onion"])
        self.assertEqual(len(newly), 0)
        self.assertFalse(self.tracker.items["onion"].found)
        self.assertEqual(self.tracker.items["onion"].consecutive_frames, 2)

        # Frame 3: detect onion -> reaches threshold of 3, spotted in view!
        newly = self.tracker.update(["onion"])
        self.assertIn("onion", newly)
        self.assertTrue(self.tracker.items["onion"].in_view)
        self.assertFalse(self.tracker.items["onion"].found)

        # Confirm to cart
        self.tracker.confirm_spotted_to_cart()
        self.assertTrue(self.tracker.items["onion"].found)

        # Test uncheck via toggle_item
        self.tracker.toggle_item(1)
        self.assertFalse(self.tracker.items["onion"].found)

    def test_decay_on_missed_frame(self):
        # Frame 1: hit
        self.tracker.update(["onion"])
        self.assertEqual(self.tracker.items["onion"].consecutive_frames, 1)

        # Frame 2: miss
        self.tracker.update([])
        self.assertEqual(self.tracker.items["onion"].consecutive_frames, 0)
        self.assertFalse(self.tracker.items["onion"].found)


if __name__ == "__main__":
    unittest.main()

