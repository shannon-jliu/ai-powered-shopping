"""
Recipe NLP Parsing Module
Converts raw natural language recipe lines into clean ingredient entity names
and expands them into visual recognition prompts.
Assigned: Shannon Liu (sjl356)
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class IngredientItem:
    name: str              # Clean canonical name (e.g., 'red onion')
    raw_text: str          # Original recipe line
    quantity: Optional[str] = None
    unit: Optional[str] = None


def parse_recipe_line(line: str) -> Optional[IngredientItem]:
    """Parses a single recipe ingredient line."""
    pass


def parse_recipe_text(text: str) -> List[IngredientItem]:
    """Parses multiline recipe text and returns a list of IngredientItem objects."""
    pass


def get_visual_aliases(name: str) -> List[str]:
    """Expands an ingredient name into visual recognition prompts for CLIP / YOLO-World."""
    pass


def load_recipe_file(filepath: str) -> List[IngredientItem]:
    """Loads a recipe from a text file."""
    pass
