"""
Recipe NLP Parsing Module
Uses ingredient-parser-nlp to convert raw natural language recipe lines
into clean ingredient entity names and quantities for open-vocabulary visual detection.
"""

from dataclasses import dataclass
from typing import List, Optional
import re

try:
    from ingredient_parser import parse_ingredient
    HAS_PARSER = True
except ImportError:
    HAS_PARSER = False


@dataclass
class IngredientItem:
    name: str              # Clean canonical name (e.g., "red onion")
    raw_text: str          # Original recipe line
    quantity: Optional[str] = None
    unit: Optional[str] = None


def clean_text_heuristically(line: str) -> str:
    """Fallback rule-based cleaner if parser is unavailable or returns empty."""
    # Strip leading numbers/fractions and standard units
    text = re.sub(r"^[\d\s/.,\-\u00BC-\u00BE\u2150-\u215E]+", "", line).strip()
    common_units = [
        "cups", "cup", "tablespoons", "tablespoon", "tbsp", "teaspoons", "teaspoon", "tsp",
        "ounces", "ounce", "oz", "pounds", "pound", "lbs", "lb", "grams", "gram", "g",
        "cloves", "clove", "cans", "can", "bunch", "bunches", "heads", "head", "pieces", "piece"
    ]
    tokens = text.split()
    if tokens and tokens[0].lower() in common_units:
        tokens = tokens[1:]
    # Remove preparation modifiers after commas
    cleaned = " ".join(tokens).split(",")[0].strip().lower()
    return cleaned if cleaned else line.strip().lower()


def parse_recipe_line(line: str) -> IngredientItem:
    """Parses a single ingredient line into an IngredientItem."""
    line = line.strip()
    if not line:
        return None

    if HAS_PARSER:
        try:
            parsed = parse_ingredient(line)
            # In ingredient-parser-nlp, parsed.name is a list of IngredientText objects
            if parsed.name and len(parsed.name) > 0:
                clean_name = parsed.name[0].text.strip().lower()
            else:
                clean_name = clean_text_heuristically(line)

            qty = str(parsed.amount[0].quantity) if parsed.amount and len(parsed.amount) > 0 else None
            unit = str(parsed.amount[0].unit) if parsed.amount and len(parsed.amount) > 0 else None

            return IngredientItem(
                name=clean_name,
                raw_text=line,
                quantity=qty,
                unit=unit
            )
        except Exception:
            pass

    # Fallback to heuristic
    clean_name = clean_text_heuristically(line)
    return IngredientItem(name=clean_name, raw_text=line)


def parse_recipe_text(text: str) -> List[IngredientItem]:
    """Parses multiline recipe text and returns a list of IngredientItem objects."""
    items = []
    seen_names = set()
    for line in text.strip().split("\n"):
        # Skip empty lines, comments, and section headers
        if (
            not line
            or line.startswith("#")
            or line.startswith("//")
            or line.lower().startswith("ingredient")
            or line.endswith(":")
        ):
            continue
        parsed_item = parse_recipe_line(line)
        if parsed_item and parsed_item.name:
            if parsed_item.name not in seen_names:
                items.append(parsed_item)
                seen_names.add(parsed_item.name)
    return items


COMMON_FOOD_ALIASES = {
    "crushed tomatoes": ["canned crushed tomatoes", "canned tomatoes", "jar of tomato sauce", "tomato sauce jar", "crushed tomatoes"],
    "tomato sauce": ["jar of tomato sauce", "tomato sauce jar", "canned tomatoes"],
    "red onion": ["red onion", "purple onion", "whole red onion"],
    "yellow onion": ["yellow onion", "whole onion"],
    "onion": ["red onion", "yellow onion", "whole onion"],
    "garlic": ["garlic bulb", "head of garlic", "garlic cloves", "whole garlic"],
    "fresh garlic": ["garlic bulb", "head of garlic", "garlic cloves"],
    "fresh basil": ["fresh basil leaves", "bunch of basil", "basil leaves"],
    "basil": ["fresh basil leaves", "bunch of basil", "basil leaves"],
    "mozzarella cheese": ["mozzarella cheese", "shredded mozzarella", "mozzarella"],
    "cheese": ["mozzarella cheese", "cheddar cheese"],
    "spaghetti": ["box of spaghetti", "spaghetti pasta box", "box of pasta", "dry spaghetti"],
    "spaghetti pasta": ["box of spaghetti", "spaghetti pasta box", "box of pasta"],
    "pasta": ["box of spaghetti", "spaghetti pasta box", "box of pasta"],
    "olive oil": ["bottle of olive oil", "olive oil"],
    "black pepper": ["black pepper container", "ground black pepper"],
    "salt": ["salt container", "salt shaker"]
}


def get_visual_aliases(name: str) -> List[str]:
    """
    Expands an ingredient name into visual recognition prompts for CLIP / YOLO-World.
    E.g., 'red onion' -> ['red onion', 'onion', 'onions', 'a red onion']
    """
    name_clean = name.lower().strip()
    if name_clean in COMMON_FOOD_ALIASES:
        return COMMON_FOOD_ALIASES[name_clean]

    # Generate heuristic variations: full name, root noun, and prefixed
    aliases = [name_clean]
    words = name_clean.split()
    if len(words) > 1:
        # Last word is often the root noun (e.g. 'yellow onion' -> 'onion')
        aliases.append(words[-1])
        # First word if noun phrase
        aliases.append(words[0])
    
    # Add natural language article if single word
    if not name_clean.startswith(("a ", "an ", "the ")):
        article = "an " if name_clean[0] in "aeiou" else "a "
        aliases.append(f"{article}{name_clean}")

    # Deduplicate while preserving order
    seen = set()
    result = []
    for a in aliases:
        if a not in seen:
            result.append(a)
            seen.add(a)
    return result


def load_recipe_file(filepath: str) -> List[IngredientItem]:
    """Loads a recipe from a text file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return parse_recipe_text(f.read())


if __name__ == "__main__":
    sample = """
    2 cups shredded mozzarella cheese, chilled
    1 red onion, finely chopped
    3 cloves fresh garlic, minced
    1 can crushed tomatoes (28 oz)
    1 bunch fresh basil leaves
    1/2 lb spaghetti pasta
    """
    parsed = parse_recipe_text(sample)
    print("=== Parsed Recipe Ingredients ===")
    for item in parsed:
        print(f" -> Name: '{item.name}' | Qty: {item.quantity} {item.unit} | Raw: '{item.raw_text}'")
