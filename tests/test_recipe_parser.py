"""
Unit tests for recipe NLP parser
"""

import unittest
from src.recipe_parser import parse_recipe_line, parse_recipe_text


class TestRecipeParser(unittest.TestCase):

    def test_single_line_extraction(self):
        line = "2 cups shredded mozzarella cheese, chilled"
        item = parse_recipe_line(line)
        self.assertIsNotNone(item)
        self.assertIn("mozzarella", item.name)

    def test_vegetable_extraction(self):
        line = "1 medium yellow onion, diced"
        item = parse_recipe_line(line)
        self.assertIsNotNone(item)
        self.assertIn("onion", item.name)

    def test_can_packaging_extraction(self):
        line = "1 can crushed tomatoes"
        item = parse_recipe_line(line)
        self.assertIsNotNone(item)
        self.assertIn("tomatoes", item.name)

    def test_multiline_recipe(self):
        recipe = """
        Ingredients:
        1 lb pasta
        2 tbsp olive oil
        3 cloves garlic
        """
        items = parse_recipe_text(recipe)
        self.assertEqual(len(items), 3)
        names = [item.name for item in items]
        self.assertTrue(any("pasta" in n for n in names))
        self.assertTrue(any("olive oil" in n for n in names))
        self.assertTrue(any("garlic" in n for n in names))


if __name__ == "__main__":
    unittest.main()
