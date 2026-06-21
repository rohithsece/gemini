import os
import unittest
import tempfile
import sqlite3
from pathlib import Path

# Import the modules under test
import importlib

# Ensure the project directory is on sys.path
project_dir = Path(__file__).parent
if str(project_dir) not in __import__('sys').path:
    __import__('sys').path.append(str(project_dir))

import item_db
import history_db

class TestItemCRUD(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for isolated DB files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_items.db"
        # Monkey‑patch the DB_PATH used by item_db
        item_db.DB_PATH = self.db_path
        # Re‑initialize the items table
        item_db.init_db()
        # Prepare a separate temp DB for history
        self.history_path = Path(self.temp_dir.name) / "test_history.db"
        history_db.DB_PATH = self.history_path
        history_db.init_history_db()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_add_and_get_item(self):
        # Add an item
        item_id = item_db.add_item("TestItem", "A description")
        self.assertIsInstance(item_id, int)
        # Retrieve it
        item = item_db.get_item(item_id)
        self.assertIsNotNone(item)
        self.assertEqual(item["name"], "TestItem")
        self.assertEqual(item["description"], "A description")

    def test_list_items(self):
        # Insert a few items
        ids = [item_db.add_item(f"Item{i}", f"Desc{i}") for i in range(3)]
        items = item_db.get_items()
        self.assertEqual(len(items), 3)
        # Ensure ordering by id asc
        retrieved_ids = [it["id"] for it in items]
        self.assertListEqual(retrieved_ids, sorted(ids))

    def test_update_item(self):
        item_id = item_db.add_item("OldName", "OldDesc")
        # Update name only
        updated = item_db.update_item(item_id, name="NewName")
        self.assertTrue(updated)
        item = item_db.get_item(item_id)
        self.assertEqual(item["name"], "NewName")
        self.assertEqual(item["description"], "OldDesc")
        # Update description only
        updated = item_db.update_item(item_id, description="NewDesc")
        self.assertTrue(updated)
        item = item_db.get_item(item_id)
        self.assertEqual(item["description"], "NewDesc")

    def test_delete_item(self):
        item_id = item_db.add_item("ToDelete", None)
        deleted = item_db.delete_item(item_id)
        self.assertTrue(deleted)
        # Verify it no longer exists
        self.assertIsNone(item_db.get_item(item_id))

    def test_history_logging(self):
        # Add an item, which should log the interaction
        item_id = item_db.add_item("HistItem", "HistDesc")
        # Manually log the interaction (same pattern used in crud_cli.py)
        result_msg = f"Added item with ID {item_id}."
        history_db.add_interaction(
            f"Add item: name=HistItem, description=HistDesc",
            result_msg,
        )
        # Verify a row exists in history DB
        rows = history_db.get_history()
        self.assertTrue(any(row["assistant_message"] == result_msg for row in rows))

if __name__ == "__main__":
    unittest.main()
