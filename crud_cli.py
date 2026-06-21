import sys
from item_db import add_item, get_item, get_items, update_item, delete_item
from history_db import add_interaction, get_history

def prompt_action():
    print("\n--- Item CRUD CLI ---")
    print("1. Add item")
    print("2. Get item by ID")
    print("3. List all items")
    print("4. Update item")
    print("5. Delete item")
    print("6. Exit")
    print("7. Show interaction history")
    print("8. Run test suite")
    return input("Choose an option (1-8): ").strip()

def add_item_flow():
    name = input("Item name: ").strip()
    description = input("Description (optional): ").strip() or None
    item_id = add_item(name, description)
    result = f"Added item with ID {item_id}."
    print(result)
    add_interaction(f"Add item: name={name}, description={description}", result)

def get_item_flow():
    try:
        item_id = int(input("Enter item ID: ").strip())
    except ValueError:
        print("Invalid ID.")
        return
    item = get_item(item_id)
    if item:
        result = f"Item {item_id}: {item['name']} - {item['description']}"
    else:
        result = f"Item {item_id} not found."
    print(result)
    add_interaction(f"Get item ID {item_id}", result)

def list_items_flow():
    items = get_items()
    if not items:
        result = "No items found."
        print(result)
    else:
        result = "Items:"
        for it in items:
            result += f"\n  {it['id']}: {it['name']} - {it['description']}"
        print(result)
    add_interaction("List items", result)

def update_item_flow():
    try:
        item_id = int(input("Enter item ID to update: ").strip())
    except ValueError:
        print("Invalid ID.")
        return
    name = input("New name (leave blank to keep unchanged): ").strip() or None
    description = input("New description (leave blank to keep unchanged): ").strip() or None
    success = update_item(item_id, name, description)
    if success:
        result = f"Item {item_id} updated."
    else:
        result = f"Item {item_id} not found or no fields provided."
    print(result)
    add_interaction(f"Update item {item_id}: name={name}, description={description}", result)

def delete_item_flow():
    try:
        item_id = int(input("Enter item ID to delete: ").strip())
    except ValueError:
        print("Invalid ID.")
        return
    success = delete_item(item_id)
    if success:
        result = f"Item {item_id} deleted."
    else:
        result = f"Item {item_id} not found."
    print(result)
    add_interaction(f"Delete item {item_id}", result)
def show_history_flow():
    """Fetch and pretty‑print interaction history."""
    records = get_history()
    if not records:
        print("No history records yet.")
        return
    print("\n--- Interaction History ---")
    for rec in records:
        print(f"[{rec['timestamp']}]")
        print(f"  User:      {rec['user_message']}")
        print(f"  Assistant: {rec['assistant_message']}")
        print("-" * 40)

def run_tests_flow():
    """Execute the unittest suite (test_crud.py) and display results in the CLI."""
    import subprocess, sys, pathlib
    test_path = pathlib.Path(__file__).with_name('test_crud.py')
    if not test_path.is_file():
        print("Test file not found.")
        return
    print("\nRunning test suite...\n")
    result = subprocess.run([sys.executable, '-m', 'unittest', str(test_path)], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Errors:\n", result.stderr)

def main():
    while True:
        choice = prompt_action()
        if choice == "1":
            add_item_flow()
        elif choice == "2":
            get_item_flow()
        elif choice == "3":
            list_items_flow()
        elif choice == "4":
            update_item_flow()
        elif choice == "5":
            delete_item_flow()
        elif choice == "6":
            print("Goodbye!")
            break
        elif choice == "7":
            show_history_flow()
        elif choice == "8":
            run_tests_flow()
        else:
            print("Invalid choice, try again.")

if __name__ == "__main__":
    main()
