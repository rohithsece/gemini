```python
import sqlite3
from document_store import save_document

def create_product(conn: sqlite3.Connection, title: str, category: str, price: float, stock_count: int) -> str:
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO products (title, category, price, stock_count) VALUES (?, ?, ?, ?)",
            (title, category, price, stock_count),
        )
        conn.commit()
        result = f"[CREATE_PRODUCT] Success: title='{title}', category='{category}', price={price}, stock_count={stock_count}"
    except Exception as e:
        result = f"[CREATE_PRODUCT] Failure: {e}"
    save_document(result, operation="create_product")
    return result

def read_products(conn: sqlite3.Connection, category: str, price: float) -> list:
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT * FROM products WHERE category=? AND price<?",
            (category, price),
        )
        rows = cur.fetchall()
        result = [dict(zip([desc[0] for desc in cur.description], row)) for row in rows]
    except Exception as e:
        result = f"[READ_PRODUCTS] Failure: {e}"
    save_document(result, operation="read_products")
    return result

def update_product(conn: sqlite3.Connection, title: str, stock_count: int) -> str:
    cur = conn.cursor()
    try:
        cur.execute(
            "UPDATE products SET stock_count=? WHERE title=?",
            (stock_count, title),
        )
        conn.commit()
        result = f"[UPDATE_PRODUCT] Success: title='{title}', stock_count={stock_count}"
    except Exception as e:
        result = f"[UPDATE_PRODUCT] Failure: {e}"
    save_document(result, operation="update_product")
    return result

def delete_product(conn: sqlite3.Connection, title: str) -> str:
    cur = conn.cursor()
    try:
        cur.execute("DELETE FROM products WHERE title=?", (title,))
        conn.commit()
        result = f"[DELETE_PRODUCT] Success: title='{title}'"
    except Exception as e:
        result = f"[DELETE_PRODUCT] Failure: {e}"
    save_document(result, operation="delete_product")
    return result

def main(conn: sqlite3.Connection):
    # Create product
    return create_product(conn, title="Wireless Headphones", category="Electronics", price=89.99, stock_count=150)

    # Read products
    # return read_products(conn, category="Electronics", price=100.0)

    # Update product
    # return update_product(conn, title="Wireless Headphones", stock_count=200)

    # Delete product
    # return delete_product(conn, title="Wireless Headphones")

# Create database and table
def create_table(conn: sqlite3.Connection):
    cur =