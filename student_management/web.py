import os
import sqlite3
from pathlib import Path
import json
from flask import Flask, request, jsonify, session
from document_store import save_document
from werkzeug.security import generate_password_hash, check_password_hash

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "student_database.db"
SECRET_KEY = os.urandom(24)

# Flask app initialization
app = Flask(__name__)  # create Flask instance
app.secret_key = SECRET_KEY

# Simple health check / landing page
@app.route("/", methods=["GET"])
def index():
    return "<h2>Student Management API is running</h2>", 200

# ------------------------------------------------------------
# DB helpers
# ------------------------------------------------------------
def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    cur = conn.cursor()
    # students table
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            password_hash TEXT NOT NULL
        )
        """
    )
    # records table
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            course TEXT NOT NULL,
            grade TEXT NOT NULL,
            FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE CASCADE
        )
        """
    )
    conn.commit()
    conn.close()

# ------------------------------------------------------------
# Authentication helpers
# ------------------------------------------------------------
def create_demo_user():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM students WHERE id = ?", (10045,))
    if not cur.fetchone():
        pwd_hash = generate_password_hash("SecurePassword123")
        cur.execute(
            "INSERT INTO students (id, name, password_hash) VALUES (?, ?, ?)",
            (10045, "John Doe", pwd_hash),
        )
        conn.commit()
    conn.close()

def login_student(student_id: int, password: str):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT name, password_hash FROM students WHERE id = ?", (student_id,))
    row = cur.fetchone()
    conn.close()
    if row and check_password_hash(row["password_hash"], password):
        return True, row["name"]
    return False, ""

# ------------------------------------------------------------
# Routes
# ------------------------------------------------------------
# DB initialization moved to main block (see __main__)
@app.route("/login", methods=["POST"])
def api_login():
    data = request.json or {}
    try:
        student_id = int(data.get("student_id", ""))
    except ValueError:
        return jsonify({"error": "Invalid student_id"}), 400
    password = data.get("password", "")
    ok, name = login_student(student_id, password)
    if ok:
        session["student_id"] = student_id
        session["name"] = name
        return jsonify({"message": f"Login successful. Welcome, {name}!"})
    return jsonify({"error": "Invalid credentials"}), 401

def require_login(fn):
    def wrapper(*args, **kwargs):
        if "student_id" not in session:
            return jsonify({"error": "Authentication required"}), 401
        return fn(*args, **kwargs)
    wrapper.__name__ = fn.__name__
    return wrapper

# CREATE
@app.route("/records", methods=["POST"])
@require_login
def api_create():
    data = request.json or {}
    course = data.get("course")
    grade = data.get("grade")
    if not course or not grade:
        return jsonify({"error": "Missing course or grade"}), 400
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO records (student_id, course, grade) VALUES (?, ?, ?)",
        (session["student_id"], course, grade),
    )
    conn.commit()
    conn.close()
    sql = f"INSERT INTO records (student_id, course, grade) VALUES ({session['student_id']}, '{course}', '{grade}')"
    save_document(sql, "create_query")
    return jsonify({"message": "Record added successfully"})

# READ
@app.route("/records", methods=["GET"])
@require_login
def api_read():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, course, grade FROM records WHERE student_id = ? ORDER BY id",
        (session["student_id"],),
    )
    rows = [dict(row) for row in cur.fetchall()]
    conn.close()
    return jsonify(rows)

# UPDATE
@app.route("/records/<int:record_id>", methods=["PUT"])
@require_login
def api_update(record_id):
    data = request.json or {}
    new_grade = data.get("grade")
    if not new_grade:
        return jsonify({"error": "Missing grade"}), 400
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "UPDATE records SET grade = ? WHERE id = ? AND student_id = ?",
        (new_grade, record_id, session["student_id"]),
    )
    if cur.rowcount == 0:
        conn.close()
        return jsonify({"error": "Record not found"}), 404
    conn.commit()
    conn.close()
    sql = f"UPDATE records SET grade = '{new_grade}' WHERE id = {record_id} AND student_id = {session['student_id']}"
    save_document(sql, "update_query")
    return jsonify({"message": f"Record {record_id} updated"})

# DELETE
@app.route("/records/<int:record_id>", methods=["DELETE"])
@require_login
def api_delete(record_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "DELETE FROM records WHERE id = ? AND student_id = ?",
        (record_id, session["student_id"]),
    )
    if cur.rowcount == 0:
        conn.close()
        return jsonify({"error": "Record not found"}), 404
    conn.commit()
    conn.close()
    return jsonify({"message": f"Record {record_id} deleted"})

# ------------------------------------------------------------
# Code Runner and Explorer UI
# ------------------------------------------------------------
@app.route("/code", methods=["GET", "POST"])
def code_explorer():
    queries_dir = BASE_DIR / "queries"
    queries_dir.mkdir(exist_ok=True)
    
    # Handle File Upload or Raw Code Submission
    if request.method == "POST":
        action = request.form.get("action")
        if action == "upload":
            filename = request.form.get("filename")
            code_content = request.form.get("code")
            if filename and code_content:
                # Ensure simple validation of filename
                filename = "".join(c for c in filename if c.isalnum() or c in "._-")
                if not filename.endswith(".py"):
                    filename += ".py"
                file_path = queries_dir / filename
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(code_content)
                return jsonify({"status": "success", "message": f"Saved {filename} successfully!"})
            return jsonify({"status": "error", "message": "Missing file details"}), 400
            
        elif action == "run":
            filename = request.form.get("filename")
            file_path = queries_dir / filename
            if file_path.exists():
                try:
                    import sys
                    import importlib.util
                    # Ensure the parent directory is in path so sub-imports like `from .document_store` can resolve (or mock/handle them)
                    queries_parent = str(queries_dir.parent)
                    if queries_parent not in sys.path:
                        sys.path.insert(0, queries_parent)
                    # Load dynamically
                    spec = importlib.util.spec_from_file_location(f"student_management.queries.{filename[:-3]}", str(file_path))
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[f"student_management.queries.{filename[:-3]}"] = module
                    spec.loader.exec_module(module)
                    
                    # Run any entry points like main() or matching functions
                    run_output = "Module executed successfully."
                    funcs = [name for name, val in module.__dict__.items() if callable(val)]
                    # Attempt to run functions matching the name pattern or a standard run/main function
                    target_func = None
                    for name in funcs:
                        if name in ["main", "run", filename[:-3]]:
                            target_func = getattr(module, name)
                            break
                    if not target_func and funcs:
                        # Fallback to the first defined function that isn't built-in
                        user_funcs = [f for f in funcs if not f.startswith("__")]
                        if user_funcs:
                            target_func = getattr(module, user_funcs[0])
                    
                    if target_func:
                        try:
                            # If function requires connection or student params, supply test inputs
                            import inspect
                            sig = inspect.signature(target_func)
                            params = list(sig.parameters.keys())
                            
                            args = {}
                            if "conn" in params:
                                conn = sqlite3.connect(str(BASE_DIR / "student_database.db"))
                                conn.row_factory = sqlite3.Row
                                args["conn"] = conn
                            if "student_id" in params:
                                args["student_id"] = 10045
                            if "course" in params:
                                args["course"] = "Biology 101"
                            if "grade" in params:
                                args["grade"] = "A-"
                            if "record_id" in params:
                                args["record_id"] = 1
                            if "new_grade" in params:
                                args["new_grade"] = "A+"
                            
                            # Run target function
                            res = target_func(**args)
                            if "conn" in args:
                                args["conn"].close()
                                
                            run_output += f"\nExecuted {target_func.__name__}(): {res}"
                        except Exception as e:
                            run_output += f"\nError running {target_func.__name__}(): {str(e)}"
                    else:
                        run_output += "\nNo callable function found to run."
                        
                    return jsonify({"status": "success", "output": run_output})
                except Exception as e:
                    return jsonify({"status": "error", "output": f"Execution Error: {str(e)}"}), 500
            return jsonify({"status": "error", "message": "File not found"}), 404

    # GET Request: Render HTML page
    files = [f.name for f in queries_dir.glob("*.py")]
    
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Code Explorer & Runner</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Fira+Code:wght@400;500&display=swap" rel="stylesheet">
        <style>
            :root {
                --bg-primary: #0f172a;
                --bg-secondary: #1e293b;
                --accent: #6366f1;
                --accent-hover: #4f46e5;
                --text-main: #f8fafc;
                --text-muted: #94a3b8;
                --border: #334155;
                --success: #10b981;
                --error: #ef4444;
            }
            * { box-sizing: border-box; margin: 0; padding: 0; }
            body {
                font-family: 'Inter', sans-serif;
                background-color: var(--bg-primary);
                color: var(--text-main);
                padding: 2rem;
                display: flex;
                flex-direction: column;
                min-height: 100vh;
            }
            header {
                margin-bottom: 2rem;
                border-bottom: 1px solid var(--border);
                padding-bottom: 1rem;
            }
            h1 { font-weight: 700; color: var(--text-main); margin-bottom: 0.5rem; }
            p { color: var(--text-muted); font-size: 0.95rem; }
            .container {
                display: grid;
                grid-template-columns: 300px 1fr;
                gap: 2rem;
                flex: 1;
            }
            .sidebar {
                background: var(--bg-secondary);
                border: 1px solid var(--border);
                border-radius: 12px;
                padding: 1.5rem;
                display: flex;
                flex-direction: column;
                gap: 1.5rem;
            }
            .sidebar h2 { font-size: 1.1rem; border-bottom: 1px solid var(--border); padding-bottom: 0.5rem; }
            .file-list { list-style: none; display: flex; flex-direction: column; gap: 0.75rem; }
            .file-item {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 0.5rem 0.75rem;
                background: var(--bg-primary);
                border: 1px solid var(--border);
                border-radius: 6px;
                font-family: 'Fira Code', monospace;
                font-size: 0.85rem;
            }
            .btn {
                background: var(--accent);
                color: white;
                border: none;
                padding: 0.4rem 0.8rem;
                border-radius: 4px;
                cursor: pointer;
                font-family: 'Inter', sans-serif;
                font-weight: 600;
                font-size: 0.8rem;
                transition: background 0.2s;
            }
            .btn:hover { background: var(--accent-hover); }
            .main-content {
                display: flex;
                flex-direction: column;
                gap: 2rem;
            }
            .card {
                background: var(--bg-secondary);
                border: 1px solid var(--border);
                border-radius: 12px;
                padding: 1.5rem;
            }
            .card h3 { margin-bottom: 1rem; font-size: 1.2rem; }
            .form-group { display: flex; flex-direction: column; gap: 0.5rem; margin-bottom: 1rem; }
            label { font-size: 0.85rem; color: var(--text-muted); font-weight: 600; }
            input[type="text"], textarea {
                background: var(--bg-primary);
                border: 1px solid var(--border);
                border-radius: 6px;
                padding: 0.75rem;
                color: var(--text-main);
                font-family: 'Fira Code', monospace;
                font-size: 0.9rem;
                outline: none;
            }
            textarea { height: 200px; resize: vertical; }
            input[type="text"]:focus, textarea:focus { border-color: var(--accent); }
            .output-box {
                background: #020617;
                border: 1px solid var(--border);
                border-radius: 6px;
                padding: 1rem;
                font-family: 'Fira Code', monospace;
                font-size: 0.9rem;
                color: #38bdf8;
                white-space: pre-wrap;
                min-height: 100px;
                max-height: 300px;
                overflow-y: auto;
            }
        </style>
    </head>
    <body>
        <header>
            <h1>Code Explorer & Runner</h1>
            <p>List, run, and upload custom Python queries directly into your project's queries folder.</p>
        </header>
        <div class="container">
            <div class="sidebar">
                <h2>Query Files</h2>
                <ul class="file-list">
    """
    for file in files:
        html += f"""
                    <li class="file-item">
                        <span>{file}</span>
                        <button class="btn" onclick="runCode('{file}')">Run</button>
                    </li>
        """
    html += """
                </ul>
            </div>
            <div class="main-content">
                <div class="card">
                    <h3>Upload/Save Python Query</h3>
                    <div class="form-group">
                        <label for="filename">Filename</label>
                        <input type="text" id="filename" placeholder="e.g. test_query.py">
                    </div>
                    <div class="form-group">
                        <label for="code">Python Code</label>
                        <textarea id="code" placeholder="def main():\n    return 'Hello World'"></textarea>
                    </div>
                    <button class="btn" onclick="uploadCode()">Save File</button>
                </div>
                
                <div class="card">
                    <h3>Execution Console</h3>
                    <div class="output-box" id="console">No code executed yet. Click "Run" next to any file in the sidebar.</div>
                </div>
            </div>
        </div>

        <script>
            function uploadCode() {
                const filename = document.getElementById('filename').value;
                const code = document.getElementById('code').value;
                if (!filename || !code) {
                    alert('Please enter both filename and python code content.');
                    return;
                }
                
                const formData = new FormData();
                formData.append('action', 'upload');
                formData.append('filename', filename);
                formData.append('code', code);
                
                fetch('/code', {
                    method: 'POST',
                    body: formData
                })
                .then(r => r.json())
                .then(data => {
                    alert(data.message || 'File saved successfully!');
                    location.reload();
                })
                .catch(e => {
                    alert('Error saving file.');
                });
            }

            function runCode(filename) {
                const consoleBox = document.getElementById('console');
                consoleBox.textContent = 'Running ' + filename + '...';
                
                const formData = new FormData();
                formData.append('action', 'run');
                formData.append('filename', filename);
                
                fetch('/code', {
                    method: 'POST',
                    body: formData
                })
                .then(r => r.json())
                .then(data => {
                    if (data.status === 'success') {
                        consoleBox.textContent = data.output;
                    } else {
                        consoleBox.textContent = data.output || data.message || 'Failed to run.';
                    }
                })
                .catch(e => {
                    consoleBox.textContent = 'Error executing code: ' + e;
                });
            }
        </script>
    </body>
    </html>
    """
    return html


# ------------------------------------------------------------
# Run server
# ------------------------------------------------------------
# Duplicate health check removed - using earlier definition

if __name__ == "__main__":
    # Initialize database and demo user before starting server
    init_db()
    create_demo_user()
    # Global after-request logger to capture every request/response
    @app.after_request
    def log_request(response):
        try:
            payload = request.get_json() if request.is_json else request.data.decode('utf-8')
            log_content = (
                f"Method: {request.method}\n"
                f"Path: {request.path}\n"
                f"Payload: {payload}\n"
                f"Status: {response.status}\n"
                f"Response: {response.get_data(as_text=True)}"
            )
            op_name = f"{request.method}_{request.endpoint or request.path}".replace('/', '_')
            save_document(log_content, op_name)
        except Exception:
            pass
        return response
    # Listen on all interfaces for easy access
    # Standalone mode — use port 7861 to avoid conflicting with main web.py on 7860
    app.run(host="0.0.0.0", port=7861, debug=True)
