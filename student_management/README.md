# Student Management System

A small **database‑driven** application that lets a student securely log in and manage their academic records (courses & grades).

The project ships both:
- **CLI version** – interactive terminal program (`student_management.py`).
- **Web version (optional)** – a minimal Flask app (`web_app/app.py`).

Both implementations use the same SQLite database (`student_database.db`) and share the authentication and DB‑initialisation helpers located in `auth.py` and `db_init.py`.

---
## Project Structure
```
student_management/               # Python package
│   auth.py          # Password hashing & AuthDB class
│   db_init.py       # DB schema creation & demo data
│   student_management.py   # CLI entry point
│   README.md        # You are reading it!
│
└── web_app/        # Minimal Flask web UI (optional)
    │   app.py
    │   templates/
    │       login.html
    │       dashboard.html
    │   static/
    │       style.css
```
---
## Getting Started (CLI)
1. **Prerequisites** – Python 3.10+ and `pip`.
2. **Install dependencies** (only the CLI needs the standard library):
   ```bash
   python -m pip install -r requirements.txt   # (empty for now)
   ```
3. **Initialise the database** (creates tables and a demo student):
   ```bash
   python -c "from student_management.db_init import init_db; init_db()"
   # You will see: Database initialized at ...
   ```
4. **Run the program**:
   ```bash
   python -m student_management.student_management
   ```
   or
   ```bash
   python student_management/student_management.py
   ```

---
## Sample Interaction (CLI)
```
$ python -m student_management.student_management
--- Main Menu ---
1. Login
2. Exit
Select Option: 1
Enter Student ID: 10045
Enter Password: **************
Login Successful! Welcome, John Doe.

--- Main Menu ---
1. View Records (Read)
2. Add New Record (Create)
3. Update Record (Update)
4. Delete Record (Delete)
5. Logout
Select Option: 2
Enter Course Name: Mathematics
Enter Grade: A
Record successfully added!

--- Main Menu ---
1. View Records (Read)
2. Add New Record (Create)
3. Update Record (Update)
4. Delete Record (Delete)
5. Logout
Select Option: 1
--- Your Academic Records ---
ID | Course      | Grade
---------------------------
1  | Mathematics | A

--- Main Menu ---
Select Option: 3
Enter Record ID to update: 1
Enter new Grade: A+
Record ID 1 updated successfully!

--- Main Menu ---
Select Option: 4
Enter Record ID to delete: 1
Record ID 1 deleted successfully!

--- Main Menu ---
Select Option: 5
Logging out...
Do you want to exit the program? (y/n): y
Goodbye!
```
*The demo student is created automatically with ID **10045**, name **John Doe**, and password **SecurePassword123**.*
---
## Web Version (Optional)
If you prefer a web UI, a tiny Flask app is provided.
### Install Flask
```bash
python -m pip install flask
```
### Run the web server
```bash
python web_app/app.py
```
Open a browser at `http://127.0.0.1:5000` and log in with the same credentials as above.

The web UI mirrors the CLI functionality – view, add, edit, and delete records.
---
## Extending the Project
- **Add more fields** (e.g., semester, credits) by extending the `records` table in `db_init.py` and updating the CRUD functions.
- **Password policies** – replace `hash_password` with `bcrypt` for stronger hashing.
- **Authentication tokens** – for a real web app, switch to session‑based auth.
- **Unit tests** – use `unittest` or `pytest` to test each CRUD function; the CLI is already fully testable because the DB layer is separated.
---
## License
MIT – feel free to adapt, extend, and ship this tiny student portal.
