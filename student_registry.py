import sqlite3
import os
import sys
from pathlib import Path

# Define database path (will be created in the same folder as this script)
DB_PATH = Path(__file__).with_name('student_registry.db')

def init_db():
    """Create the students table if it doesn't exist."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                grade REAL
            )
        ''')
        conn.commit()

def create_student(name: str, grade: float) -> int:
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(
            'INSERT INTO students (name, grade) VALUES (?, ?)',
            (name, grade)
        )
        conn.commit()
        return cursor.lastrowid

def list_students():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute('SELECT id, name, grade FROM students ORDER BY id')
        rows = cursor.fetchall()
        if not rows:
            print('\n[Info] No students found in the registry.\n')
            return
        # Determine column widths for pretty printing
        id_width = max(len('ID'), max((len(str(r[0])) for r in rows), default=0)
        name_width = max(len('Name'), max((len(r[1]) for r in rows), default=0)
        grade_width = max(len('Grade'), 5)
        header = f"{'ID'.ljust(id_width)}  {'Name'.ljust(name_width)}  {'Grade'.ljust(grade_width)}"
        print('\n' + header)
        print('-' * len(header))
        for row in rows:
            print(f"{str(row[0]).ljust(id_width)}  {row[1].ljust(name_width)}  {str(row[2]).ljust(grade_width)}")
        print()

def update_student(student_id: int, new_grade: float):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(
            'UPDATE students SET grade = ? WHERE id = ?',
            (new_grade, student_id)
        )
        conn.commit()
        if cursor.rowcount == 0:
            print(f"[Error] No student with ID {student_id} found.")
        else:
            print(f"[Success] Student ID {student_id} updated with new grade {new_grade}.")

def delete_student(student_id: int):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute('DELETE FROM students WHERE id = ?', (student_id,))
        conn.commit()
        if cursor.rowcount == 0:
            print(f"[Error] No student with ID {student_id} found.")
        else:
            print(f"[Success] Student ID {student_id} has been removed.")

def prompt_float(prompt_msg: str) -> float:
    while True:
        try:
            value = float(input(prompt_msg))
            return value
        except ValueError:
            print('[Error] Please enter a valid numeric value.')

def prompt_int(prompt_msg: str) -> int:
    while True:
        try:
            value = int(input(prompt_msg))
            return value
        except ValueError:
            print('[Error] Please enter a valid integer.')

def main_menu():
    menu = """
Student Registry - Menu
1. Create student (C)
2. List all students (R)
3. Update student grade (U)
4. Delete student (D)
5. Exit
Choose an option (1-5): """
    while True:
        choice = input(menu).strip()
        if choice == '1':
            name = input('Enter student full name: ').strip()
            grade = prompt_float('Enter student grade (numeric): ')
            new_id = create_student(name, grade)
            print(f"[Success] Student added with ID {new_id}.\n")
        elif choice == '2':
            list_students()
        elif choice == '3':
            sid = prompt_int('Enter student ID to update: ')
            new_grade = prompt_float('Enter new grade: ')
            update_student(sid, new_grade)
        elif choice == '4':
            sid = prompt_int('Enter student ID to delete: ')
            delete_student(sid)
        elif choice == '5':
            print('Exiting. Goodbye!')
            break
        else:
            print('[Error] Invalid selection. Please choose 1-5.')

if __name__ == '__main__':
    init_db()
    try:
        main_menu()
    except KeyboardInterrupt:
        print('\nInterrupted. Exiting.')
        sys.exit(0)
