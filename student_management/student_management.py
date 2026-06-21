import sys
import getpass
from pathlib import Path

from .auth import AuthDB, hash_password
from .db_init import get_connection, init_db


def login_flow(conn) -> tuple[bool, int, str]:
    """Prompt for student ID and password, authenticate using AuthDB.
    Returns (success, student_id, student_name)."""
    auth = AuthDB(conn)
    try:
        student_id_input = input('Enter Student ID: ').strip()
        student_id = int(student_id_input)
    except ValueError:
        print('Invalid ID format.')
        return False, 0, ''
    password = getpass.getpass('Enter Password: ')
    success, name = auth.authenticate(student_id, password)
    if success:
        print(f'Login Successful! Welcome, {name}.')
        return True, student_id, name
    else:
        print('Invalid Student ID or Password.')
        return False, 0, ''


def add_record(conn, student_id):
    course = input('Enter Course Name: ').strip()
    grade = input('Enter Grade: ').strip()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO records (student_id, course, grade) VALUES (?, ?, ?)",
        (student_id, course, grade),
    )
    conn.commit()
    print('Record successfully added!')


def view_records(conn, student_id):
    cur = conn.cursor()
    cur.execute(
        "SELECT id, course, grade FROM records WHERE student_id = ? ORDER BY id", (student_id,)
    )
    rows = cur.fetchall()
    if not rows:
        print('No records found.')
        return
    print('--- Your Academic Records ---')
    print('ID | Course      | Grade')
    print('---------------------------')
    for row in rows:
        print(f"{row['id']}  | {row['course']:<10} | {row['grade']}")


def update_record(conn, student_id):
    record_id_input = input('Enter Record ID to update: ').strip()
    try:
        record_id = int(record_id_input)
    except ValueError:
        print('Invalid Record ID.')
        return
    new_grade = input('Enter new Grade: ').strip()
    cur = conn.cursor()
    cur.execute(
        "UPDATE records SET grade = ? WHERE id = ? AND student_id = ?",
        (new_grade, record_id, student_id),
    )
    if cur.rowcount == 0:
        print('Record not found or not owned by you.')
    else:
        conn.commit()
        print(f'Record ID {record_id} updated successfully!')


def delete_record(conn, student_id):
    record_id_input = input('Enter Record ID to delete: ').strip()
    try:
        record_id = int(record_id_input)
    except ValueError:
        print('Invalid Record ID.')
        return
    cur = conn.cursor()
    cur.execute(
        "DELETE FROM records WHERE id = ? AND student_id = ?",
        (record_id, student_id),
    )
    if cur.rowcount == 0:
        print('Record not found or not owned by you.')
    else:
        conn.commit()
        print(f'Record ID {record_id} deleted successfully!')


def main_menu(conn, student_id):
    while True:
        print('\n--- Main Menu ---')
        print('1. View Records (Read)')
        print('2. Add New Record (Create)')
        print('3. Update Record (Update)')
        print('4. Delete Record (Delete)')
        print('5. Logout')
        option = input('Select Option: ').strip()
        if option == '1':
            view_records(conn, student_id)
        elif option == '2':
            add_record(conn, student_id)
        elif option == '3':
            update_record(conn, student_id)
        elif option == '4':
            delete_record(conn, student_id)
        elif option == '5':
            print('Logging out...')
            break
        else:
            print('Invalid option, please try again.')


def run():
    # Ensure DB and tables exist
    init_db()
    conn = get_connection()
    # Authentication loop
    while True:
        success, student_id, _ = login_flow(conn)
        if success:
            main_menu(conn, student_id)
            # After logout, ask if they want to exit entirely
            cont = input('Do you want to exit the program? (y/n): ').strip().lower()
            if cont == 'y' or cont == 'yes':
                break
        else:
            retry = input('Try again? (y/n): ').strip().lower()
            if retry not in ('y', 'yes'):
                break
    conn.close()
    print('Goodbye!')


if __name__ == '__main__':
    run()
