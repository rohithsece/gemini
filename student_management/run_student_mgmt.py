import sys
from .db_init import init_db, get_connection
from .student_management import login_flow

# Import CRUD operation modules
from .create_student import create_student
from .read_student import read_student
from .update_student import update_student
from .delete_student import delete_student

def student_crud_menu(conn, user_id):
    """Interactive menu for student CRUD operations.

    Parameters
    ----------
    conn: sqlite3.Connection
        Active database connection.
    user_id: int
        ID of the authenticated user (not used directly but kept for parity).
    """
    while True:
        print('\n--- Student CRUD Menu ---')
        print('1. Create Student')
        print('2. Read Student')
        print('3. Update Student')
        print('4. Delete Student')
        print('5. Logout')
        choice = input('Select option: ').strip()

        if choice == '1':
            try:
                sid = int(input('New Student ID: ').strip())
                name = input('Student Name: ').strip()
                pwd = input('Password: ').strip()
                create_student(conn, sid, name, pwd)
            except ValueError:
                print('Invalid ID.')
        elif choice == '2':
            try:
                sid = int(input('Student ID to read: ').strip())
                read_student(conn, sid)
            except ValueError:
                print('Invalid ID.')
        elif choice == '3':
            try:
                sid = int(input('Student ID to update: ').strip())
                new_name = input('New name (leave blank to keep unchanged): ').strip() or None
                new_pwd = input('New password (leave blank to keep unchanged): ').strip() or None
                update_student(conn, sid, name=new_name, password=new_pwd)
            except ValueError:
                print('Invalid ID.')
        elif choice == '4':
            try:
                sid = int(input('Student ID to delete: ').strip())
                delete_student(conn, sid)
            except ValueError:
                print('Invalid ID.')
        elif choice == '5':
            print('Logging out...')
            break
        else:
            print('Invalid selection, try again.')


def run():
    # Ensure database and tables exist
    init_db()
    conn = get_connection()
    while True:
        success, user_id, _ = login_flow(conn)
        if success:
            student_crud_menu(conn, user_id)
            cont = input('Exit program? (y/n): ').strip().lower()
            if cont in ('y', 'yes'):
                break
        else:
            retry = input('Login failed. Try again? (y/n): ').strip().lower()
            if retry not in ('y', 'yes'):
                break
    conn.close()
    print('Goodbye!')

if __name__ == '__main__':
    run()
