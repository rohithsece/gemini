import sys
import pathlib

# Ensure the project root is on the Python path
project_root = pathlib.Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from student_management.student_management import run

if __name__ == "__main__":
    run()
