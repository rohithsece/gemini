from ..document_store import save_document

def create_record(conn, student_id, course="Biology 101", grade="A-"):
    """Add a new academic record for the logged‑in student and save the result as a document."""
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO records (student_id, course, grade) VALUES (?, ?, ?)",
        (student_id, course, grade),
    )
    conn.commit()
    message = f"Record added for student {student_id}: {course} – {grade}"
    print('Record successfully added!')
    # Persist the output
    save_document(message, 'create')
