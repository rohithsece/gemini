# run_all_crud.py
"""
Utility script to run all CRUD operations at once to verify functionality
and test document store logging.
"""

import os
import sys
# Ensure parent directory is in path before importing queries
parent_dir = os.path.dirname(os.path.dirname(__file__))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
# Now queries and document_store can import correctly
if os.path.dirname(__file__) not in sys.path:
    sys.path.insert(0, os.path.dirname(__file__))

import sqlite3
from create_record import create_record
from read_records import read_records
from update_record import update_record
from delete_record import delete_record

def run():
    # Insert parent directory in sys.path so document_store imports correctly
    import sys
    parent_dir = os.path.dirname(os.path.dirname(__file__))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
        
    db_path = os.path.join(parent_dir, "student_database.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    output = []
    student_id = 10045 # Demo student ID
    
    try:
        output.append("=== Starting Complete CRUD Test Pipeline ===")
        
        # 1. Create
        output.append("\n1. Executing CREATE:")
        create_record(conn, student_id, "Biology 101", "A-")
        output.append("   Record created (logged to documents)")
        
        # 2. Read
        output.append("\n2. Executing READ:")
        records = read_records(conn, student_id)
        output.append(f"   Read {len(records)} records for student {student_id}")
        for r in records:
            output.append(f"      ID: {r['id']}, Course: {r['course']}, Grade: {r['grade']}")
            
        # Get the record ID of the created record if available
        record_id = records[-1]['id'] if records else None
        
        if record_id:
            # 3. Update
            output.append(f"\n3. Executing UPDATE on Record ID {record_id}:")
            update_record(conn, record_id, student_id, "A+")
            output.append("   Record updated")
            
            # Read again to verify update
            updated_records = read_records(conn, student_id)
            for r in updated_records:
                if r['id'] == record_id:
                    output.append(f"      Verified: New Grade is {r['grade']}")
            
            # 4. Delete
            output.append(f"\n4. Executing DELETE on Record ID {record_id}:")
            delete_record(conn, record_id, student_id)
            output.append("   Record deleted")
        else:
            output.append("\n   No records found. Skipping Update & Delete tests.")
            
        output.append("\n=== CRUD Pipeline Completed Successfully ===")
    except Exception as e:
        output.append(f"\nPipeline Error: {e}")
    finally:
        conn.close()
        
    return "\n".join(output)

if __name__ == "__main__":
    print(run())
