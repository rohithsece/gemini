import os
from datetime import datetime

# Directory where all documents will be stored (relative to this file)
DOCS_DIR = os.path.join(os.path.dirname(__file__), 'documents')
os.makedirs(DOCS_DIR, exist_ok=True)

def save_document(content: str, operation: str) -> None:
    """Save *content* to a timestamped text file under ``documents/``.

    The filename format is ``<operation>_YYYYMMDD_HHMMSS.txt``.
    This function is deliberately simple – it just writes plain text.
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    safe_op = operation.replace(' ', '_').lower()
    filename = f"{safe_op}_{timestamp}.txt"
    file_path = os.path.join(DOCS_DIR, filename)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    # Optional: you could print the path for debugging, but keep CLI output clean.
    # print(f'Document saved: {file_path}')
