import json, sys
import httpx

payload = {
    "code": "print('Hello')",
    "language": "python",
    "filename": "test.py"
}

try:
    resp = httpx.post('http://127.0.0.1:8000/review', json=payload, timeout=15.0)
    print('Status:', resp.status_code)
    print('Response:', resp.text)
except Exception as e:
    print('Error:', e, file=sys.stderr)
