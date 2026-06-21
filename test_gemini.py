import httpx, json, sys
payload = {
    "code": "print('Hello')",
    "language": "python",
    "filename": "hello.py"
}
try:
    resp = httpx.post('http://127.0.0.1:8000/review', json=payload, timeout=15.0)
    print('Status:', resp.status_code)
    print('Response:', resp.text)
except Exception as e:
    print('Error:', e)
    sys.exit(1)
