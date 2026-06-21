import httpx
import json

# Read API key from .env
with open('.env') as f:
    for line in f:
        if line.startswith('GEMINI_API_KEY='):
            api_key = line.strip().split('=', 1)[1]
            break

model = 'gemini-2.5-flash'
url = f'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}'

# Simulate what your agents do - ask for JSON output
prompt = (
    "System Instruction: You are a code reviewer.\n\n"
    "Task: Review this code: print(hello)\n\n"
    "Strict Output Requirements: Respond with valid JSON like: "
    '{"issues": ["string"], "summary": "string"}'
)

payload = {
    'contents': [{'parts': [{'text': prompt}]}],
    'generationConfig': {
        'responseMimeType': 'application/json',
        'temperature': 0.2
    }
}

resp = httpx.post(url, json=payload, timeout=30.0)
if resp.status_code == 200:
    text = resp.json()['candidates'][0]['content']['parts'][0]['text']
    parsed = json.loads(text.strip())
    print('FULL TEST PASSED - API + JSON response working!')
    print('Model:', model)
    print('Parsed JSON:', json.dumps(parsed, indent=2))
else:
    print(f'FAILED: {resp.status_code}')
    print(resp.text[:400])
