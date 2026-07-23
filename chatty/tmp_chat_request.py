import json
import urllib.request
import urllib.error

req = urllib.request.Request(
    'http://127.0.0.1:8010/chat/',
    data=json.dumps({'db': 'sample_text_db', 'message': 'What does this document say about safe driving?', 'top_k': 3, 'openai_model': 'gpt-4o-mini'}).encode(),
    headers={'Content-Type': 'application/json'},
    method='POST'
)
try:
    with urllib.request.urlopen(req, timeout=120) as resp:
        print(resp.read().decode())
except urllib.error.HTTPError as e:
    print('HTTP', e.code)
    print(e.read().decode())
