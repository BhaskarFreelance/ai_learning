import json
import urllib.request
import urllib.error
from pathlib import Path

pdf_path = Path(r'C:\Users\bhask\OneDrive\Documents\ROR\RoR-9x14-Brochure_30th-May-Final-lr_compressed.pdf')
print('pdf_exists', pdf_path.exists(), 'size', pdf_path.stat().st_size if pdf_path.exists() else None)

boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW'
body = []
body.append(f'--{boundary}\r\n'.encode())
body.append(b'Content-Disposition: form-data; name="name"\r\n\r\n')
body.append(b'ror_pdf_test\r\n')
body.append(f'--{boundary}\r\n'.encode())
body.append(b'Content-Disposition: form-data; name="file"; filename="RoR-9x14-Brochure_30th-May-Final-lr_compressed.pdf"\r\nContent-Type: application/pdf\r\n\r\n')
body.append(pdf_path.read_bytes())
body.append(b'\r\n')
body.append(f'--{boundary}--\r\n'.encode())
payload = b''.join(body)
req = urllib.request.Request('http://127.0.0.1:8010/vector_db/create', data=payload, method='POST')
req.add_header('Content-Type', f'multipart/form-data; boundary={boundary}')
try:
    with urllib.request.urlopen(req, timeout=180) as resp:
        print(resp.read().decode())
except urllib.error.HTTPError as e:
    print('HTTP', e.code)
    print(e.read().decode())
