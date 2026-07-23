from pathlib import Path
from pypdf import PdfReader

p = Path(r'C:\Users\bhask\OneDrive\Documents\ROR\RoR-9x14-Brochure_30th-May-Final-lr_compressed.pdf')
reader = PdfReader(str(p))
print('pages', len(reader.pages))
for i, page in enumerate(reader.pages[:3]):
    text = page.extract_text() or ''
    print('page', i + 1, 'chars', len(text))
    print(text[:800])
    print('---')
