import fitz, json, os, re, pickle
import pandas as pd

MODEL_PATH = 'heading_model.pkl'
with open(MODEL_PATH, 'rb') as mf:
    clf = pickle.load(mf)

stops = {"and","the","of","in","to","for","with","on","by","at","from"}

# ---------- feature extraction ----------

def extract_features(span):
    text, fs, fonts = span['text'], span['font_size'], span.get('font', [])
    return [
        fs,
        int(any('Bold' in f for f in fonts)),
        int(bool(re.match(r'^\d+(?:\.\d+)*\b', text))),
        int(text.endswith(':')),
        len(text.split()),
        sum(w.lower() in stops for w in text.split()) / max(1, len(text.split())),
        int(text[0].isupper()),
        int(text.endswith(('.', '?', '!')))
    ]

# ---------- heuristic heading check, returns bool + reason ---------

def heuristic_heading(text, prev_blank, next_blank, fs, max_fs):
    reasons = []
    if prev_blank or next_blank:
        reasons.append('surrounded by blank line')
    if fs >= (max_fs - 0.5):
        reasons.append(f'large font {fs}')
    if re.match(r'^\d+(?:\.\d+)*\b', text):
        reasons.append('numbered pattern')
    if text.endswith(':'):
        reasons.append('trailing colon')
    if text.strip().startswith('-') or text.strip().startswith('**'):
        reasons.append('bullet/bold prefix')
    return (bool(reasons), reasons)

# ---------- main extractor ----------

def extract_outline(pdf_path):
    doc = fitz.open(pdf_path)
    if doc.is_encrypted and not doc.authenticate(''):
        title = os.path.splitext(os.path.basename(pdf_path))[0]
        return {'title': title, 'outline': []}

    title = (doc.metadata or {}).get('title', os.path.splitext(os.path.basename(pdf_path))[0])

    sizes = [s['size'] for p in doc for b in p.get_text('dict')['blocks'] for l in b.get('lines', []) for s in l.get('spans', [])]
    max_fs, min_fs = max(sizes, default=12), min(sizes, default=10)

    outline, seen = [], set()

    for pno, page in enumerate(doc, 1):
        raw_lines = [ln for blk in page.get_text('blocks') for ln in blk[4].split('\n')]
        lines = [ln.strip() for ln in raw_lines]
        for idx, txt in enumerate(lines):
            if not txt or txt in seen:
                continue
            seen.add(txt)
            prev_blank = idx == 0 or lines[idx-1] == ''
            next_blank = idx == len(lines)-1 or lines[idx+1] == ''

            span = next((s for b in page.get_text('dict')['blocks'] for l in b.get('lines', []) for s in l.get('spans', []) if txt in s['text']), None)
            if not span:
                continue
            fs = span['size']; fonts = [span.get('font','')]

            heuristic_hit, h_reasons = heuristic_heading(txt, prev_blank, next_blank, fs, max_fs)
            reason = []
            if heuristic_hit:
                reason.extend(h_reasons)
            else:
                feats = extract_features({'text':txt,'font_size':fs,'font':fonts})
                if clf.predict(pd.DataFrame([feats]))[0] == 1:
                    reason.append('ML classifier positive')
                else:
                    continue  # not a heading

            # level selection
            if fs >= (max_fs - 0.5):
                level = 'H1'
            elif re.match(r'^\d+(?:\.\d+)*\b', txt) or txt.endswith(':'):
                level = 'H2'
            elif txt.strip().startswith('**') or txt.strip().startswith('-') or fs >= (min_fs+1.5):
                level = 'H3'
            else:
                level = 'H4'

            outline.append({
                'level': level,
                'text': txt,
                'page': pno,
                'font_size': round(fs,1),
                'reason': '; '.join(reason)
            })

    return {'title': title, 'outline': outline}

if __name__ == '__main__':
    in_dir, out_dir = 'input','output'
    os.makedirs(out_dir, exist_ok=True)
    for fn in sorted(os.listdir(in_dir)):
        if fn.lower().endswith('.pdf'):
            res = extract_outline(os.path.join(in_dir, fn))
            with open(os.path.join(out_dir, fn.replace('.pdf','.json')), 'w') as fp:
                json.dump(res, fp, indent=2)
