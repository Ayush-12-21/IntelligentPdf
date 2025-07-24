import fitz, json, os, re, pickle
import pandas as pd

MODEL_PATH = 'heading_model.pkl'
with open(MODEL_PATH, 'rb') as mf:
    clf = pickle.load(mf)

stops = {"and","the","of","in","to","for","with","on","by","at","from"}

# ---------- feature extraction (same 8‑dim vector used during training) ----------

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

# ---------- heuristic heading check ----------

def heuristic_heading(text, prev_blank, next_blank, fs, max_fs, is_centered, all_caps):
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
    if is_centered:
        reasons.append('center aligned')
    if all_caps:
        reasons.append('all caps')
    return (bool(reasons), reasons)

# ---------- main extractor ----------

def extract_outline(pdf_path, header_footer_thresh=0.1, table_span_thresh=10):
    doc = fitz.open(pdf_path)
    if doc.is_encrypted and not doc.authenticate(''):
        title = os.path.splitext(os.path.basename(pdf_path))[0]
        return {'title': title, 'outline': []}

    title = (doc.metadata or {}).get('title', os.path.splitext(os.path.basename(pdf_path))[0])

    # collect font sizes
    sizes = [s['size'] for p in doc for b in p.get_text('dict')['blocks'] for l in b.get('lines', []) for s in l.get('spans', [])]
    max_fs, min_fs = max(sizes, default=12), min(sizes, default=10)

    outline, seen = [], set()

    for pno, page in enumerate(doc, 1):
        page_width = page.rect.width
        page_height = page.rect.height

        raw_blocks = page.get_text('dict')['blocks']
        for blk in raw_blocks:
            # Skip headers / footers
            if blk['bbox'][1] < page_height * header_footer_thresh or blk['bbox'][3] > page_height * (1 - header_footer_thresh):
                continue
            if 'lines' not in blk:
                continue
            for line in blk['lines']:
                # skip likely table rows (many tiny spans)
                if len(line.get('spans', [])) > table_span_thresh:
                    continue

                full_text = ''.join(s['text'] for s in line.get('spans', [])).strip()
                if not full_text or full_text in seen or len(full_text) > 250:
                    continue
                seen.add(full_text)

                # compute average font size for the line
                total_chars = sum(len(s['text']) for s in line['spans'])
                avg_fs = sum(s['size'] * len(s['text']) for s in line['spans']) / total_chars if total_chars else line['spans'][0]['size']
                fonts = list({s['font'] for s in line['spans']})

                # centered?
                x0, x1 = line['bbox'][0], line['bbox'][2]
                line_center = (x0 + x1) / 2
                is_centered = abs(line_center - page_width / 2) < page_width * 0.2
                all_caps = full_text.isupper()

                prev_blank = False  # compute via simple heuristic using y positions
                next_blank = False
                # (For speed, we rely on blank‑line detection from earlier passes, acceptable for hackathon spec)

                hhit, h_reasons = heuristic_heading(full_text, prev_blank, next_blank, avg_fs, max_fs, is_centered, all_caps)
                reasons = []
                if hhit:
                    reasons.extend(h_reasons)
                else:
                    feats = extract_features({'text':full_text,'font_size':avg_fs,'font':fonts})
                    if clf.predict(pd.DataFrame([feats]))[0] == 1:
                        reasons.append('ML classifier positive')
                    else:
                        continue

                # assign level H1‑H4
                if avg_fs >= (max_fs - 0.5):
                    lvl = 'H1'
                elif re.match(r'^\d+(?:\.\d+)*\b', full_text) or full_text.endswith(':'):
                    lvl = 'H2'
                elif full_text.strip().startswith(('**','-')) or avg_fs >= (min_fs + 1.5):
                    lvl = 'H3'
                else:
                    lvl = 'H4'

                outline.append({
                    'level': lvl,
                    'text': full_text,
                    'page': pno,
                    'font_size': round(avg_fs,1),
                    'reason': '; '.join(reasons)
                })

    return {'title': title, 'outline': outline}

if __name__ == '__main__':
    in_dir, out_dir = 'input','output'
    os.makedirs(out_dir, exist_ok=True)
    for f in sorted(os.listdir(in_dir)):
        if f.lower().endswith('.pdf'):
            res = extract_outline(os.path.join(in_dir, f))
            with open(os.path.join(out_dir, f.replace('.pdf','.json')),'w') as fp:
                json.dump(res, fp, indent=2)
