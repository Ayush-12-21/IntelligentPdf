import fitz  # PyMuPDF
import json
import os
import sys
from collections import Counter
from pathlib import Path



def is_heading_candidate(span, min_font_size, min_text_length=2, max_words=8, min_caps_ratio=0.3):
    text = span.get("text", "").strip()
    # Basic filters: non-empty, enough length, contains letters
    if not text or len(text) < min_text_length:
        return False
    # Exclude decorative or bullet-like text
    if all(c in "-–—•. " for c in text):
        return False
    # Exclude full sentences / paragraphs
    if text.endswith('.') or len(text.split()) > max_words:
        return False
    # Must contain letters
    if not any(c.isalpha() for c in text):
        return False
    # Font size filter
    if span.get("size", 0) < min_font_size:
        return False
    # Uppercase ratio heuristic (headings often have more caps)
    caps = sum(1 for c in text if c.isupper())
    if caps / max(len(text), 1) < min_caps_ratio and not text.istitle():
        return False
    return True

def detect_repeated_headers(headings_by_page, top_margin=50, bottom_margin=50, tolerance=5):
    """
    Identify text spans that appear on almost every page in similar positions (likely headers/footers) and filter them.
    """
    # Collect candidate texts by (text, y-position bucket)
    counter = Counter()
    total_pages = max(h['page'] for h in headings_by_page) if headings_by_page else 0

    for h in headings_by_page:
        y = h['y']
        page = h['page']
        bucket = round(y / tolerance) * tolerance
        if y < top_margin or h['page_height'] - y < bottom_margin:
            counter[(h['text'], bucket)] += 1

    # Headers/footers appear on >80% of pages
    repeated = {text for (text, _), count in counter.items() if count / total_pages > 0.8}
    return repeated


def extract_heading_spans(page, min_font_size):
    spans = []
    page_dict = page.get_text("dict")
    page_height = page.rect.height

    for block in page_dict.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                if is_heading_candidate(span, min_font_size):
                    item = {
                        "text": span["text"].strip(),
                        "font_size": span.get("size", 0),
                        "font": span.get("font", ""),
                        "flags": span.get("flags", 0),
                        "bbox": span.get("bbox", []),
                        "y": span.get("bbox", [0, 0])[1],
                        "page_height": page_height
                    }
                    spans.append(item)
    return spans


def assign_levels(spans):
    # Unique font sizes sorted descending => larger size = higher-level heading
    unique_sizes = sorted({s['font_size'] for s in spans}, reverse=True)
    size_to_level = {size: idx + 1 for idx, size in enumerate(unique_sizes)}
    for s in spans:
        s['level'] = size_to_level[s['font_size']]
    return spans


def build_hierarchy(flat_headings):
    """
    Build nested hierarchy based on levels.
    """
    root = []
    stack = []  # will hold (level, node)

    for h in flat_headings:
        node = {"text": h['text'], "level": h['level'], "page": h['page'], "children": []}
        # Pop until finding parent level
        while stack and stack[-1][0] >= h['level']:
            stack.pop()
        if stack:
            stack[-1][1]['children'].append(node)
        else:
            root.append(node)
        stack.append((h['level'], node))
    return root


def extract_headings_from_pdf(pdf_path, min_font_size=None, json_output=None):
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(pdf_path)

    # Auto-detect a good min_font_size if not provided
    if min_font_size is None:
        # Collect all font sizes
        all_sizes = []
        for page in doc:
            for block in page.get_text("dict")["blocks"]:
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        size = span.get("size", 0)
                        if size:
                            all_sizes.append(size)
        # Use the 20th percentile as threshold
        if all_sizes:
            import numpy as np
            min_font_size = float(np.percentile(all_sizes, 20))
        else:
            min_font_size = 10.0

    headings = []
    for page_num, page in enumerate(doc, start=1):
        spans = extract_heading_spans(page, min_font_size)
        for s in spans:
            s['page'] = page_num
            headings.append(s)

    # Filter out repeated headers/footers
    repeated = detect_repeated_headers(headings)
    headings = [h for h in headings if h['text'] not in repeated]

    # Sort and assign levels
    headings.sort(key=lambda x: (x['page'], x['y']))
    headings = assign_levels(headings)

    # Build hierarchy tree
    nested = build_hierarchy(headings)

    # Prepare output
    output = {
        'path': str(Path(pdf_path).absolute()),
        'min_font_size': min_font_size,
        'headings': nested
    }

    # Save to JSON
    if json_output:
        with open(json_output, 'w', encoding='utf-8') as fp:
            json.dump(output, fp, indent=2, ensure_ascii=False)

    return output


if __name__ == '__main__':
    if len(sys.argv) < 2 or len(sys.argv) > 4:
        print("Usage: python extract_headings.py <input.pdf> [min_font_size] [output.json]")
        sys.exit(1)

    pdf_file = sys.argv[1]
    min_size = float(sys.argv[2]) if len(sys.argv) >= 3 else None
    out_file = sys.argv[3] if len(sys.argv) == 4 else Path(pdf_file).stem + '_headings.json'

    try:
        result = extract_headings_from_pdf(pdf_file, min_size, out_file)
        print(f"[✓] Extracted {len(result['headings'])} top-level headings to {out_file}")
    except Exception as e:
        print(f"Error: {e}")
