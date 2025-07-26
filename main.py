import fitz  # PyMuPDF
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

def is_heading_candidate(span):
    text = span.get("text", "").strip()
    if not text or len(text) < 2:
        return False
    if all(c in "-–—•. " for c in text):  # bullets or lines
        return False
    if any(char.isalpha() for char in text):  # must contain letters
        return True
    return False

def extract_heading_spans(page, min_font_size=10):
    heading_spans = []
    blocks = page.get_text("dict")["blocks"]

    for block in blocks:
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "").strip()
                if not is_heading_candidate(span):
                    continue

                font_size = span.get("size", 0)
                font = span.get("font", "")
                flags = span.get("flags", 0)
                bbox = span.get("bbox", [])
                color = span.get("color", 0)

                is_bold = bool(flags & 2)
                is_caps = text.isupper()
                left_aligned = abs(bbox[0]) < 50  # close to left edge

                if font_size < min_font_size:
                    continue

                heading_spans.append({
                    "text": text,
                    "font_size": font_size,
                    "font": font,
                    "flags": flags,
                    "bbox": bbox,
                    "color": color,
                    "is_bold": is_bold,
                    "is_caps": is_caps,
                    "left_aligned": left_aligned,
                    "y": bbox[1],
                })
    return heading_spans

def assign_levels(spans):
    unique_sizes = sorted({s["font_size"] for s in spans}, reverse=True)
    size_to_level = {sz: i + 1 for i, sz in enumerate(unique_sizes)}
    for s in spans:
        s["level"] = size_to_level[s["font_size"]]
    return spans

def group_by_page(doc):
    headings_by_page = []
    for i, page in enumerate(doc):
        spans = extract_heading_spans(page)
        spans = assign_levels(spans)
        for s in spans:
            s["page"] = i + 1
        headings_by_page.extend(spans)
    return headings_by_page

def generate_json_structure(headings):
    # Sort by page and y-position to maintain logical order
    headings.sort(key=lambda x: (x["page"], x["y"]))

    output = []
    for h in headings:
        output.append({
            "text": h["text"],
            "level": h["level"],
            "page": h["page"],
            "font_size": h["font_size"],
            "bold": h["is_bold"],
            "caps": h["is_caps"],
        })
    return output

def save_to_json(data, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def extract_headings_from_pdf(pdf_path, json_output="headings_output.json"):
    if not os.path.exists(pdf_path):
        print(f"Error: PDF not found: {pdf_path}")
        return

    doc = fitz.open(pdf_path)
    headings = group_by_page(doc)
    structured = generate_json_structure(headings)
    save_to_json(structured, json_output)
    print(f"[✓] Headings extracted and saved to: {json_output}")

# ---------- Run this part ----------
if __name__ == "__main__":
    # Example usage:
    # python extract_headings.py input.pdf
    if len(sys.argv) != 2:
        print("Usage: python extract_headings.py <input_pdf>")
        sys.exit(1)
    pdf_path = sys.argv[1]
    output_json = Path(pdf_path).stem + "_headings.json"
    extract_headings_from_pdf(pdf_path, output_json)

