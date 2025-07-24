import fitz, os

PDF = "input/NEP2020_Rural_Development.pdf"

def debug_font_sizes(pdf_path):
    doc = fitz.open(pdf_path)

    # try decrypt
    if doc.is_encrypted:
        doc.authenticate("14062025")  # replace if needed

    print(f"\n=== DEBUG: Scanning '{os.path.basename(pdf_path)}' for font-sizes ===")
    for pno, page in enumerate(doc, start=1):
        raw = page.get_text("text")
        if not raw.strip():
            print(f"\nPage {pno:>2} raw text: <empty or image-only>\n")
        else:
            print(f"\nPage {pno:>2} raw text preview:\n{raw[:200]!r}\n")

        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    fs  = span["size"]
                    txt = span["text"].strip().replace("\n"," ")
                    if not txt:
                        continue
                    print(f"Page {pno:>2} → '{txt[:30]}' (fs={fs})")

if __name__ == "__main__":
    debug_font_sizes(PDF)
