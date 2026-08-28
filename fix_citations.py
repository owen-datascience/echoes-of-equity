"""
One-shot: fix pre-existing citation-number errors.
[1, Table 8]  -> [20, Table 8]
[1, Table 10] -> [20, Table 10]
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from docx import Document

SRC = "Owen-Wu-Yau-IEEE (1) with Data Updated.docx"

def replace_paragraph_text(paragraph, new_text):
    for r in list(paragraph.runs):
        r._element.getparent().remove(r._element)
    paragraph.add_run(new_text)

d = Document(SRC)
n = 0
for p in d.paragraphs:
    txt = p.text
    new = txt
    if "[1, Table 8]" in new:
        new = new.replace("[1, Table 8]", "[20, Table 8]")
    if "[1, Table 10]" in new:
        new = new.replace("[1, Table 10]", "[20, Table 10]")
    if new != txt:
        replace_paragraph_text(p, new)
        n += 1
        print(f"Fixed: {new[:180]}")

d.save(SRC)
print(f"\nTotal citations fixed: {n}. Saved: {SRC}")
