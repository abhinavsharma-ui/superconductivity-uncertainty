"""Extract a PDF to UTF-8 text so the paper can be verified without the source.

Console output on Windows is cp1252 and dies on the first alpha; everything
here writes to a file in UTF-8 and prints only ASCII status.
"""
import sys, io, os


def extract(pdf_path, out_path=None):
    from pypdf import PdfReader
    r = PdfReader(pdf_path)
    pages = [p.extract_text() or "" for p in r.pages]
    text = "\n\n".join(f"[[page {i}]]\n{t}" for i, t in enumerate(pages, 1))
    out_path = out_path or os.path.splitext(pdf_path)[0] + ".txt"
    io.open(out_path, "w", encoding="utf-8").write(text)
    print(f"pages {len(pages)}  words {len(text.split())}  chars {len(text)}")
    print(f"wrote {out_path}")
    return out_path


if __name__ == "__main__":
    extract(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
