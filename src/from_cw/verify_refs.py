"""CC's suggestion: assert that every section cross-reference in the prose has a
matching heading. The count-the-marks check applied to cross-references.

Manual section numbering with secnumdepth=-1 means a literal "S 4.9" in prose is
not a \\ref -- LaTeX will never complain about a dangling one. This is the only
thing that would.
"""
import re, pathlib, sys
FILES = ["00_abstract","01_introduction","02_methods","04_results",
         "05_results","06_discussion","07_references"]
base = pathlib.Path("/home/claude/work/paper")
text = "\n".join((base/f"{f}.md").read_text() for f in FILES)

heads = set()
# NB: headings read "# 1. Introduction" and "## 4.9 The floor ..." -- the
# top-level ones carry a trailing period. A regex requiring whitespace right
# after the number silently drops all six of them and then reports every
# top-level cross-reference as dangling. That is an instrument limitation
# reported as a document defect, which is the failure this script exists to
# prevent, so it is spelled out here.
for m in re.finditer(r'^#{1,3}\s+(\d+)(?:\.(\d+))?[.\s]', text, flags=re.M):
    heads.add(m.group(1) if m.group(2) is None else f"{m.group(1)}.{m.group(2)}")
refs = {}
for m in re.finditer(r'§\s?(\d+(?:\.\d+)?)', text):
    refs.setdefault(m.group(1), 0)
    refs[m.group(1)] += 1

print(f"headings found ({len(heads)}): {', '.join(sorted(heads, key=lambda s:[int(x) for x in s.split('.')]))}")
print(f"distinct cross-references ({len(refs)}), {sum(refs.values())} occurrences")
dangling = {r:n for r,n in refs.items() if r not in heads}
unref = sorted(h for h in heads if h not in refs and "." in h)
if dangling:
    print("\nDANGLING -- cited but no such heading:")
    for r,n in sorted(dangling.items()): print(f"  §{r}  ({n} occurrence{'s' if n>1 else ''})")
else:
    print("\nOK  every cross-reference resolves to a heading")
if unref:
    print(f"\nnever cross-referenced (not an error): {', '.join(unref)}")
sys.exit(1 if dangling else 0)
