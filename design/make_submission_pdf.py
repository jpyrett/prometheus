"""Turn design/weekN_submission.md into design/weekN_submission.pdf (default: week 4).

    python design/make_submission_pdf.py        # week 4
    python design/make_submission_pdf.py 3      # week 3
    python design/make_submission_pdf.py 5 week5_submission_jpyrett.pdf   # week 5, named output
    python design/make_submission_pdf.py final  # design/final_capstone_report.md -> design/final_capstone_report.pdf

Markdown (tables, fenced code) -> HTML -> PDF with WeasyPrint. Letter paper, 0.75 inch margins. Edit the markdown, run this again.
Anything before the first line starting with "# " is left out (week 5 keeps the assignment text above its submission).
"""
import sys
from pathlib import Path

import markdown
from weasyprint import HTML

HERE = Path(__file__).parent
WEEK = sys.argv[1] if len(sys.argv) > 1 else "4"
FINAL = WEEK == "final"  # the final capstone report: its own file names, title and footer, and its sections are "# " headings
STEM = "final_capstone_report" if FINAL else f"week{WEEK}_submission"
TITLE = "Prometheus - Final Capstone Report" if FINAL else f"Prometheus Capstone: Week {WEEK} Submission"
FOOTER = "Prometheus - Final Capstone Report" if FINAL else f"Prometheus Capstone - Week {WEEK} Submission"
SOURCE = HERE / f"{STEM}.md"
TARGET = HERE / (sys.argv[2] if len(sys.argv) > 2 else f"{STEM}.pdf")

CSS = """
@page { size: Letter; margin: 0.75in 0.8in 0.8in 0.8in;
        @bottom-center { content: "FOOTER_TEXT - page " counter(page) " of " counter(pages); font: 8pt 'DejaVu Sans', sans-serif; color: #666; } }
body { font: 10pt/1.38 'DejaVu Serif', Georgia, serif; color: #1a1a1a; }
h1 { font: bold 17pt 'DejaVu Sans', sans-serif; margin: 0 0 4pt 0; color: #14305c; border-bottom: 2px solid #14305c; padding-bottom: 4pt; }
h1 + p { color: #444; margin-top: 2pt; font-size: 9.5pt; }
h2 { font: bold 12pt 'DejaVu Sans', sans-serif; margin: 15pt 0 4pt 0; color: #14305c; break-after: avoid; }
p { margin: 4pt 0 5pt 0; text-align: left; }
ul { margin: 3pt 0 5pt 0; padding-left: 16pt; } li { margin: 2pt 0; }
code { font: 8.4pt 'DejaVu Sans Mono', monospace; background: #f1f3f6; padding: 0 2pt; border-radius: 2pt; }
pre { font: 7.2pt/1.3 'DejaVu Sans Mono', monospace; background: #f1f3f6; border: 1px solid #d5d9e0; border-left: 3px solid #14305c; padding: 5pt 7pt;
      margin: 5pt 0 7pt 0; white-space: pre-wrap; word-break: break-word; break-inside: avoid; }
pre code { background: none; padding: 0; font-size: inherit; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0 8pt 0; font: 8.8pt/1.3 'DejaVu Sans', sans-serif; }
tr { break-inside: avoid; }
thead { display: table-header-group; }
th { background: #14305c; color: white; text-align: left; padding: 3.5pt 5pt; font-weight: bold; }
td { border: 1px solid #c9ced6; padding: 3.5pt 5pt; vertical-align: top; }
tr:nth-child(even) td { background: #f6f7f9; }
strong { color: #000; }
img { max-width: 100%; display: block; margin: 6pt auto 8pt auto; }
"""

FINAL_CSS = """
h1 { font-size: 13pt; margin: 14pt 0 4pt 0; break-after: avoid; }
body > h1:first-child { font-size: 18pt; margin-top: 0; }
h1 + p { color: #555; font-size: 9pt; break-after: avoid; }
img { break-inside: avoid; border: 1px solid #d5d9e0; }
p:has(+ pre) { break-after: avoid; }
p:has(> img) { break-inside: avoid; text-align: center; color: #555; font-size: 8.8pt; }
"""


def main() -> None:
    text = SOURCE.read_text(encoding="utf-8")
    text = text[text.index("# "):] if text.startswith("# ") else text[text.index("\n# ") + 1:]
    body = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists"])
    html = f"<html><head><meta charset='utf-8'><title>{TITLE}</title><style>{CSS.replace('FOOTER_TEXT', FOOTER)}{FINAL_CSS if FINAL else ''}</style></head><body>{body}</body></html>"
    HTML(string=html, base_url=str(HERE)).write_pdf(TARGET)
    print(f"wrote {TARGET}")


if __name__ == "__main__":
    main()
