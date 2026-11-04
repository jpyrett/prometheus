"""Turn design/weekN_submission.md into design/weekN_submission.pdf (default: week 4).

    python design/make_submission_pdf.py        # week 4
    python design/make_submission_pdf.py 3      # week 3

Markdown (tables, fenced code) -> HTML -> PDF with WeasyPrint. Letter paper, 0.75 inch margins. Edit the markdown, run this again.
"""
import sys
from pathlib import Path

import markdown
from weasyprint import HTML

HERE = Path(__file__).parent
WEEK = sys.argv[1] if len(sys.argv) > 1 else "4"
SOURCE, TARGET = HERE / f"week{WEEK}_submission.md", HERE / f"week{WEEK}_submission.pdf"

CSS = """
@page { size: Letter; margin: 0.75in 0.8in 0.8in 0.8in;
        @bottom-center { content: "Prometheus Capstone - Week WEEK_NUMBER Submission - page " counter(page) " of " counter(pages); font: 8pt 'DejaVu Sans', sans-serif; color: #666; } }
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


def main() -> None:
    body = markdown.markdown(SOURCE.read_text(encoding="utf-8"), extensions=["tables", "fenced_code", "sane_lists"])
    html = f"<html><head><meta charset='utf-8'><title>Prometheus Capstone: Week {WEEK} Submission</title><style>{CSS.replace('WEEK_NUMBER', WEEK)}</style></head><body>{body}</body></html>"
    HTML(string=html, base_url=str(HERE)).write_pdf(TARGET)
    print(f"wrote {TARGET}")


if __name__ == "__main__":
    main()
