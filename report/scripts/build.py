"""Build and check the report without training models or changing their recipes."""

import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]


def run(command):
    """Run a build command and keep verbose output in the build directory."""
    result = subprocess.run(
        command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    log = ROOT / "build" / f"{Path(command[0]).name}.txt"
    log.write_text(result.stdout)
    if result.returncode:
        print(result.stdout[-8000:])
        raise RuntimeError(f"Command failed; see {log}")


def verify_sources():
    """Check that every copied numerical result remains present in its source."""
    evidence = json.loads((ROOT / "data" / "results.json").read_text())
    for key in ("baseline", "encoding", "advanced"):
        block = evidence[key]
        for row in block["rows"]:
            source = ROOT.parent / row.get("source", block.get("source", ""))
            if not source.is_file():
                raise ValueError(f"Missing measurement source: {source}")
            number = f"{row['f1']:.4f}"
            if number not in source.read_text():
                raise ValueError(f"Recorded value {number} absent from {source}")
            if not 0 <= row["f1"] <= 1:
                raise ValueError(f"Invalid F1: {row}")
    for row in evidence["public_submissions"]["rows"]:
        source = ROOT.parent / row["source"]
        if not source.is_file():
            raise ValueError(f"Missing AIcrowd score source: {source}")
        source_text = source.read_text()
        for metric in ("f1", "accuracy"):
            value = row[metric]
            if not 0 <= value <= 1 or f"{value:.3f}" not in source_text:
                raise ValueError(f"AIcrowd {metric} {value} absent from {source}")
        if row["file"] not in source_text:
            raise ValueError(f"AIcrowd CSV {row['file']} absent from {source}")
    tuning_source = (ROOT.parent / "docs" / "part3_results.md").read_text()
    for value in ("0.21474", "0.21403", "0.21445", "0.21463"):
        if value not in tuning_source:
            raise ValueError(f"GBDT tuning value {value} absent from its source")


def check_pdf(path):
    """Enforce the page budget, portrait orientation, and resolved references."""
    reader = PdfReader(path)
    pages = [page.extract_text() for page in reader.pages]
    if len(pages) != 3:
        raise ValueError(
            f"Expected two content pages and one reference page: {len(pages)}"
        )
    if "REFERENCES" not in pages[2].upper():
        raise ValueError("References must start on page three.")
    if "CONCLUSIONS" not in pages[1].upper():
        raise ValueError("The conclusion must finish within page two.")
    if "RESULTS" not in pages[1].upper():
        raise ValueError("The results must appear on page two.")
    for index, page in enumerate(reader.pages, start=1):
        box = page.mediabox
        if float(box.width) >= float(box.height) or (page.get("/Rotate", 0) % 360):
            raise ValueError(f"Page {index} is not portrait.")
    if re.search(r"\[\?\]|\?\?", "\n".join(pages)):
        raise ValueError("Unresolved citation or reference in the PDF.")
    latex_log = (ROOT / "build" / "report.log").read_text()
    problems = [
        line
        for line in latex_log.splitlines()
        if "Overfull" in line
        or "undefined" in line.lower()
        or "Missing character" in line
        or "Rerun to get" in line
    ]
    if problems:
        raise ValueError("LaTeX layout/reference problems:\n" + "\n".join(problems))
    summary = {
        "pages": len(pages),
        "content_pages": 2,
        "reference_pages": 1,
        "portrait_pages": 3,
        "overfull_boxes": 0,
        "unresolved_references": 0,
        "source_values_checked": True,
    }
    (ROOT / "build" / "checks.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    """Compile LaTeX, optionally render pages, and validate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--render", action="store_true", help="render page previews with gs"
    )
    args = parser.parse_args()
    (ROOT / "build").mkdir(exist_ok=True)
    verify_sources()
    run(
        [
            "latexmk",
            "-pdf",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-file-line-error",
            "-outdir=build",
            "report.tex",
        ]
    )
    pdf = ROOT / "build" / "report.pdf"
    if args.render:
        run(
            [
                "gs",
                "-q",
                "-dNOPAUSE",
                "-dBATCH",
                "-sDEVICE=png16m",
                "-r150",
                "-sOutputFile=build/page-%02d.png",
                str(pdf),
            ]
        )
    checks = check_pdf(pdf)
    shutil.copy2(pdf, ROOT / "report.pdf")
    print(json.dumps(checks, indent=2))
    print(f"Ready: {ROOT / 'report.pdf'}")


if __name__ == "__main__":
    main()
