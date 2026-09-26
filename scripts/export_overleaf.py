#!/usr/bin/env python3
"""
export_overleaf.py
==================
Automated packager to create a ready-to-upload Overleaf zip package from the project.

This script packages:
  1. paper/paper.tex -> paper.tex (at zip root for Overleaf compilation)
  2. figures/        -> figures/ (all production figure assets)
  3. Any local bibliography or class files if present.

Usage:
  python scripts/export_overleaf.py
  python scripts/export_overleaf.py --output exports/my_paper_package.zip
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import zipfile
from pathlib import Path


def get_referenced_figures(paper_path: Path) -> set[str]:
    """Extract figure filenames referenced in paper.tex via \\includegraphics."""
    if not paper_path.is_file():
        return set()

    content = paper_path.read_text(encoding="utf-8", errors="ignore")
    # Match \includegraphics[...]{path} or \includegraphics{path}
    pattern = r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}"
    matches = re.findall(pattern, content)

    referenced = set()
    for m in matches:
        clean = m.strip().replace("\\", "/")
        # Strip leading figures/ if present for matching
        if clean.startswith("figures/"):
            clean = clean[len("figures/"):]
        referenced.add(clean)
    return referenced


def package_overleaf(output_zip: Path, project_root: Path) -> None:
    paper_path = project_root / "paper" / "paper.tex"
    figures_dir = project_root / "figures"
    output_dir = output_zip.parent

    if not paper_path.is_file():
        print(f"Error: LaTeX manuscript not found at {paper_path}", file=sys.stderr)
        sys.exit(1)

    if not figures_dir.is_dir():
        print(f"Error: Figures directory not found at {figures_dir}", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    referenced = get_referenced_figures(paper_path)
    print(f"[*] Found {len(referenced)} figure reference(s) in paper.tex")

    files_packed: list[tuple[Path, str]] = []

    # 1. paper.tex at zip root
    files_packed.append((paper_path, "paper.tex"))

    # 2. Package all files from figures/
    for root, _, files in os.walk(figures_dir):
        for f in sorted(files):
            file_path = Path(root) / f
            rel_to_figures = file_path.relative_to(figures_dir)
            arcname = f"figures/{rel_to_figures.as_posix()}"
            files_packed.append((file_path, arcname))

    # 3. Create zip file
    print(f"[*] Writing Overleaf package to: {output_zip}")
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path, arcname in files_packed:
            zf.write(file_path, arcname=arcname)
            print(f"  + {arcname} ({file_path.stat().st_size:,} bytes)")

    size_mb = output_zip.stat().st_size / (1024 * 1024)
    print("\n" + "=" * 65)
    print(f" SUCCESS: Overleaf package created successfully!")
    print(f" Location : {output_zip}")
    print(f" Size     : {size_mb:.2f} MB ({output_zip.stat().st_size:,} bytes)")
    print(f" Total    : {len(files_packed)} files packaged")
    print("=" * 65)
    print("Next steps:")
    print("  1. Go to https://www.overleaf.com/project")
    print("  2. Click 'New Project' -> 'Upload Project'")
    print(f"  3. Select '{output_zip.name}'")
    print("=" * 65 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Overleaf upload zip package.")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="exports/overleaf_package.zip",
        help="Path for output zip file (default: exports/overleaf_package.zip)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    output_zip = Path(args.output)
    if not output_zip.is_absolute():
        output_zip = project_root / output_zip

    package_overleaf(output_zip, project_root)


if __name__ == "__main__":
    main()
