# Publication Manuscript (`paper/`)

This directory houses the primary LaTeX manuscript for the paper:
> **"A Scalable Parallel Hybrid Quantum-Classical Convolutional Architecture Using Parameterized Quantum Circuits for Robust Image Classification"**

## Contents
- **[`paper.tex`](paper.tex)**: The canonical, single-source-of-truth LaTeX document formatted for IEEE Transactions on Quantum Engineering.
- **[`PAPER_WRITE.md`](PAPER_WRITE.md)**: Master manuscript tracker detailing section outlines, figure and table registries, theorem summaries, and drafting progress.

## Compilation

### Local Compilation
The document includes `\graphicspath{{../}{./}{../figures/}{figures/}}` in its preamble, allowing it to seamlessly resolve figures from the central `../figures/` directory:

```bash
cd paper
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```

### Overleaf Export
To create a standalone, self-contained zip file ready for Overleaf:

```bash
# From project root:
python scripts/export_overleaf.py
```

This packages `paper.tex` and all referenced figures from `figures/` directly into `exports/overleaf_package.zip`. You can then drag and drop the zip file directly into Overleaf.
