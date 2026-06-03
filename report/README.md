# Technical Report (LaTeX / Overleaf)

**Author:** Saed Abdalgani

This folder contains the academic-style technical report for the **Malicious PDF
Detector** project. It expands on the top-level `README.md` with a full
abstract, related work, methodology, experimental setup, results, discussion,
limitations, and conclusion.

## Files

| File | Purpose |
|------|---------|
| `main.tex` | The complete report. **Compiler: pdfLaTeX.** Includes a **Theoretical foundations** section (scaling, SMOTE, MLP/loss/optimization, tree ensembles, INT8 quantization, SHAP, metrics). Uses an embedded bibliography, so it builds in a single pass (no BibTeX step required). |
| `references.bib` | Optional BibTeX sources, in case you prefer a BibTeX workflow (see the header comment inside the file). |

## Build on Overleaf

1. Go to [Overleaf](https://www.overleaf.com/) → **New Project** → **Upload Project**.
2. Upload `main.tex` (and `references.bib` if you want it). Tip: zip this
   `report/` folder and upload the zip.
3. In **Menu → Settings**, set:
   - **Compiler:** `pdfLaTeX`
   - **Main document:** `main.tex`
4. Click **Recompile**. The report builds as-is — no figures or external assets
   are required (diagrams are drawn with TikZ; results are typeset as tables).

## Build locally

Requires a TeX distribution (TeX Live / MiKTeX) with `latexmk`:

```bash
cd report
latexmk -pdf main.tex
# or, without latexmk:
pdflatex main.tex
pdflatex main.tex   # second pass resolves \cref and the table of contents
```

The output is `main.pdf`.

## Switching to a BibTeX workflow (optional)

`main.tex` ships with an embedded `thebibliography` block for maximum
portability. To use `references.bib` instead, replace that block with:

```latex
\bibliographystyle{plain}
\bibliography{references}
```

then compile with: `pdflatex → bibtex → pdflatex → pdflatex`.

## Adding figures (optional)

To embed generated figures (e.g. ROC curves, confusion matrices, SHAP
importance), first regenerate them from the pipeline:

```bash
python -m src.run_all                 # writes reports/figures/*.png
python -m src.features.explain        # writes reports/figures/shap_global_importance.png
```

Then copy the desired PNGs into `report/figures/` and include them in `main.tex`
with, e.g.:

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=0.8\linewidth]{figures/roc_curves.png}
  \caption{ROC curves for the model zoo.}
\end{figure}
```
