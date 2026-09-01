# figures

Figures exported for writing up. Nothing here is committed except this file.

Every figure is drawn by a notebook or a script and written out by
`fts_research.core.style.save_figure`, as a PDF for the document and a PNG for anything that
cannot take one:

```python
from fts_research.core import style

style.save_figure(fig, "basis_target_grid")
```

So a figure in the thesis is reproduced by re-running the cell that drew it, rather than
recovered from whatever was last screenshotted out of a notebook. The visual system the
figures share — one hue per method, and nothing else coloured — is in
`src/fts_research/core/style.py`.
