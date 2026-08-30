"""One visual system for the figures in `notebooks/basis_and_coef_comp.ipynb`,
`notebooks/tecator_application.ipynb` and `notebooks/overnight_jumps.ipynb`.

Colour carries exactly one meaning wherever the three estimators appear together: **which
method**. Nothing else is coloured. In `basis_and_coef_comp` the basis is a row facet and the
coefficient function a column facet, both labelled in text, so a reader who learns "orange is
method B" on the first figure is never asked to unlearn it on the fifth -- and does not have
to relearn it in the two empirical notebooks either.

The three methods under comparison take the first three slots of a categorical palette
validated for colour-vision deficiency at all pairs (worst pair Delta E 9.2 deutan, 24.0
normal, on the light surface). The untruncated FPCA pilot is a control rather than a fourth
competitor, so it is deliberately recessive grey, and the truth is ink, never a hue.
`SEQUENTIAL` is a single-hue light-to-dark blue ramp, the only correct encoding for a magnitude
such as "how often was this point left in the support".

Where a figure is not comparing methods -- a panel of price curves, a spectrum coloured by its
response -- nothing takes a method hue. `ordinal` supplies the ramp for a set that is ordered,
and `CATEGORICAL` the hues for one that is not.

`clip_to` is the other half of readability here. Method A's failures reach 10^6 on a scale where
the truth is order 1, so a panel drawn honestly is a solid block of ink and a panel drawn with
`ylim` alone is a solid block of vertical strokes. Masking the excursion and labelling its size
keeps both the shape and the magnitude legible.
"""

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

__all__ = [
    "METHOD_COLOUR", "CATEGORICAL", "INK", "MUTED", "GRID", "BAND", "SURFACE",
    "SEQUENTIAL", "SEQUENTIAL_STEPS", "ordinal", "use", "clip_to", "fmt",
]

# Slots 1-3 of the validated categorical palette, in its fixed order.
CATEGORICAL = ("#2a78d6", "#eb6834", "#1baf7a")

METHOD_COLOUR = {
    "HH-A": CATEGORICAL[0],     # blue
    "HH-B": CATEGORICAL[1],     # orange
    "OGL": CATEGORICAL[2],      # aqua
    "FPCA": "#8f8e88",          # grey: the no-sparsity control, not a fourth hue
}

INK = "#0b0b0b"        # the truth, and any reference the reader must not mistake for a series
MUTED = "#52514e"      # secondary text
GRID = "#e6e5e1"       # hairline grid, one shade off the surface
BAND = "#d9d8d2"       # the true null set, shaded behind everything
SURFACE = "#ffffff"

# Single-hue blue ramp, light to dark, for magnitude.
SEQUENTIAL_STEPS = ("#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
                    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
                    "#0d366b")
SEQUENTIAL = LinearSegmentedColormap.from_list("bench_blue", list(SEQUENTIAL_STEPS))


def ordinal(n, lightest=3):
    """`n` steps of the sequential ramp, for a set that is ordered rather than nominal.
    Returns a list of hex strings, dark increasing with the index.

    The leading basis functions of a family are such a set: they run from smoothest to
    roughest, so a categorical hue per function would throw that ordering away and, past three,
    would start cycling. The lightest step is held at ramp step 250, which is the lightest that
    still clears 2:1 against a light surface.

    PARAMETERS
    lightest : int, index into `SEQUENTIAL_STEPS` of the lightest step allowed.
    """
    n = int(n)
    if n < 1:
        raise ValueError(f"need at least one step, got {n}.")
    if n == 1:
        return [SEQUENTIAL_STEPS[-1]]
    idx = np.linspace(lightest, len(SEQUENTIAL_STEPS) - 1, n)
    return [SEQUENTIAL_STEPS[int(round(i))] for i in idx]


def use(plt):
    """Set the notebook's shared matplotlib defaults: thin marks, a hairline recessive grid,
    no top or right spine, and the categorical order as the property cycle.

    Mutates global rcParams, so it is called once from the notebook rather than on import.

    PARAMETERS
    plt : the `matplotlib.pyplot` module, passed in so this module imports no pyplot of its own.
    """
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 120,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.prop_cycle": plt.cycler(color=list(CATEGORICAL)),
        "axes.edgecolor": "#b9b8b2",
        "axes.linewidth": 0.7,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.labelcolor": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "lines.linewidth": 1.4,
        "font.size": 9,
    })


def clip_to(y, lo, hi):
    """Blank the parts of `y` that fall outside [`lo`, `hi`], and report how far out they went.

    Returns a copy of `y` with the out-of-range points set to NaN, so a line plot draws the
    in-range shape and simply stops where the curve leaves the axes instead of painting a
    vertical stroke to the boundary; and the largest absolute value among the points removed,
    or None where nothing was removed, which is the number to put in the label.

    PARAMETERS
    y  : (N,) array, the curve to draw;
    lo : float, bottom of the visible range;
    hi : float, top of the visible range.
    """
    y = np.asarray(y, dtype=float)
    outside = (y < lo) | (y > hi)
    shown = np.where(outside, np.nan, y)
    worst = float(np.max(np.abs(y[outside]))) if outside.any() else None
    return shown, worst


def fmt(x, digits=3):
    """A number for a plot label: fixed point where it is readable that way, scientific where
    it is not. Returns a str.

    PARAMETERS
    digits : int, significant digits for the fixed-point form.
    """
    x = float(x)
    if x == 0.0:
        return "0"
    if 1e-3 <= abs(x) < 1e4:
        return f"{x:.{digits}g}"
    mantissa, exponent = f"{x:.0e}".split("e")
    return f"{mantissa}e{int(exponent)}"
