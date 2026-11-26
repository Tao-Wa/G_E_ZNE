"""Zero-noise extrapolation ansatz functions.

Each function maps a noise coordinate ``x`` (a fold factor, or an error/purity
derived coordinate) plus fitted parameters onto an expectation value.  They are
consumed by ``tools.zne_loocv``, which fits them with ``scipy.optimize.curve_fit``
and extrapolates to ``x0``.

WARNING for future edits
------------------------
* The *names* and *parameter order* are referenced by string in
  ``zne.py`` / ``stable.py`` / ``zne_grover_shots.py`` (``"poly_exp"`` etc.) and
  by position in the ``p0`` / ``bounds`` lists.  Renaming or reordering the
  arguments silently changes every extrapolated value.
* The commented-out variants are kept from the original file: they document the
  model-selection process behind the paper and some are still referenced in
  earlier results.
"""

import numpy as np


# --- Extrapolation functions -------------------------------------------------
# Original alternative that was replaced by poly_exp_extrapolation; kept for
# provenance.
# def linear_extrapolation(x, a, b, c, d, f):
#     e = 2 * d
#     # return a + b * x
#     # return (a + b * (x)) * np.exp(c * x + d * (x ** 2))
#     # return a * np.exp(c * x + d * (x ** 2)) + b * x
#     return (a + (b + f * x) * (np.sqrt(np.exp(e * (x ** 2)) - 1))) * np.exp(c * x + d * (x ** 2))
#     # return b * np.exp(c * x + d * (x ** 2)) + a


def poly_exp_extrapolation(x, a, b, c, d, e):
    """Hybrid Gaussian-exponential ansatz (7-parameter family, 5 fitted)."""
    f = 2 * b
    return (a + (d + e * x) * (np.sqrt(np.exp(f * (x ** 2)) - 1))) * np.exp(c * x + b * (x ** 2))


def poly_exp_extrapolation1(x, a, b, c, d, e):
    """As above, with a constant offset ``e`` and a linear (not affine) bracket."""
    f = 2 * b
    return (a + d * (np.sqrt(np.exp(f * (x ** 2)) - 1))) * np.exp(c * x + b * (x ** 2)) + e


def poly_exp_extrapolation2(x, a, b, c, d, e):
    """Two-branch ansatz: Gaussian envelope plus a pure-exponential term."""
    return a * np.exp(c * x + b * (x ** 2)) + (d + e * x) * np.exp(c * x)


def poly_exp_extrapolation3(x, a, b, c, d, e):
    """Ansatz used for Grover circuits: affine prefactor, Gaussian envelope, offset."""
    return (a + e * x) * np.exp(c * x + b * (x ** 2)) + d


def poly_exp_extrapolation4(x, a, b, c, e):
    """Four-parameter variant of :func:`poly_exp_extrapolation3` (no offset)."""
    return (a + e * x) * np.exp(c * x + b * (x ** 2))


def linear_extrapolation1(x, a, b, c, d):
    """Quadratic-prefactor times Gaussian envelope."""
    # return a + b * x
    # return (a + b * (x)) * np.exp(c * x + d * (x ** 2))
    # return a * np.exp(c * x + d * (x ** 2)) + b * x
    return (a + b * (x ** 2)) * np.exp(c * x + d * (x ** 2))
    # return b * np.exp(c * x + d * (x ** 2)) + a


def exp_extrapolation(x, a, b, c):
    """Single exponential ``a * b**x + c`` (3 parameters)."""
    return a * (b ** x) + c


def exp_extrapolation1(x, a, b, c, d, e):
    """Double exponential ``a * b**x + c * d**x + e`` (5 parameters)."""
    return a * (b ** x) + c * (d ** x) + e


def fid_extrapolation(x, a, b):
    """Linear fit of the expectation value against the error coordinate."""
    return a + b * x


def purity_sqrt_extrapolation(x, a, b, c):
    """Purity-based ansatz; ``b`` is bounded above by ``min(purity_mean)``."""
    # return a * np.sqrt((x - b) / (1 - b))
    # b = 0.0625
    return a * np.exp(1 / 2 * np.log((x - b) / (1 - b))) + c


# Original alternative (power law) kept for provenance.
# def purity_sqrt_extrapolation(x, a, b, c):
#     # p = a * noise_value^(b) + c
#     return a * (np.abs(x) ** b) + c
