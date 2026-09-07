"""Error control for a selection made by subsampling, after Shah and Samworth (2013).

Stability selection (Meinshausen and Buhlmann, 2010) runs a selection procedure on many
subsamples and keeps whatever is chosen often enough. Its point is not the aggregation but
the bound that comes with it: the expected number of things chosen that the base procedure
rarely chooses is controlled by the threshold, and by nothing the statistician has to assume
about the model.

Complementary pairs stability selection draws the subsamples as disjoint halves
`(A_{2j-1}, A_{2j})`, `j = 1, ..., n_pairs`, and records

    Pi(k) = (1 / 2 n_pairs) sum_j 1{k selected on A_j},

an unbiased estimate of the selection probability `p_k` at sample size `floor(n / 2)`. Write
`L_theta = {k : p_k <= theta}` for the indices the base procedure rarely selects and
`H_theta` for its complement. Shah and Samworth's Theorem 1 is two-sided, and this module
implements both halves, because a locally sparse estimate makes two claims and not one:

    (i)  tau > 1/2   E|S_CPSS(tau) ∩ L_theta| <= theta / (2 tau - 1) * E|S ∩ L_theta|,
    (ii) tau < 1/2   E|N_CPSS(tau) ∩ H_theta| <= (1 - theta) / (1 - 2 tau) * E|N ∩ H_theta|,

with `N` the complement of `S`. The first bounds what is wrongly declared informative, the
second what is wrongly declared null. Neither needs the exchangeability of the noise
variables or the not-worse-than-random-guessing condition Meinshausen and Buhlmann assume,
and both hold for every `n_pairs`, down to one pair.

The first bound is loose, because it reaches `S` through `1 + Pi_tilde >= 2 Pi` and then
applies Markov's inequality, whose extremal distribution is nothing like the one the
proportions actually follow. Sections 3.2 and 3.3 of the paper tighten it under shape
restrictions on that distribution, and their sharpest form is equation (8),

    E|S_CPSS(tau) ∩ L_theta| <= min{D(theta^2, 2 tau - 1, n_pairs, -1/2),
                                    D(theta, tau, 2 n_pairs, -1/4)} |L_theta|,

where `D(eta, t, n_pairs, r)` is the largest tail probability `P(X >= t)` attainable by an
`r`-concave distribution on `{0, 1/n_pairs, ..., 1}` with mean at most `eta`. `D` has no
closed form; `r_concave_tail` computes it by the algorithm of the paper's Section A.4, which
parametrises the extremal mass functions by their support and one scale.

`|L_theta|` and `E|S ∩ L_theta|` are unknown, and Section 3.4 says to replace them by the
number of candidates and by `q = E|S|`. Those substitutions are arguments here rather than
assumptions inside, so a caller can see which of the two quantities each bound scales with.

References
Full entries for the two papers named above are in `docs/references.md`, this repository's one
bibliography.
"""

import math

import numpy as np
from scipy import optimize

__all__ = [
    "markov_multiplier",
    "unimodal_multiplier",
    "r_concave_tail",
    "false_selection_bound",
    "missed_exclusion_bound",
    "achievable_thresholds",
    "threshold_for_inclusion",
    "threshold_for_exclusion",
]


def markov_multiplier(theta, tau):
    """Theorem 1(i)'s factor `theta / (2 tau - 1)`, which multiplies `E|S ∩ L_theta|`.

    Returns the factor, or infinity at `tau <= 1/2`, where the theorem says nothing.

    PARAMETERS
    theta : float in [0, 1], the selection probability below which an index counts as one the
            base procedure rarely selects;
    tau   : float, the proportion of subsamples an index must be selected on.
    """
    theta, tau = float(theta), float(tau)
    if tau <= 0.5:
        return np.inf
    return theta / (2.0 * tau - 1.0)


def unimodal_multiplier(theta, tau, n_pairs):
    """Theorem 2's factor `C(tau, n_pairs) theta`, valid where the proportions are unimodal.

    The paper gives `C` only for `theta <= 3 ** -0.5`, which is the range it expects to be
    used, and on the lattice of thresholds its statement names. Returns the factor, or
    infinity outside either range.

    PARAMETERS
    theta   : float in [0, 1], as in `markov_multiplier`;
    tau     : float, the selection threshold;
    n_pairs : int, complementary pairs of subsamples.
    """
    theta, tau, n_pairs = float(theta), float(tau), int(n_pairs)
    if theta > 3.0 ** -0.5 or tau <= 0.5:
        return np.inf
    if tau <= 0.75:
        if tau <= min(0.5 + theta ** 2, 0.5 + 0.5 / n_pairs + 0.75 * theta ** 2):
            return np.inf
        factor = 1.0 / (2.0 * (2.0 * tau - 1.0 - 0.5 / n_pairs))
    else:
        factor = 4.0 * (1.0 - tau + 0.5 / n_pairs) / (1.0 + 1.0 / n_pairs)
    return factor * theta


def _support_scale(mean_index, k, power):
    """The `a` for which the mass function proportional to `(a + i) ** power` on
    `i = 0, ..., k` has mean `mean_index`. Returns a float, or None where no such `a` exists.

    `power = 1 / r < 0`, so the weights decrease in `i`: as `a` falls the mass piles onto
    `i = 0` and the mean falls to zero, and as `a` grows the weights flatten and the mean
    rises to `k / 2`. The mean is strictly increasing in `a` between those limits, so the
    solution is unique when `0 < mean_index < k / 2` and there is none otherwise.
    """
    if not 0.0 < mean_index < 0.5 * k:
        return None
    index = np.arange(k + 1, dtype=float)

    def gap(log_a):
        weight = (math.exp(log_a) + index) ** power
        return float(index @ weight) / float(weight.sum()) - mean_index

    lo, hi = -60.0, 60.0
    if gap(lo) > 0.0 or gap(hi) < 0.0:                  # the root has left the bracket
        return None
    return math.exp(optimize.brentq(gap, lo, hi, xtol=1e-13, rtol=1e-14))


def _extremal_tail(a, k, power, mean_index, cut):
    """Tail probability `P(X >= cut / n_pairs)` of the paper's `g_{a,k}`, its equation (17).

    `g_{a,k}` is the mass function whose `r`th power is linear on `0, ..., k` and which
    carries a final atom at `k + 1` sized to hold the mean at `mean_index / n_pairs`. Returns
    the tail, clipped to [0, 1].
    """
    index = np.arange(k + 1, dtype=float)
    weight = (a + index) ** power
    denominator = float((k + 1.0 - index) @ weight)
    if denominator <= 0.0:
        return 0.0
    tail = 1.0 - (k + 1.0 - mean_index) * float(weight[:cut].sum()) / denominator
    return float(min(1.0, max(0.0, tail)))


def r_concave_tail(eta, t, n_pairs, r):
    """The paper's `D(eta, t, n_pairs, r)`: the largest `P(X >= t)` an `r`-concave
    distribution on `{0, 1/n_pairs, ..., 1}` with mean at most `eta` can have.

    Computed by the algorithm of Section A.4. Proposition 11 says the maximiser's `r`th power
    is linear up to its penultimate support point, which leaves a one-parameter family per
    support size: `g_{a,k}` on `{0, ..., k + 1}`, with `a` running over the interval on which
    the mean constraint binds. The tail is maximised over `a` within each family and then
    over `k`. Returns the bound, and 1 for `t <= eta`, where Markov's inequality is already
    vacuous, or for `t <= 0`, the convention equation (8) is stated under.

    That algorithm is stated for a decreasing maximiser, which Section A.4 says holds when
    `t > 2 eta` and leaves the other case aside as "less important for our application". It
    is not unimportant here: the groups of a functional penalty are few and often selected,
    so `eta = q / n_groups` is routinely above a half and the condition routinely fails.
    Outside it this returns Markov's `eta / t`, which holds for every distribution and so
    keeps the bound a bound; inside it the two are taken together, since `D` cannot exceed
    Markov either. A silent zero there would make `false_selection_bound` too small rather
    than too large, which is the one direction an error control may not fail in.

    PARAMETERS
    eta     : float, the largest mean the distribution may have;
    t       : float, the point the tail is taken from;
    n_pairs : int, the lattice `{0, 1/n_pairs, ..., 1}` the distribution lives on;
    r       : float below zero, the concavity index; the paper uses -1/2 and -1/4.
    """
    eta, t, n_pairs, r = float(eta), float(t), int(n_pairs), float(r)
    if r >= 0.0:
        raise ValueError(f"r must be negative, got {r}.")
    if t <= eta:
        return 1.0
    if t > 1.0:
        return 0.0

    markov = min(1.0, eta / t)
    if t <= 2.0 * eta:                                  # the maximiser need not be decreasing
        return float(markov)

    power = 1.0 / r
    mean_index = n_pairs * eta
    cut = int(math.ceil(n_pairs * t - 1e-9))            # least index with i / n_pairs >= t
    if cut > n_pairs:
        return 0.0

    scale = {k: _support_scale(mean_index, k, power) for k in range(cut, n_pairs + 1)}
    best = 0.0
    for k in range(cut, n_pairs):
        upper = scale.get(k)
        if upper is None:
            continue
        lower = scale.get(k + 1)
        if lower is None or lower >= upper:
            lower = 1e-12 * upper

        def negative_tail(a, k=k):
            return -_extremal_tail(a, k, power, mean_index, cut)

        found = optimize.minimize_scalar(negative_tail, bounds=(lower, upper),
                                         method="bounded",
                                         options={"xatol": 1e-12 * max(1.0, upper)})
        best = max(best, -float(found.fun),
                   _extremal_tail(lower, k, power, mean_index, cut),
                   _extremal_tail(upper, k, power, mean_index, cut))

    if scale.get(n_pairs) is not None:                  # the family filling the whole lattice
        best = max(best, _extremal_tail(scale[n_pairs], n_pairs, power, mean_index, cut))
    return float(min(markov, max(0.0, best)))


def false_selection_bound(theta, tau, n_pairs, n_low, n_selected=None, shape="r-concave"):
    """Expected number of rarely-selected indices that stability selection nonetheless keeps.

    Three bounds on one quantity, in increasing order of what they assume and decreasing
    order of size: `"worst-case"` is Theorem 1(i) and assumes nothing; `"unimodal"` is
    Theorem 2 and assumes the paired proportions are unimodal; `"r-concave"` is equation (8),
    the tightest, and assumes the paired proportions are `-1/2`-concave and the single
    proportions `-1/4`-concave. Returns the bound, which is infinite where the shape's
    theorem does not reach the given `tau`.

    Only the first two scale with `E|S ∩ L_theta|`, for which `n_selected` stands in;
    equation (8) scales with `|L_theta|`, for which `n_low` does. Section 3.4 of the paper
    replaces the first by `q` and the second by the number of candidates.

    PARAMETERS
    theta      : float in [0, 1], the selection probability an index counts as rare below;
    tau        : float, the proportion of subsamples an index must be selected on;
    n_pairs    : int, complementary pairs of subsamples;
    n_low      : int, stands in for `|L_theta|`, the number of rarely-selected indices;
    n_selected : int or None, stands in for `E|S ∩ L_theta|`; `n_low` when None;
    shape      : one of "r-concave", "unimodal" and "worst-case", which theorem to use.
    """
    n_selected = n_low if n_selected is None else n_selected
    if shape == "worst-case":
        return markov_multiplier(theta, tau) * n_selected
    if shape == "unimodal":
        return unimodal_multiplier(theta, tau, n_pairs) * n_selected
    if shape != "r-concave":
        raise ValueError(f"shape must be 'r-concave', 'unimodal' or 'worst-case', "
                         f"got {shape!r}.")
    if tau <= theta:
        return float(n_low)
    paired = r_concave_tail(theta ** 2, 2.0 * tau - 1.0, n_pairs, -0.5)
    single = r_concave_tail(theta, tau, 2 * n_pairs, -0.25)
    return float(min(paired, single) * n_low)


def missed_exclusion_bound(theta, tau, n_excluded):
    """Expected number of often-selected indices that stability selection nonetheless drops.

    Theorem 1(ii), the other half of the theorem, and the one a null region needs: declaring
    an index null when it survives fewer than `tau < 1/2` of the subsamples costs at most
    `(1 - theta) / (1 - 2 tau)` times what the base procedure itself drops. Returns the bound.

    Unlike `false_selection_bound`'s worst-case form, `E|N ∩ H_theta|` has no conservative
    stand-in worth using -- the number of candidates makes the bound vacuous at every `tau`,
    including zero, where nothing is declared null and the truth is zero. `stability.resolve`
    therefore passes the plug-in `sum_k (1 - Pi(k)) 1{Pi(k) > theta}`, which is the quantity
    itself with the unbiased `Pi` in place of `p_k` throughout. That is an estimate and not a
    bound, and it is the one term across the two halves that is.

    PARAMETERS
    theta      : float in [0, 1], the selection probability an index counts as frequent above;
    tau        : float below 1/2, the proportion of subsamples below which an index is
                 dropped;
    n_excluded : float, stands in for `E|N ∩ H_theta|`, what the base procedure itself drops.
    """
    theta, tau = float(theta), float(tau)
    if tau >= 0.5:
        return np.inf
    return (1.0 - theta) / (1.0 - 2.0 * tau) * float(n_excluded)


def achievable_thresholds(n_pairs):
    """The values `Pi` can take, `{0, 1/2B, 1/B, ..., 1}` for `B = n_pairs`. Returns an array.

    A threshold between two of them selects exactly what the lower one does, so these are the
    only thresholds worth searching over.

    PARAMETERS
    n_pairs : int, complementary pairs of subsamples.
    """
    n_pairs = int(n_pairs)
    return np.arange(2 * n_pairs + 1) / (2.0 * n_pairs)


def threshold_for_inclusion(theta, budget, n_pairs, n_low, n_selected=None,
                            shape="r-concave"):
    """Least threshold above 1/2 whose `false_selection_bound` is within `budget`.

    Returns the threshold, or NaN where no achievable threshold meets the budget.

    PARAMETERS
    theta      : float in [0, 1], as in `false_selection_bound`;
    budget     : float, the largest expected number of false inclusions allowed;
    n_pairs    : int, complementary pairs of subsamples;
    n_low      : int, stands in for `|L_theta|`;
    n_selected : int or None, stands in for `E|S ∩ L_theta|`;
    shape      : one of "r-concave", "unimodal" and "worst-case", which theorem to use.
    """
    for tau in achievable_thresholds(n_pairs):
        if tau <= 0.5:
            continue
        if false_selection_bound(theta, tau, n_pairs, n_low, n_selected, shape) <= budget:
            return float(tau)
    return float("nan")


def threshold_for_exclusion(theta, budget, n_pairs, n_excluded):
    """Greatest threshold below 1/2 whose `missed_exclusion_bound` is within `budget`.

    Returns the threshold, or NaN where none is. Zero always qualifies where `n_excluded` is
    finite -- it declares nothing null -- so a NaN here says the plug-in was not finite.

    PARAMETERS
    theta      : float in [0, 1], as in `missed_exclusion_bound`;
    budget     : float, the largest expected number of missed exclusions allowed;
    n_pairs    : int, complementary pairs of subsamples;
    n_excluded : float, stands in for `E|N ∩ H_theta|`.
    """
    best = float("nan")
    for tau in achievable_thresholds(n_pairs):
        if tau >= 0.5:
            break
        if missed_exclusion_bound(theta, tau, n_excluded) <= budget:
            best = float(tau)
    return best
