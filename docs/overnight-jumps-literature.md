# What is known about the overnight jump, and what of it this repository can test

A reading of the literature on the close-to-open return, organised by the mechanism each
strand proposes rather than by date, and written to answer one question: **which observable,
measurable before the opening bell, is correlated with the sign of the jump.** Every claim is
followed by whether the intraday panels under `data/` can carry it, and by what happened when
they were asked to. `notebooks/jump_covariates.ipynb` is the measurement;
`src/fts_research/jump/covariates.py` is the design matrix, one column per claim.

The short answer, and it is the organising fact of the whole literature: **the sign of the
overnight move is almost unforecastable from the market's own history, and everything that
does forecast it comes from outside that history.** The size is a different matter and is
forecastable from almost anything. `notebooks/overnight_jumps.ipynb` already found this in
this repository's own data -- a Ljung-Box test rejects independence of the signed jump for
none of nine instruments and of its absolute value for all nine -- and the literature says the
same thing from many directions at once.

---

## 1. The stylised facts the sign has to be explained against

**Overnight and intraday returns are different animals.** Almost the whole of the equity
premium accrues while the market is shut. French and Roll (1986) established that variance per
hour is far higher during trading than during non-trading hours, so the overnight window is
long, quiet, and carries most of the drift. Cliff, Cooper and Gulen (2008) put the point
bluntly: essentially all of the US equity premium is earned overnight, and the intraday return
is on average negative. Berkman, Koch, Tuttle and Zhang (2012) sharpen it into a cross-section
-- positive overnight returns systematically reverse during the day -- and locate the effect in
stocks that have just drawn retail attention, that are hard to value and costly to arbitrage,
and in periods of high retail sentiment. Their conclusion is a cost, not a signal: buying at
the open is expensive in a way the quoted spread does not show.

- Cliff, M. T., Cooper, M. J. and Gulen, H. (2008). *Return differences between trading and
  non-trading hours: like night and day.* Working paper.
  [SSRN 1004081](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1004081)
- French, K. R. and Roll, R. (1986). Stock return variances: the arrival of information and the
  reaction of traders. *Journal of Financial Economics* 17(1), 5-26.
- Berkman, H., Koch, P. D., Tuttle, L. and Zhang, Y. J. (2012). Paying attention: overnight
  returns and the hidden cost of buying at the open. *Journal of Financial and Quantitative
  Analysis* 47(4), 715-741.
  [cambridge.org](https://www.cambridge.org/core/journals/journal-of-financial-and-quantitative-analysis/article/abs/paying-attention-overnight-returns-and-the-hidden-cost-of-buying-at-the-open/F9AAD159B512C651F09D5D52011D88E0)

**The two components have opposite risk prices.** Hendershott, Livdan and Rösch (2020) find
returns positively related to beta overnight and negatively related to it during the day, in
the United States and internationally, across beta-sorted, industry and book-to-market
portfolios. The CAPM fails over twenty-four hours because it is two offsetting regressions
stacked on top of each other. Lou, Polk and Skouras (2019) find firm-level continuation *within*
each period and reversal *across* them, persisting when the signal is lagged by up to sixty
months, and trace it to a tug of war between institutional traders who transact during the day
and retail flow that arrives at the open. Bogousslavsky (2021) shows the anomalies themselves
accrue on radically different intraday schedules, with the size and illiquidity premia realised
in the last thirty minutes of trading.

- Hendershott, T., Livdan, D. and Rösch, D. (2020). Asset pricing: a tale of night and day.
  *Journal of Financial Economics* 138(3), 635-662.
  [sciencedirect.com](https://www.sciencedirect.com/science/article/abs/pii/S0304405X20301732)
- Lou, D., Polk, C. and Skouras, S. (2019). A tug of war: overnight versus intraday expected
  returns. *Journal of Financial Economics* 134(1), 192-213.
  [sciencedirect.com](https://www.sciencedirect.com/science/article/abs/pii/S0304405X19300650)
- Bogousslavsky, V. (2021). The cross-section of intraday and overnight returns. *Journal of
  Financial Economics* 141(1), 172-194.
  [sciencedirect.com](https://www.sciencedirect.com/science/article/abs/pii/S0304405X21000854)
- Bogousslavsky, V. and Muravyev, D. (2023). Market return around the clock: a puzzle.
  *Journal of Financial and Quantitative Analysis*.
  [cambridge.org](https://www.cambridge.org/core/journals/journal-of-financial-and-quantitative-analysis/article/abs/market-return-around-the-clock-a-puzzle/089E33AC0B4D3B9A02CBA31EDF6505B3)

**The session has its own internal momentum.** Gao, Han, Li and Zhou (2018) find the first
half-hour return of the S&P 500 ETF predicting the last half-hour return, with a predictive
R-squared of 1.6% rising to 2.6% when the twelfth half-hour is added, stronger on volatile
days, high-volume days, recession days and macroeconomic release days, and present in ten
other actively traded domestic and international ETFs. That is a statement about the *shape* of
a session rather than about its total, and it is the reason `jump.covariates` records the
opening and closing fractions of a session separately from the whole of it: if a session's
beginning predicts its end, its beginning and its end are not interchangeable as predictors of
what follows the close either.

- Gao, L., Han, Y., Li, S. Z. and Zhou, G. (2018). Market intraday momentum. *Journal of
  Financial Economics* 129(2), 394-414.
  [sciencedirect.com](https://www.sciencedirect.com/science/article/abs/pii/S0304405X18301351)

**What this buys the sign question:** the cross-period reversal is the one stylised fact that
is directly a *directional* prediction about the next overnight return, and it is what makes
the previous session's return, `ret`, a candidate rather than a control.

---

## 2. Inventory and the closing imbalance: the strongest mechanical story

Boyarchenko, Larsen and Whelan (2023) is the sharpest result in the literature and the one this
project should take most seriously. Almost the whole US equity premium accrues in a one-hour
window between 02:00 and 03:00 New York time -- the European open -- and that return is
strongly related to the **order imbalance at the close of the preceding US session**, measured
over 15:15 to 16:15. The story is inventory risk: intermediaries absorb the closing imbalance,
and the overnight return is what compensates them. The reaction is asymmetric, with selloffs
generating robust positive overnight reversals and rallies generating much weaker ones.

This is a directional prediction with a sign attached, from a quantity observable before the
close. Its companion result matters as much: Boyarchenko, Larsen and Whelan (2026) report that
the drift **has vanished since 2021**, averaging close to zero over 2021-2025 against roughly
3.7% a year before, and attribute it to the compression of end-of-day imbalance dispersion --
a standard deviation falling from 6.5% to 2.9%. Volatility and overnight liquidity were
unchanged; the imbalance was the channel and the channel closed.

- Boyarchenko, N., Larsen, L. C. and Whelan, P. (2023). The overnight drift. *Review of
  Financial Studies* 36(9), 3502-3547.
  [academic.oup.com](https://academic.oup.com/rfs/article-abstract/36/9/3502/7076616)
- Boyarchenko, N., Larsen, L. C. and Whelan, P. (2026). *The disappearing overnight drift.*
  Liberty Street Economics, Federal Reserve Bank of New York, 1 July 2026.
  [libertystreeteconomics.newyorkfed.org](https://libertystreeteconomics.newyorkfed.org/2026/07/the-disappearing-overnight-drift/)

The closing auction literature says the same thing from the microstructure side. Price
deviations created in the closing auction revert almost fully overnight, one third to one half
of the reversion happening within the first half-hour after the close where after-hours
liquidity permits; 21% of the closing auction return reverses overnight on the NYSE and 43% on
Nasdaq. Systematic reversals follow the publication of imbalance information, which is what
temporary price pressure from liquidity trading looks like.

- Bogousslavsky, V. and Muravyev, D. (2023). Who trades at the close? Implications for price
  discovery and liquidity. *Journal of Financial Markets* 66.
  [sciencedirect.com](https://www.sciencedirect.com/science/article/abs/pii/S1386418123000502)
- Wu, Y. and Jegadeesh, N. (2022). Closing auctions: Nasdaq versus NYSE. *Journal of Financial
  Economics* 143(3), 1120-1139.
  [sciencedirect.com](https://www.sciencedirect.com/science/article/abs/pii/S0304405X21005092)

**What this buys, and what it does not.** Signed order imbalance is not in a bar panel. The
best proxies a five-minute or hourly panel carries are the return over the closing fraction of
the session (`late`) and where the close sat within the session's range (`close_loc`), and
neither is the imbalance -- a closing return of a given size can come from a large imbalance
absorbed cheaply or a small one absorbed expensively. Both are in the design and are tested;
the null result reported below is a null result about *these proxies*, and given the 2026
finding that the channel itself has closed over exactly the window `data/intraday_1h.csv`
covers, it is the result the literature now predicts.

---

## 3. Information arriving from markets that are open while this one is shut

This is the strand that works, and it is the oldest. Hamao, Masulis and Ng (1990) and Engle, Ito
and Lin (1990) established transmission across non-overlapping sessions -- the "meteor shower",
volatility and returns propagating from a closing market to one opening hours later, as against
the "heat wave" of a market's own persistence. Everything since is a refinement of where the
information is and how quickly it travels.

The modern versions are directional and specific. Bao, Chen and Ma (2025) show the **US last
half-hour return predicts the next day's first half-hour return in international markets**, in
and out of sample, and that a strategy on it survives transaction costs, working best where
spreads are low or information uncertainty high. Running the clock the other way -- which is
what this project needs, since the market being forecast is American -- the European session
trades through the New York night and is the natural predictor of the New York open.

- Hamao, Y., Masulis, R. W. and Ng, V. (1990). Correlations in price changes and volatility
  across international stock markets. *Review of Financial Studies* 3(2), 281-307.
- Engle, R. F., Ito, T. and Lin, W.-L. (1990). Meteor showers or heat waves? Heteroskedastic
  intra-daily volatility in the foreign exchange market. *Econometrica* 58, 525-542.
- Xu, D., Li, B., Singh, T., Chen, X. and Li, J. (2025). Cross-market overnight time-series
  momentum. *Journal of International Financial Markets, Institutions and Money*.
  [sciencedirect.com](https://www.sciencedirect.com/science/article/abs/pii/S1042443125001295)
- Rapach, D. E., Strauss, J. K. and Zhou, G. (2013). International stock return predictability:
  what is the role of the United States? *Journal of Finance* 68(4), 1633-1662.

### 3.1 The look-ahead that inflates the machine-learning results

The applied literature reports overnight *direction* accuracies far above anything in the
finance literature, and the gap is a timing artefact rather than a modelling advance. Gao,
Zhang, Zhang, Zhao and Wang (2022) forecast the close-to-open direction of nine indices with a
multi-branch convolutional network selected by a genetic algorithm, and report accuracies of
0.6923 for the S&P 500 against 0.5160 using domestic data alone -- a stated gain of eighteen
points from adding international indices. The paper is explicit that its third input block is
"the current returns of stock markets that open before the target market on the same trading
day". For an American target that block includes European markets, which open at 07:00 UTC
**and close at 19:00 UTC** -- four and a half hours after the 14:30 UTC New York open. A
same-day European return therefore contains most of the New York session it is being used to
forecast the start of.

The correction is recent and explicit. Bai and Li (2026) impose a common-clock ordering, build
predictors only from markets that have already closed, and use a last-known-value rule where a
market is on holiday, precisely so that same-date variables cannot embed an implicit
look-ahead; their point is that "overnight" is not simultaneous and that the incremental
predictive content has to be separated from purely contemporaneous dependence. Their table also
shows the diagnostic that gives the game away in the 2022 paper: its FTSE row has a 99.945%
"down" rate and an imbalance ratio of 1820, which is a stale opening print, not a market.

- Gao, R., Zhang, X., Zhang, H., Zhao, Q. and Wang, Y. (2022). Forecasting the overnight return
  direction of stock market index combining global market indices: a multiple-branch deep
  learning approach. *Expert Systems with Applications* 194, 116506.
  [doi:10.1016/j.eswa.2022.116506](https://doi.org/10.1016/j.eswa.2022.116506)
- *When "overnight" is not simultaneous: turning cross index overnight returns into feasible
  forecasts of morning gaps* (2026). The Journal of Finance and Data Science. Cited by title:
  no public listing consulted named its authors.
  [sciencedirect.com](https://www.sciencedirect.com/science/article/pii/S2214845026000918)

This is why `jump.covariates.foreign_covariates` takes a wall-clock cutoff rather than a
session, and why `notebooks/overnight_jumps.ipynb` truncates the London session at 13:00 local
-- before the New York open under either daylight-saving offset.

---

## 4. Volatility, jumps and the sign of past variation

The size of the overnight move is forecastable and the literature on it is settled. Realised
volatility, the Parkinson (1980) range and the overnight gap itself all persist. Two results go
further and bear on the *sign*.

Barndorff-Nielsen, Kinnebrock and Shephard (2010) split realised variance into upside and
downside semivariances. Patton and Sheppard (2015) show that future volatility responds much
more strongly to the volatility of past *negative* returns than of positive ones, and that the
impact of a jump on future volatility depends on the jump's sign: negative jumps raise future
volatility and positive ones lower it. Their signed jump variation, the difference of the two
semivariances, is therefore a summary of a session that a plain return and a plain realised
volatility both throw away -- two sessions can end flat with identical realised volatility and
opposite signed variation.

- Barndorff-Nielsen, O. E., Kinnebrock, S. and Shephard, N. (2010). Measuring downside risk:
  realised semivariance. In T. Bollerslev, J. Russell and M. Watson (eds.), *Volatility and Time
  Series Econometrics*, 117-136. Oxford University Press.
- Patton, A. J. and Sheppard, K. (2015). Good volatility, bad volatility: signed jumps and the
  persistence of volatility. *Review of Economics and Statistics* 97(3), 683-697.
  [direct.mit.edu](https://direct.mit.edu/rest/article/97/3/683/58249)
- Andersen, T. G., Bollerslev, T. and Diebold, F. X. (2007). Roughing it up: including jump
  components in the measurement, modeling, and forecasting of return volatility. *Review of
  Economics and Statistics* 89(4), 701-720.
- Barndorff-Nielsen, O. E. and Shephard, N. (2006). Econometrics of testing for jumps in
  financial economics using bipower variation. *Journal of Financial Econometrics* 4(1), 1-30.
- Parkinson, M. (1980). The extreme value method for estimating the variance of the rate of
  return. *Journal of Business* 53(1), 61-65.

---

## 5. Scheduled news, and why it is a calendar rather than a signal

Lucca and Moench (2015) find US equity returns of 49 basis points on average in the twenty-four
hours before a scheduled FOMC announcement, more than 80% of the equity premium over their
sample, with no counterpart in Treasuries or money-market futures and none before other
macroeconomic releases. Since it is a *scheduled* event the drift is a directional forecast of
particular overnight windows, and one that a calendar supplies for free. It has also faded: the
pre-FOMC drift essentially disappeared after 2015, in announcements with and without a press
conference.

Overnight news more broadly is a weaker instrument than it looks. Glasserman, Krstovski,
Laliberte and Mamaysky (2024) model overnight news topics and find them more informative about
the *magnitude* of the overnight move than about its sign, which is the same asymmetry
everything else in this document reports.

- Lucca, D. O. and Moench, E. (2015). The pre-FOMC announcement drift. *Journal of Finance*
  70(1), 329-371. [doi:10.1111/jofi.12196](https://doi.org/10.1111/jofi.12196)
- Kurov, A., Wolfe, M. H. and Gilbert, T. (2021). The disappearing pre-FOMC announcement drift.
  *Finance Research Letters* 40, 101781.
  [sciencedirect.com](https://www.sciencedirect.com/science/article/abs/pii/S1544612320315956)
- Glasserman, P., Krstovski, K., Laliberte, P. and Mamaysky, H. (2024). *Does overnight news
  explain overnight returns?* arXiv:2507.04481.
  [arxiv.org/abs/2507.04481](https://arxiv.org/abs/2507.04481)
- Barclay, M. J. and Hendershott, T. (2003). Price discovery and trading after hours. *Review of
  Financial Studies* 16(4), 1041-1073.

Neither the FOMC calendar nor a news feed is in `data/`, so both are outside what this
repository can currently test. They are the obvious extension and are recorded as such in
`docs/research-log.md`.

---

## 6. What the panels here actually say

`notebooks/jump_covariates.ipynb` runs every column of `jump.covariates.SOURCE` against both
the signed jump and its size, for eight US instruments over 477 nights of the committed hourly
panel, with Newey-West errors and a Benjamini-Hochberg correction across the whole table of 208
tests. The results are in that notebook; the shape of them is this.

**The size is forecastable by almost everything, as the literature says.** Realised volatility
and the Parkinson range survive the correction for all eight instruments, the number of
calendar days the market was shut for six, the London session's realised volatility for five.

**The sign survives for exactly two things, and neither is the market's own return.** The
truncated London session return survives for five of eight instruments, positively -- news
propagating between markets, which is Hamao, Masulis and Ng's transmission. Signed jump
variation survives for five of eight, *negatively*: a session whose variation was mostly upside
is followed by a lower overnight jump. The previous session's return itself survives for none,
its closing fraction for none, its opening fraction for none, and its own lagged jump for none.

That last contrast is the finding worth having. The reversal is not of the session's *return*;
it is of the sign composition of the session's *variation*, which is a quantity that a return
cannot see and that realised volatility deliberately discards. A session that ends flat after
rising hard and falling back is followed by a different overnight move than one that ends flat
after falling hard and recovering, and the two have identical `ret` and identical `rvol`.

**Truncating the foreign session improves it, and not only for feasibility.** The London
session cut at 13:00 local correlates 0.272 with the SPY jump against 0.147 for the whole
session, and its date-blocked out-of-sample R-squared is 0.055 against -0.008. The London
afternoon overlaps the New York session, so it responds to the move that follows the open
rather than predicting the open, and adding it dilutes a real signal with a variance that
carries none. Truncation is doing the work of an estimator here, not the work of a compliance
rule -- which is the whole argument for `truncFLR` and `fdaSP`, arriving from the data.

**The closing-pressure channel does not appear.** Neither the closing return nor the close's
position in the session's range predicts the sign for any instrument. Two readings are
consistent with the literature and both should be said: an hourly bar's return is a poor proxy
for a signed order imbalance, and the panel covers 2024 to 2026, over which Boyarchenko, Larsen
and Whelan (2026) report the imbalance channel itself as having closed.

---

## 7. Where the functional model enters

The two surviving directional signals are both *summaries of a session*, and both raise the
question this repository's estimators exist to answer: which part of the session.

`notebooks/overnight_jumps.ipynb` established the negative half. Where a curve's informative
summary is one of its values -- the endpoint, say, for running realised volatility -- the
functional term is the wrong slot, because a bounded coefficient function integrated against a
curve cannot evaluate that curve at a point, and the unpenalised functional block only adds
variance. That is why `jump.covariates` exists at all.

The positive half is the foreign curve. "The London session cut at 13:00" is a truncated
functional regression with the truncation point chosen by hand, and the right way to choose it
is not by hand. `notebooks/region_selection.ipynb` puts the whole London return-rate curve in
as the regressor and lets `region` decide which part of it is informative, under an error bound
rather than under a convention -- and the New York opening bell sits at 0.81 of the London
session, so there is a known answer for the method to be scored against. That is the test the
next chapter turns on.

- Gabrys, R., Horváth, L. and Kokoszka, P. (2010). Tests for error correlation in the
  functional linear model. *Journal of the American Statistical Association* 105(491),
  1113-1125. The paper the cumulative intraday return transform comes from.
- Kokoszka, P. and Reimherr, M. (2013). Predictability of shapes of intraday price curves.
  *Econometrics Journal* 16(3), 285-308.
  [academic.oup.com](https://academic.oup.com/ectj/article-abstract/16/3/285/5061235)
- Hall, P. and Hooker, G. (2016). Truncated linear models for functional data. *Journal of the
  Royal Statistical Society: Series B* 78(3), 637-653.
  [doi:10.1111/rssb.12125](https://doi.org/10.1111/rssb.12125)
- Stefanucci, M., Bernardi, M. and Canale, A. (2025). Locally sparse estimation for functional
  linear models with scalar response. In G. Aneiros et al. (eds.), *New Trends in Functional
  Statistics and Related Fields*, ch. 62, 521-527. Springer.
  [doi:10.1007/978-3-031-92383-8_62](https://doi.org/10.1007/978-3-031-92383-8_62)

---

## 8. What is not testable here, and would be worth getting

In rough order of how much each would buy.

1. **Signed order imbalance at the close**, from TAQ or from an exchange imbalance feed. This is
   the one variable the strongest mechanism in the literature is stated in, and every proxy in
   `jump.covariates` is a proxy for it. Boyarchenko, Larsen and Whelan (2026) also make it the
   variable that decides whether the mechanism is still alive.
2. **A scheduled-announcement calendar.** FOMC dates are public and would test Lucca and
   Moench's drift directly on this panel, and would let every other coefficient be estimated
   with announcement nights held out.
3. **Index futures traded overnight.** The ES contract prices the American index continuously
   through the night, which turns the jump from an unobserved gap into an observed path and
   changes the question from forecasting to filtering.
4. **After-hours and pre-market volume.** Barclay and Hendershott (2003) show price discovery
   after hours is real but thin; the Yahoo endpoint the panels come from serves regular-session
   bars only.
5. **A longer panel.** The hourly panel is 730 days by construction of the endpoint, and the
   drift literature's most interesting current claim is a structural break in 2021 that a
   two-year window cannot see either side of.
