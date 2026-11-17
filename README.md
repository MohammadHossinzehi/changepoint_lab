# changepoint_lab

Changepoint detection from first principles in NumPy. Given a time series, find the moments where its statistical behaviour changes: a server's latency regime shifts after a deploy, a sensor drifts, a count process jumps after a policy change. This library implements the exact offline algorithms (Optimal Partitioning, PELT, Segment Neighbourhood, CROPS), the classic greedy one (Binary Segmentation) and the streaming Bayesian one (Adams and MacKay's BOCPD), and tests every exact method against brute force enumeration.

The only dependency is NumPy.

## What is in the box

**Segment costs** (`costs.py`), each evaluated in O(1) from prefix sums and vectorised over many candidate segments at once:

* `l2`: change in mean, Gaussian noise. The noise scale is estimated robustly from first differences (MAD), so mean jumps do not inflate it.
* `normal`: change in mean and variance, with a variance floor that keeps constant segments finite.
* `poisson`: change in rate of a count process.

All three return twice the negative maximised log likelihood, so one penalty scale (BIC, AIC, Hannan Quinn or a number) means the same thing for every cost. Multivariate signals of shape `(n, d)` are supported for all of them.

**Offline search** (`offline.py`):

* `optimal_partitioning`: the O(n²) penalised dynamic program, kept simple as the reference.
* `pelt`: same optimum, with pruning that makes it close to linear when changes keep arriving.
* `segment_neighbourhood`: exact best segmentation for every number of changepoints k = 0..K, no penalty needed.
* `crops`: every distinct optimal segmentation for all penalties in an interval, using about one PELT run per segmentation size.
* `binary_segmentation`: greedy top down splitting with a heap, O(n log n), not exact.
* `brute_force`: enumeration of all admissible segmentations, used as the test oracle.

**Online detection** (`online.py`): `BOCPD` maintains the exact posterior over the current run length one observation at a time, with Normal Gamma (unknown mean and variance) and Poisson Gamma models, constant or custom hazard functions, and probability pruning or a hard cap on hypotheses. It reports each changepoint together with the step at which it was declared, so you can measure detection delay.

**Evaluation** (`metrics.py`): maximum cardinality F1 within a margin, Hausdorff distance, Rand index in O(n) from a contingency table, and annotation error.

## Quick start

```bash
git clone https://github.com/MohammadHossinzehi/changepoint_lab.git
cd changepoint_lab
python -m pip install numpy          # the only requirement
python -m unittest discover -s tests -t .
python -m changepoint_lab demo
```

The demo builds three synthetic signals (mean shifts, mean plus variance shifts, Poisson rate shifts), draws a terminal strip plot of each with the true changepoints marked, and scores every method:

```
== mean signal, n=400, true changepoints [54, 110, 318]
|#@%@@@#%@#+=++*==*+* ::.:.. :.:-.:.... . ...::.:::... ...:+-==+==-==+===|
          ^         ^                                     ^
  optimal_partitioning   f1=1.00 hausdorff=3      rand=0.986 time=   15.5ms  [54, 109, 321]
  pelt                   f1=1.00 hausdorff=3      rand=0.986 time=    9.4ms  [54, 109, 321]
  binary_segmentation    f1=1.00 hausdorff=3      rand=0.986 time=    0.6ms  [54, 109, 321]
  bocpd                  f1=1.00 hausdorff=3      rand=0.986 time=   80.9ms  [54, 109, 321]
```

### On your own data

Any CSV or whitespace separated file works; header lines are skipped and `--column` picks the column.

```bash
python -m changepoint_lab detect latency.csv --column 1 --cost normal --penalty bic
python -m changepoint_lab online latency.csv --column 1 --hazard 200
python -m changepoint_lab crops  latency.csv --column 1 --pen-min 2 --pen-max 200
```

`detect` prints the changepoints, per segment means and spreads, and a strip plot. `online` streams the file through BOCPD and prints each changepoint the moment it is declared, with its delay. `crops` prints a table of every optimal segmentation in the penalty range, which is the honest answer to "how many changepoints are there?" when you do not trust any single penalty.

### As a library

```python
import numpy as np
from changepoint_lab import pelt, crops, bocpd, BOCPD, NormalGamma, score, piecewise

y, true = piecewise(n=1000, k=6, kind="meanvar", seed=1)

seg = pelt(y, cost="normal", penalty="bic")
print(seg.changepoints, seg.segments)
print(score(true, seg.changepoints, len(y), margin=5))

for lo, hi, s in crops(y, 2, 300, cost="normal").solutions:
    print(f"penalty {lo:.1f} to {hi:.1f}: {len(s.changepoints)} changepoints")

det = BOCPD(NormalGamma.from_data(y[:50]), hazard=150)
for t, x in enumerate(y):
    info = det.update(x)
    if info["declared"] is not None:
        print(f"step {t}: change at {info['declared']}")
```

Changepoints are always the index where a new segment starts, so the segments are `y[0:c1]`, `y[c1:c2]`, and so on.

## Benchmarks

`python examples/benchmark.py` on a laptop class CPU. Signals have one change per 100 samples on average:

```
n       OP evals     PELT evals  ratio   OP time  PELT time  same answer
1000       500500         65941      8x     0.03s      0.03s  True
2000      2001000        162299     12x     0.07s      0.05s  True
4000      8002000        334713     24x     0.24s      0.09s  True
8000     32004000        637984     50x     0.82s      0.23s  True
16000     skipped       1260255                        0.37s
```

Optimal Partitioning's work grows quadratically while PELT's stays at roughly 80 cost evaluations per sample, and the two always agree. (Wall clock understates the gap at small n because OP's inner loop is one big vectorised NumPy call.)

Average accuracy over 30 random signals per row (n = 600, five changes, deliberately small jumps):

```
kind     method                 F1(margin 5)  rand index  mean ms
mean     pelt                          0.970      0.9932     14.6
mean     binary_segmentation           0.873      0.9804      0.8
mean     bocpd (online)                0.906      0.9896     88.9
meanvar  pelt                          0.896      0.9870     17.3
meanvar  binary_segmentation           0.827      0.9757      0.8
meanvar  bocpd (online)                0.872      0.9858     88.5
poisson  pelt                          0.902      0.9555     15.4
poisson  binary_segmentation           0.863      0.9502      0.6
poisson  bocpd (online)                0.890      0.9707     63.7
```

The exact offline search wins on F1, binary segmentation is twenty times faster and noticeably less precise, and BOCPD gets close to the offline methods while only ever looking at the past.

## Design notes

**PELT and the minimum segment length.** PELT drops candidate s at time t when F(s) + C(s, t) > F(t), because every cost here satisfies C(s, T) ≥ C(s, t) + C(t, T) (splitting never hurts a maximum likelihood fit). That argument silently assumes the segment [t, T) is allowed. With a minimum segment length m it is not allowed for T < t + m, and in that window the pruned s can still be the best start. Many implementations drop s immediately, which is exact only for m = 1. Here a pruned candidate is marked and stays eligible until T = t + m. I wrote the eager version too while testing: it returns a worse objective than Optimal Partitioning on 2 of 180 random signals with m in {2, 4, 7}, and the regression test in `tests/test_offline.py` exercises exactly that setup.

**Floored variance done properly.** The mean and variance cost needs a floor so that a constant segment does not have log(0) cost. The tempting shortcut cost = len·(log 2πv + 1) is only correct at the unfloored MLE; once v is floored the correct expression is len·log 2πv + SSE/v. Because the Gaussian profile likelihood is unimodal in v, max(sample variance, floor) is the exact constrained optimum, so subadditivity still holds and PELT stays exact. A test checks subadditivity on 400 random splits per cost, including a constant stretch.

**CROPS instead of guessing a penalty.** The optimal penalised cost as a function of the penalty β is the lower envelope of lines Q_k + βk. CROPS only runs PELT where two known lines cross, so it discovers every size on the envelope with at most about two runs per size; the tests confirm that for 59 penalties on a grid every PELT answer is one CROPS already found, and that each CROPS solution equals the Segment Neighbourhood optimum for its size.

**BOCPD indexing.** Papers disagree by one about what r_t means. Here r_t is the number of observations in the current segment after seeing t of them, and the recursion is checked against brute force enumeration of all 2^(t−1) segmentations under the same product partition prior, for both conjugate models, to 1e‑10. A side effect of this convention is worth knowing: in the "r_t = 0 is a fresh empty run" convention, P(r_t = 0 | data) under a constant hazard is identically equal to the hazard and is useless for detection, whereas here P(r_t = 1 | data) is close to 1 right after a jump (also tested). Detection itself tracks the MAP segment start and declares it after it persists for a few steps, which keeps one noisy sample from producing a false alarm.

**Pruning BOCPD.** The exact posterior has t hypotheses after t steps. Dropping run lengths below 1e‑6 probability (or keeping the top 100) leaves the detected changepoints unchanged on a 1500 sample test signal while cutting memory by more than half; with clearer segment structure the saving is far larger.

**Metrics that do not flatter.** Matching detections to truth by "closest pair first" is a common bug: with truth [0, 3], detections [2, 5] and margin 2 it matches 3 to 2 and reports one hit instead of two. Since every margin window has the same width, a left to right sweep that assigns each true change the leftmost free detection in its window is optimal; it is tested against brute force matching on 300 random cases.

## Testing

```bash
python -m unittest discover -s tests -t .
```

36 tests, about five seconds. The interesting ones are oracles: brute force enumeration for Optimal Partitioning, PELT and Segment Neighbourhood across all three costs and minimum sizes 1 to 3; brute force run length posteriors and evidence for BOCPD; closed form segment evidence versus the product of sequential predictives; prefix sum costs versus direct formulas; pairwise Rand index versus the contingency table version. The rest cover recovery of known changes, no false alarms on stationary noise, multivariate input, pruning efficiency, CROPS coverage and the command line interface.

## Layout

```
changepoint_lab/
  costs.py       segment costs with prefix sums
  penalties.py   BIC, AIC, Hannan Quinn
  offline.py     OP, PELT, segment neighbourhood, binary segmentation, CROPS, brute force
  online.py      BOCPD with Normal Gamma and Poisson Gamma models
  metrics.py     F1 with margin, Hausdorff, Rand index
  synth.py       piecewise stationary signal generator
  cli.py         detect, online, crops, demo
tests/           unittest suite with brute force oracles
examples/        benchmark.py
```

## References

* R. Killick, P. Fearnhead, I. A. Eckley (2012). Optimal detection of changepoints with a linear computational cost. JASA.
* K. Haynes, I. A. Eckley, P. Fearnhead (2017). Computationally efficient changepoint detection for a range of penalties. JCGS.
* R. P. Adams, D. J. C. MacKay (2007). Bayesian online changepoint detection. arXiv:0710.3742.
* C. Truong, L. Oudre, N. Vayatis (2020). Selective review of offline change point detection methods. Signal Processing.

MIT licensed.
