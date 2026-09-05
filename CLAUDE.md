# Project: hypothesis_testing_arena

Guidance for Claude Code working in this repo. Source of truth for scope and intent:
[.llm/PRD_hypothesis_testing_arena.md](.llm/PRD_hypothesis_testing_arena.md). This file wins on
*how* to build; the PRD wins on *what* to build. If they conflict, say so instead of silently picking.

---

## 1. Identity

A head-to-head comparison of six two-sample tests, judged by simulation against a known truth.

**Six contenders** (all written from scratch in `src/tests/`, all validated against scipy):
Student t · Welch t · Mann-Whitney U · permutation · bootstrap difference · Bayesian estimation.

**Scenario axes:** normal · right-skewed · heavy-tailed/outliers · unequal variance · small n.

**Thesis (must be *measured*, never asserted rhetorically):**
1. Student t inflates the false-positive rate above α under unequal variance.
2. Student t loses power under skew + small n.
3. Welch is the free fix for (1) and should be the default.
4. Permutation and bootstrap hold α across the scenario grid, assuming nothing — with one
   measured exception, `skewed_extreme_unequal`, where *no* mean-based test holds α. That
   exception is in the grid and asserted in `tests/`; do not quietly drop it.
5. Under the null, p-values must be Uniform(0,1). Non-uniform ⇒ the test is invalid *here*.
6. The t-test wins only where its assumptions hold — and it does win there. Say so.

**Measured corrections to the PRD.** Three of the PRD's predictions did not survive simulation.
They are documented at their implementation sites and asserted in `tests/`; do not "fix" the
code back toward the PRD:
- Student does **not** lose power to a permutation test on the same mean difference under skew
  + small n — the two are within a point of each other. The permutation test's advantage under
  skew comes from being free to test a *different statistic*. See `tests/test_power.py`.
- Cohen's d is **not** comparable across population shapes; a skewed population's sd is inflated
  by its tail, so the same d separates the bulk far more. Compare tests within an arena row,
  never power across rows of different kinds.
- Student inflates α under unequal variance only when the **sample sizes are also unequal**.
  With equal n it is close to fine, which is why the failure hides in textbook examples.

Deliverables: 6 notebooks, a `src/` library, a Streamlit app, a pytest suite, CI.
Everything is simulated. There is no dataset to download and none should be added.

---

## 2. Environment and constraints

| Thing | Value |
|---|---|
| Runtime | 2-CPU / 8 GB GitHub Codespace, **no GPU** |
| Python | 3.11+ |
| Package manager | `uv` only. **Never `pip install`**, never `conda`, never a bare `python`/`pytest` — always `uv run ...` |
| Runtime deps | scipy, numpy, pandas, matplotlib, plotly, streamlit, pydantic, pydantic-settings, pyyaml |
| Dev deps | pytest, ruff, ty, jupyter, ipykernel, nbconvert, nbclient |
| Budget | Any notebook < 5 min end to end; the full arena cell ≈ 4 min and must be cached |

**Do not add a dependency without asking.** In particular there is no PyMC, no arviz, no statsmodels,
no scikit-learn, no numba. The Bayesian contender must be implementable in numpy/scipy alone — see §6.6.

---

## 3. Commands

```bash
make setup       # first run: install uv, uv sync --all-extras, register the ipykernel
make sync        # uv sync --all-extras
make lint        # format + check + typecheck
make test        # uv run pytest
make notebooks   # execute all notebooks/0*.ipynb via nbconvert (timeout 300s each)
make run         # streamlit app on :8501
make lab         # jupyterlab on :8888
make ci          # sync + lint + test   <- what CI runs
make dev         # lint + test          <- fast inner loop
```

Fast loop while iterating on one module:
`uv run pytest tests/test_permutation.py -x -q -m "not slow"`.

**Before declaring any task done: `make lint && make test` must be green.** Report failures with the
actual output; never describe a red suite as passing.

---

## 4. Layout

```
config/settings.yaml        # effect sizes, skew, variance ratios, sample sizes, seeds, n_reps
src/config.py               # pydantic-settings loader for the above
src/tests/                  # THE SIX STATISTICAL TESTS (not pytest)
  student.py welch.py mann_whitney.py permutation.py bootstrap.py bayesian.py
src/simulation/populations.py   # normal / skewed / heavy-tailed / unequal-variance generators
src/simulation/scenarios.py     # null + effect scenarios carrying known ground truth
src/evaluation/false_positive.py  power.py  calibration.py  comparison.py
src/visualisation.py        # every figure in the project
app/streamlit_app.py        # 4 tabs (§9)
notebooks/01..06            # narrative only, no logic
tests/                      # PYTEST SUITE
outputs/figures/            # committed figures;  outputs/cache/ is git-ignored
```

### Two naming traps — read before scaffolding

1. **`src/tests/` is the six statistical tests; `tests/` is pytest.** Never let pytest collect
   `src/tests/`. `pyproject.toml` pins `testpaths = ["tests"]`; keep it, and never name a file in
   `src/tests/` `test_*.py`. When writing prose or commit messages, disambiguate explicitly
   ("the permutation *test module*" vs "the permutation *unit tests*").
2. **`src` is imported as a flat package** (`from src.tests.welch import welch_test`), while the
   project is named `hypothesis-testing-arena`. That mismatch breaks a default build backend, so at
   scaffold time either set `[tool.uv] package = false` (simplest — it is an app, not a library) or
   point the backend at the directory, e.g. `[tool.hatch.build.targets.wheel] packages = ["src"]`.
   Also add `[tool.pytest.ini_options] pythonpath = ["."]` so `tests/` can import `src`.
   `make test-cov` uses `--cov`, which needs `pytest-cov` in the dev extras — add it when you first
   wire coverage, not before.

---

## 5. Cross-cutting conventions

**Argument order and sign.** Every test takes `(a, b)` where `a` is control/group 1 and `b` is
treatment/group 2. The effect is always `b − a`. `alternative="greater"` always means "b exceeds a".
One sign convention, everywhere — sign flips are the most common bug in this codebase.

**Uniform return type.** All six tests return the same frozen dataclass from `src/tests/__init__.py`:

```python
@dataclass(frozen=True)
class TestResult:
    name: str                     # "welch", "permutation", ...
    statistic: float | None       # t, U, observed difference, ...
    p_value: float | None         # None for the Bayesian contender
    effect: float                 # the point estimate of b − a
    ci: tuple[float, float] | None
    alpha: float
    reject: bool                  # the decision the arena scores
    extra: dict[str, float] = field(default_factory=dict)   # df, prob_effect, n_perm, ...
```

`reject` is what `src/evaluation/` counts. It must be defined for every contender, including the
Bayesian one (§6.6), and the definition must be documented in that module's docstring.

**Signature shape.**
```python
def welch_test(a, b, *, alternative="two-sided", alpha=0.05, rng=None, **specific) -> TestResult
```
Keyword-only after the two samples. Resampling tests take `rng: np.random.Generator`.

**Randomness.** `np.random.default_rng(seed)` only. **Never** `np.random.seed`, never the legacy
`np.random.*` global functions. Generators are passed in explicitly, never created inside a hot loop.
For a batch of `R` replications, either draw the whole `(R, n)` block from one generator or spawn
children with `np.random.SeedSequence(seed).spawn(R)` — both are reproducible; pick one per module
and stay consistent. Every published number must be regenerable from `config/settings.yaml`.

**Vectorize; don't parallelize.** Two cores. Speed comes from numpy over the replication axis
(`(R, n)` arrays, `axis=-1` reductions), not from multiprocessing. Do not add joblib.

**Variance.** `ddof=1` everywhere. A `np.var` call with default `ddof=0` in a t-test is a bug.

**Config, not constants.** α, `n_reps`, sample sizes, skew, variance ratios and seeds live in
`config/settings.yaml` and are read via `src/config.py`. Notebooks and the app read the same config.
Hard-coded `10_000` or `0.05` scattered through `src/` is a defect.

**Style.** ruff `line-length = 99`, `E501` ignored; ruff rules include `N` (naming) and `A`
(builtin shadowing). Statistical capitals therefore need lowercase names — write `u_stat`, `t_stat`,
`p_val`, not `U`, `T`. Type-hint every public function. Docstrings are short and state **the formula
and the assumption the test makes**, since the assumptions are the subject of the project.

---

## 6. The six tests — implementation contracts

Each module: from-scratch implementation, a docstring naming its assumptions, and a unit test
asserting agreement with scipy on clean data to **1e-9**.

### 6.1 `student.py`
Pooled variance `s_p² = ((n_a−1)s_a² + (n_b−1)s_b²) / (n_a+n_b−2)`, `df = n_a+n_b−2`.
Assumes normality **and equal variances**. Validate vs `scipy.stats.ttest_ind(equal_var=True)`.
This is the cautionary protagonist — implement it correctly so its failures are real failures, not
implementation bugs.

### 6.2 `welch.py`
`se = sqrt(s_a²/n_a + s_b²/n_b)`; Welch–Satterthwaite df. Assumes normality, **not** equal variance.
Validate vs `scipy.stats.ttest_ind(equal_var=False)`.

### 6.3 `mann_whitney.py`
Average ranks via `scipy.stats.rankdata`; `u_a = r_a − n_a(n_a+1)/2`. Normal approximation with
**tie correction and continuity correction**; validate vs
`scipy.stats.mannwhitneyu(method="asymptotic", use_continuity=True)`.
**Estimand caveat is mandatory** in the docstring, the notebooks and the arena table: it tests
stochastic dominance, *not* a difference in means. Never score it as though it were estimating Δμ
without a footnote — that is the risk called out in the PRD.

### 6.4 `permutation.py`
Shuffle the pooled labels `B` times, recompute the statistic, place the observation in that null.
- p-value **must** use the add-one correction: `p = (1 + #{|t_b| ≥ |t_obs|}) / (B + 1)`. A p-value of
  exactly 0 is a bug.
- Vectorized: one `(B, n_a+n_b)` permutation of indices, reduce on `axis=1`. At `B=10_000` and
  `n≤1_000` that is < 100 MB; chunk over `B` if a scenario goes larger.
- **The default `statistic=` is `welch_t_statistic`, not the mean difference.** Permuting raw
  mean differences is exact only under full exchangeability and does *not* hold α under unequal
  variance with unequal n (measured: 0.39 → the studentized version gives 0.057). `effect_fn=`
  is a separate parameter so the reported effect stays on the scale of the data.
- Must accept an arbitrary `statistic=` callable (median, trimmed mean, ratio) — "works for any
  statistic" is one of the project's claims, and it is where its power advantage under skew
  actually comes from.
- Sanity check against `scipy.stats.permutation_test` on a small case.

### 6.5 `bootstrap.py`
Resample each group with replacement `B` times, form the difference, report the **percentile CI** by
default; `reject = 0 ∉ CI` at level `1−α`. Document that this is a CI-inversion decision, not a
p-value, and that a strict bootstrap *test* resamples under a null-shifted distribution —
`null_shift=True`, which the arena registry uses so bootstrap has a p-value and can join the
calibration comparison. That p-value is **studentized by default** (`studentize=True`) for the same
reason the permutation test is: raw null-shifted mean differences run at 0.083 against α = 0.05 on
small unbalanced samples, studentized they run at 0.047. The cost is stated, not hidden — the
studentized version is *conservative* on contaminated data (0.015). BCa intervals are an optional
extra, not the default.

### 6.6 `bayesian.py`
No PyMC in this project. Default implementation: independent Jeffreys/reference priors per group, so
the posterior for each mean is `μ | data ~ t_{n−1}(x̄, s/√n)`; draw from both, report
`P(μ_b > μ_a)` and the credible interval of `μ_b − μ_a`. Fast, exact, vectorizable.
- `reject := credible interval of the difference excludes 0` (equivalently `P(μ_b>μ_a)` outside
  `[α/2, 1−α/2]`). Optionally support a ROPE.
- `p_value=None`; `extra["prob_effect"] = P(μ_b > μ_a)`.
- Be honest in the write-up: scoring a Bayesian decision rule by its frequentist false-positive rate
  is a frequentist audit of a Bayesian procedure. Label it as such. Expect this specification to land
  close to Welch (it is the Behrens–Fisher problem) — that is an interesting result, not a bug.
- A Kruschke-style BEST with a Student-t likelihood is a documented future extension, not the default.

---

## 7. Simulation and evaluation contracts

**`simulation/populations.py`** — generators for: normal, right-skewed (lognormal), heavy-tailed
(t with small df, or contaminated normal), unequal-variance normal pairs. Each returns samples *and*
declares its true mean and true median.

**`simulation/scenarios.py`** — a `Scenario` object carrying the knobs (`kind`, `sd_ratio`, `n_a`,
`n_b`, `effect_d`) plus **known ground truth**. Two fairness rules, both load-bearing:

1. An effect is a **pure location shift** (`ShiftedPopulation`), so the treated group keeps the
   control group's shape. Raising the target mean instead silently de-skews the treated group —
   measured: a lognormal's cv drops 1.00 → 0.56 at d = 0.8, and the arena then scores each test's
   reaction to a shape change as if it were power.
2. The population mean difference is exactly `delta`, and the scenario **also records the true
   median difference**, because the mean-based tests and Mann-Whitney target different estimands.
   They coincide under a pure shift and diverge as soon as the spreads differ on a skewed
   population — which is precisely where reading Mann-Whitney as a statement about means goes
   wrong. Without both recorded the arena would convict Mann-Whitney of an error it did not make.

**`evaluation/false_positive.py`** — rejection rate over `R` reps of a null scenario. Always report
the Monte Carlo standard error `sqrt(p(1−p)/R)` alongside the rate (at `R=10_000`, `p=0.05` →
SE ≈ 0.0022). Any claim of "inflated" must clear several SEs, not one.

**`evaluation/power.py`** — the same over an effect scenario.

**`evaluation/calibration.py`** — the p-value uniformity check: PP-plot against Uniform(0,1) plus a
KS statistic. This is the visual proof in notebook 02. The Bayesian contender has no p-value; show
its posterior-probability calibration separately rather than faking a p-value for it.

**`evaluation/comparison.py`** — the arena harness. Returns a **tidy long DataFrame**, one row per
`(scenario, test)`:
`scenario, test, n_a, n_b, alpha, delta, reps, reject_rate, mc_se, metric ∈ {fpr, power}, seed`.
Wide/pivoted frames are for display only.

**Caching.** The arena writes to `outputs/cache/` (git-ignored) keyed by a hash of the resolved
config. Cached results must be invalidated when the config changes — hash the config, don't trust a
filename. Figures land in `outputs/figures/` and *are* committed.

**CI profile.** CI cannot afford 10 000 reps. Support a reduced-rep profile selected by config/env
(e.g. `HTA_PROFILE=ci` → `n_reps=1_000`) and make CI use it. Published numbers always come from the
full profile.

---

## 8. Notebooks

Six notebooks, in order: `01_the_question_and_the_reflex`, `02_when_the_t_test_breaks` (the
showpiece), `03_the_assumption_free_tests`, `04_arena`, `05_power_and_sample_size`,
`06_decision_framework`.

Rules:
- **No logic in notebooks.** They import from `src/` and narrate. If a notebook needs a loop over
  scenarios, that loop belongs in `src/evaluation/`.
- Markdown-first: every code cell is preceded by prose saying what it will show and followed by prose
  saying what it showed.
- Deterministic: seeds from config; must run top-to-bottom from a clean kernel in **< 5 min**.
- Save every figure to `outputs/figures/` with a stable filename.
- Notebook 06 ships the practical artifact: `compare_groups(a, b, test=...)`, which runs the chosen
  test **and warns when the data violates that test's assumptions**. That warning behavior needs a
  unit test.
- `make notebooks` must pass before any notebook change is considered done.

---

## 9. Streamlit app

`app/streamlit_app.py`, four tabs, all reading `src/`: (1) assumption breaker — skew and
variance-ratio sliders, watch Student's false-positive rate climb past 5 %; (2) test race — all six
verdicts side by side with disagreements highlighted; (3) null calibration — repeated runs, p-value
uniformity; (4) power explorer — effect size and n vs each test's detection rate.
Keep every interaction under a couple of seconds: use a reduced rep count for live sliders,
`@st.cache_data` on anything heavier, and reuse the same functions the notebooks call.

---

## 10. Testing requirements

`tests/` mirrors the PRD's four files and must include:

| Test | Assertion |
|---|---|
| `test_tests.py` | all six from-scratch tests match scipy on clean data to **1e-9** |
| `test_permutation.py` | permutation holds α across the grid; fails only on `skewed_extreme_unequal`; studentizing is what buys that |
| `test_false_positive.py` | Student's FPR is **materially above** α under unequal variance *with unequal n*; Welch's is at α; Student is fine at equal n |
| `test_power.py` | a **trimmed-mean** permutation test beats Student under skew at matched α; permutation *on the mean* does not; Student's apparent power edge under unequal variance is bought with a 39% FPR |

Monte-Carlo assertions must not be flaky: fix the seed, use a reduced rep count, and set the
tolerance from the binomial SE (roughly `3 × sqrt(p(1−p)/R)`), not from a number that happened to
pass once. Mark anything over ~5 s `@pytest.mark.slow` and register the marker in `pyproject.toml`.
The success criteria in PRD §8 are the acceptance tests — each row should map to an assertion.

---

## 11. Figures

All plotting lives in `src/visualisation.py`; notebooks and the app call it, they don't build axes.
Matplotlib for static figures, plotly where interactivity earns its place (arena heatmaps,
the Streamlit tabs). Every figure that makes a claim shows the **α reference line** and, where it is
a rate estimated from simulation, its Monte Carlo error. Colour-code contenders consistently across
the whole project — one test, one colour, in every figure.

---

## 12. Working style in this repo

- **Don't strawman the t-test.** Implement it correctly, show it winning on clean normal data, then
  show precisely where and why it fails. The project's credibility is the whole point.
- **Every claim gets a number, and every headline number gets an assertion in `tests/`.** If it isn't
  measured with a stated seed and rep count, it doesn't go in the README, the notebooks or the
  article.
- Report Monte Carlo uncertainty. "5.4 % vs 5.0 %" is not an effect at R=1 000.
- Prefer editing over creating; keep the file tree exactly as PRD §4.1 specifies. Don't invent extra
  modules, don't add a `utils.py` grab bag.
- Cross-link the sibling projects (`ab_testing_lab`, `bootstrap_101`, `bayesian_101`,
  `imputation_arena`) rather than duplicating their material.
- Commits: Conventional Commits. Don't commit or push unless asked. `outputs/cache/`, `.venv/`,
  `.llm/`, `.agents/`, `agent/` stay git-ignored; `outputs/figures/` is committed.

## 13. Token rules

Caveman mode is ON for this repo — code-first, minimal prose, no restating the plan back. The
`caveman*` skills are installed; `/caveman-review` for diffs, `/caveman-commit` for messages.
`/compact` at notebook boundaries. Read `config/settings.yaml` and the one module you're changing;
don't re-read the PRD every turn.
