# STEP 1 Beacon Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce green/red/grey VSAT maps showing which VSATs decode the gNB's nonce beacon, for P ∈ {30, 40, 45} dBm × T ∈ {20 µs, T_rule}.

**Architecture:** A small package `beacon/` holds the pure-numpy link budget, the quasi-omni precoder, the Sionna link-level beacon simulation (Layer A), channel extraction and run I/O. Three slow/fast experiment scripts in `experiments/` use it. Ray tracing (slow) saves channels once. Map analysis (fast) runs from the saved channels.

**Tech Stack:** Python 3.12, TensorFlow, Sionna 1.x (`sionna.phy` FEC/mapping), Sionna RT 1.x (`sionna-rt`), numpy, matplotlib, pytest.

**Spec:** `docs/superpowers/specs/2026-10-14-beacon-detection-design.md`

## Global Constraints

- Experiments are `.py` scripts in `experiments/`, run from the repo root as `python -m experiments.<name>`. Shared logic lives in `beacon/`. Do not edit any existing notebook or the root modules.
- Each run writes to `results/<YYYY-MM-DD>_<name>/`: `params.json` (all parameters + git commit hash), raw arrays `.npz`, figures as `.pdf` and `.png`.
- Smoke run (few VSATs, one power level) before every full run. Time the smoke run and extrapolate; **ask the user before any run expected to exceed ~10 minutes.**
- Use the root `SceneConfigSionna` and `Denver_Scene/Boulder/Boulder.xml`, never `NTN-NULLING-SYS-BP/`.
- Constants: fc = 9.999e9 Hz, B = 100e6 Hz, kT = −174 dBm/Hz, NF_vsat = 2 dB (record `nf_vsat_notebook_db = 3` too), P_FA = 1e-8, INR_th = −6 dB, nonce = 128 bits, N_tx = 64 (4×16), 3 sectors, P ∈ {30, 40, 45} dBm per sector.
- `P` is per-sector power in W internally; dBm only at the edges (`dbm_to_w`).
- Harm is worst case summed over sectors: `INR_worst = Σ_s P‖h_s‖²/(N0_psd·B)`. Green = at least one sector's beacon decoded.
- Preamble fraction of beacon samples = 0.25 (*builder's call, recorded in params*).

## Review Focus

1. VSATs with no ray-traced path (`h = 0`) must classify grey with no NaN/inf or warnings breaking the run. Test: Task 2 `test_zero_channel_is_grey`.
2. A run with a single VSAT (smoke run) must keep shape `(1, 3, 64)`. The baseline notebook's `.squeeze()` breaks this. Test: Task 5 `test_single_rx_keeps_shape`.
3. A beacon too short to carry one nonce codeword must raise, not silently decode garbage. Test: Task 4 `test_too_short_beacon_raises`.
4. A success curve that never reaches 0.99 must raise, not extrapolate γ_req. Test: Task 4 `test_gamma_req_unreached_raises`.
5. dBm/W and PSD/bandwidth mix-ups shift every map by 30 or 80 dB. Test: Task 2 hand-value tests, plus the Task 6 LoS check.

---

### Task 1: Environment, package skeleton and run I/O

**Files:**
- Create: `beacon/__init__.py` (empty), `beacon/io.py`, `experiments/__init__.py` (empty), `tests/test_io.py`, `pytest.ini`, `requirements-beacon.txt`

**Interfaces:**
- Produces: `make_run_dir(name: str, params: dict, root: str = "results", date: str | None = None) -> pathlib.Path` creates `<root>/<date or today YYYY-MM-DD>_<name>/` and writes `params.json` = params + `"git_commit"` (from `git rev-parse HEAD`) + `"git_dirty"` (bool, from `git status --porcelain`). `save_figure(fig, run_dir: Path, stem: str) -> None` writes `<stem>.pdf` and `<stem>.png` (dpi 200).

- [ ] **Step 1: Create the venv.** `/opt/homebrew/bin/python3.12 -m venv sionna_env && source sionna_env/bin/activate && pip install --upgrade pip tensorflow tqdm numpy scipy matplotlib importlib_metadata sionna sionna-rt pytest`. Python 3.12, not the system 3.13.
- [ ] **Step 2: Verify Sionna RT.** Run `python -c "import sionna.phy, sionna.rt; from sionna.rt import load_scene; s=load_scene('Denver_Scene/Boulder/Boulder.xml'); print('ok', sionna.__version__)"`. Expected: `ok <version>`. If DrJit reports it cannot find LLVM, run `brew install llvm` and `export DRJIT_LIBLLVM_PATH=/opt/homebrew/opt/llvm/lib/libLLVM.dylib`, then retry. Record the export in `requirements-beacon.txt` as a comment.
- [ ] **Step 3: Pin versions.** Write `requirements-beacon.txt` from `pip freeze | grep -iE '^(tensorflow|sionna|sionna-rt|mitsuba|drjit|numpy|scipy|matplotlib|pytest)=='`. Write `pytest.ini` with `[pytest]`, `testpaths = tests`, `pythonpath = .`. Add `sionna_env/` to `.gitignore`.
- [ ] **Step 4: Write failing tests** in `tests/test_io.py`:

```python
def test_make_run_dir_writes_params(tmp_path):
    d = make_run_dir("smoke", {"a": 1}, root=str(tmp_path), date="2026-10-14")
    assert d.name == "2026-10-14_smoke"
    p = json.loads((d / "params.json").read_text())
    assert p["a"] == 1 and len(p["git_commit"]) == 40 and isinstance(p["git_dirty"], bool)

def test_save_figure_writes_pdf_and_png(tmp_path):
    fig = plt.figure(); save_figure(fig, tmp_path, "f")
    assert (tmp_path / "f.pdf").exists() and (tmp_path / "f.png").exists()
```

- [ ] **Step 5: Run** `pytest tests/test_io.py -v`. Expected: FAIL (ImportError).
- [ ] **Step 6: Implement** `beacon/io.py`. Use `subprocess.run(["git", ...], capture_output=True, text=True, check=True)` with `cwd` set to the repo root (the parent of `beacon/`).
- [ ] **Step 7: Run** `pytest tests/test_io.py -v`. Expected: 2 passed.
- [ ] **Step 8: Commit** `beacon/ experiments/__init__.py tests/test_io.py pytest.ini requirements-beacon.txt .gitignore` with message `Add beacon package skeleton, run I/O and pinned env`.

---

### Task 2: Link budget and classification (pure numpy)

**Files:**
- Create: `beacon/linkbudget.py`, `tests/test_linkbudget.py`

**Interfaces:**
- Produces (all arrays numpy; `h` has shape `(n_vsat, n_sect, n_tx)` complex; `w` has shape `(n_tx,)`):
  - `dbm_to_w(p_dbm: float) -> float`
  - `noise_psd_w(nf_db: float, kt_dbm_hz: float = -174.0) -> float` (W/Hz)
  - `inr_worst(h, p_w: float, b_hz: float, nf_db: float) -> np.ndarray` shape `(n_vsat,)`, linear, `Σ_s p_w·‖h_s‖² / (noise_psd·b_hz)`
  - `beacon_snr(h, w, p_w: float, t_s: float, nf_db: float) -> np.ndarray` shape `(n_vsat, n_sect)`, linear, `p_w·t_s·|h_s·w*|² / noise_psd`, i.e. `np.abs(h @ w.conj())**2`
  - `classify(gamma, inr, gamma_req: float, inr_th: float) -> np.ndarray` int8, shape `(n_vsat,)`: `GREEN=1` if any sector `gamma ≥ gamma_req`; else `RED=2` if `inr ≥ inr_th`; else `GREY=0`. Module constants `GREY, GREEN, RED`.
  - `t_rule(snr_rate, inr, gamma_req: float, inr_th: float) -> float`: `snr_rate = beacon_snr(..., t_s=1.0)`. Returns `max over harmed VSATs of gamma_req / max_s snr_rate`, or `0.0` if none harmed. A harmed VSAT with `max_s snr_rate == 0` returns `inf`.
  - `t_rule_bound(gamma_req: float, n_tx: int, n_sect: int, inr_th: float, b_hz: float) -> float` = `n_sect·gamma_req·n_tx/(inr_th·b_hz)` (the spec's analytic bound, including the 3× sector factor).

- [ ] **Step 1: Write failing tests** with these hand values:

```python
def test_noise_psd():            assert noise_psd_w(2.0) == pytest.approx(6.30957e-21, rel=1e-5)
def test_dbm_to_w():             assert dbm_to_w(30.0) == pytest.approx(1.0)
def test_inr_worst_sums_sectors():
    h = np.zeros((1, 3, 64), complex); h[0, :, 0] = 1e-6          # ‖h_s‖² = 1e-12 each
    assert inr_worst(h, 1.0, 1e8, 2.0)[0] == pytest.approx(4.7547, rel=1e-3)
def test_beacon_snr_hand_value():
    h = np.zeros((1, 3, 64), complex); h[0, 0, 0] = 1e-6
    w = np.full(64, 1/8, complex)
    assert beacon_snr(h, w, 1.0, 20e-6, 2.0)[0, 0] == pytest.approx(49.528, rel=1e-3)
def test_classify_cases():
    gamma = np.array([[10., 0, 0], [1., 1, 1], [1., 1, 1]]); inr = np.array([0.1, 1.0, 0.1])
    assert classify(gamma, inr, 5.0, 0.25).tolist() == [GREEN, RED, GREY]
def test_zero_channel_is_grey():
    h = np.zeros((2, 3, 64), complex); w = np.full(64, 1/8, complex)
    with np.errstate(all="raise"):
        c = classify(beacon_snr(h, w, 1., 1e-3, 2.), inr_worst(h, 1., 1e8, 2.), 256., 0.25)
    assert c.tolist() == [GREY, GREY]
def test_t_rule_picks_worst_harmed():
    rate = np.array([[100., 50, 0], [10., 0, 0], [1., 0, 0]]); inr = np.array([1., 1., 0.1])
    assert t_rule(rate, inr, 256., 0.25) == pytest.approx(25.6)     # VSAT 2 not harmed
def test_t_rule_bound_value():
    assert t_rule_bound(256., 64, 3, 10**-0.6, 1e8) == pytest.approx(1.9568e-3, rel=1e-3)
```

- [ ] **Step 2: Run** `pytest tests/test_linkbudget.py -v`. Expected: FAIL (ImportError).
- [ ] **Step 3: Implement** `beacon/linkbudget.py` with the signatures above.
- [ ] **Step 4: Run** `pytest tests/test_linkbudget.py -v`. Expected: 8 passed.
- [ ] **Step 5: Commit** with message `Add beacon link budget and VSAT classification`.

---

### Task 3: Quasi-omni Zadoff-Chu precoder

**Files:**
- Create: `beacon/precoder.py`, `tests/test_precoder.py`

**Interfaces:**
- Produces: `zc_precoder(n_rows: int = 4, n_cols: int = 16, root: int = 1) -> np.ndarray` shape `(n_rows·n_cols,)` complex128, `np.kron(z(n_rows), z(n_cols)) / sqrt(n_rows·n_cols)`, where `z(N)[n] = exp(-1j·π·root·n·(n + N%2)/N)` (the ZC formula for even and odd N). The index order is row-major (row index outer), matching the Kronecker product.

- [ ] **Step 1: Write failing tests:**

```python
def test_constant_modulus_unit_norm():
    w = zc_precoder(); assert np.allclose(np.abs(w), 1/8) and np.isclose(np.linalg.norm(w), 1)
def test_flat_on_dft_grid():
    w = zc_precoder()
    F = np.kron(np.fft.fft(np.eye(4)), np.fft.fft(np.eye(16))) / 8      # unit-norm 2D DFT beams
    assert np.allclose(np.abs(F.conj().T @ w)**2, 1/64)
```

- [ ] **Step 2: Run** `pytest tests/test_precoder.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** `beacon/precoder.py`.
- [ ] **Step 4: Run** `pytest tests/test_precoder.py -v`. Expected: 2 passed.
- [ ] **Step 5: Commit** with message `Add 2D Zadoff-Chu quasi-omni beacon precoder`.

---

### Task 4: Layer A, link-level beacon simulation (Sionna) and curves

**Files:**
- Create: `beacon/phy.py`, `tests/test_phy.py`, `experiments/beacon_curves.py`

**Interfaces:**
- Consumes: `beacon.io.make_run_dir`, `save_figure`.
- Produces:
  - `detection_threshold(pfa: float) -> float` = `-ln(pfa)` (normalized matched-filter statistic `|pᴴr_p|²/(N0‖p‖²)` is Exp(1) under noise only).
  - `simulate_beacon(gamma_db: np.ndarray, t_s: float, b_hz: float, n_trials: int, pfa: float = 1e-8, seed: int = 0, preamble_frac: float = 0.25) -> dict` with keys `"gamma_db"`, `"pd"`, `"bler"` (nonce block error rate given detection), `"psucc"` (fraction of trials detected AND nonce bits exactly correct). All arrays have shape `gamma_db.shape`.
  - `measure_pfa(t_s: float, b_hz: float, pfa: float, n_trials: int, seed: int = 0, preamble_frac: float = 0.25) -> float` (noise only).
  - `gamma_req_db(gamma_db: np.ndarray, psucc: np.ndarray, target: float = 0.99) -> float`: linear interpolation in dB at the first crossing of `target`; `ValueError` if never reached.
  - Saved file `results/<date>_beacon_curves/curves.npz` with `gamma_db`, `pd_<tag>`, `bler_<tag>`, `psucc_<tag>` for tags `t20us`, `t1ms`, and scalar `gamma_req_db`. `gamma_req_db` is the **max** over the two T values (conservative, *builder's call*).

**Model (decided; implement exactly):**
- `γ` is total received beacon energy over N0 for the whole beacon. `n = round(t_s·b_hz)` samples, per-sample Es = γ·N0/n with N0 = 1. Flat channel `g = e^{jφ}` with φ ~ U[0, 2π) per trial (narrowband model, as in the spec's `|wᴴh|²`).
- Preamble: the first `n_p = round(preamble_frac·n)` samples, a known unit-modulus QPSK sequence (fixed seed).
- Nonce: 128 random bits → `sionna.phy.fec.polar.Polar5GEncoder(k=128, n=512)` → QPSK (`sionna.phy.mapping.Mapper("qam", 2)`) → 256 symbols, repeated cyclically over the remaining `n − n_p` samples. Raise `ValueError` if `n − n_p < 256`.
- Receiver: detect if the statistic ≥ `detection_threshold(pfa)`. `ĝ = pᴴr_p/‖p‖²`. Coherent MRC of all repetitions of each symbol using `ĝ`. Max-log QPSK LLRs with the effective post-combining noise variance. `Polar5GDecoder(encoder, dec_type="SCL", list_size=8)`. Process trials in batches of ≤ 500.

- [ ] **Step 1: Write failing tests:**

```python
def test_threshold():                 assert detection_threshold(1e-8) == pytest.approx(18.4207, rel=1e-4)
def test_measured_pfa_matches():      # 10⁻⁸ is unmeasurable; same formula checked at 10⁻²
    assert measure_pfa(20e-6, 1e8, 1e-2, 20000, seed=1) == pytest.approx(1e-2, abs=3e-3)
def test_high_snr_always_succeeds():
    assert simulate_beacon(np.array([40.0]), 20e-6, 1e8, 200)["psucc"][0] == 1.0
def test_low_snr_fails():
    assert simulate_beacon(np.array([0.0]), 20e-6, 1e8, 200)["psucc"][0] < 0.05
def test_too_short_beacon_raises():
    with pytest.raises(ValueError): simulate_beacon(np.array([30.]), 2e-6, 1e8, 10)   # 200 samples
def test_gamma_req_interpolates():
    assert gamma_req_db(np.array([20., 21, 22]), np.array([0.5, 0.98, 1.0])) == pytest.approx(21.5)
def test_gamma_req_unreached_raises():
    with pytest.raises(ValueError): gamma_req_db(np.array([20., 21]), np.array([0.5, 0.9]))
```

- [ ] **Step 2: Run** `pytest tests/test_phy.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** `beacon/phy.py` per the model above.
- [ ] **Step 4: Run** `pytest tests/test_phy.py -v`. Expected: 7 passed.
- [ ] **Step 5: Implement** `experiments/beacon_curves.py` with `argparse`: `--smoke` (γ ∈ {15, 25, 35} dB, 200 trials, T = 20 µs only), and the default full run (γ from 10 to 35 dB step 0.5, 5000 trials, T ∈ {20 µs, 1 ms}). It saves `curves.npz`, the figure `curves` (P_det, 1−BLER and P_succ vs γ for both T, with a vertical line at γ_req), and `params.json` (all model constants). It prints `gamma_req_db`.
- [ ] **Step 6: Smoke run** `python -m experiments.beacon_curves --smoke`. Expected: the run dir is created and P_succ rises with γ. Time it, extrapolate to the full run, and ask the user if the estimate is over 10 min.
- [ ] **Step 7: Full run.** Check the result: `gamma_req_db` lies between 20 and 30 dB (the spec's rough estimate was ~24 dB), and the two T curves agree within 1 dB. Report both numbers.
- [ ] **Step 8: Commit** `beacon/phy.py tests/test_phy.py experiments/beacon_curves.py` with message `Add Layer A beacon link-level simulation and curves`.

---

### Task 5: Channel extraction and ray-traced drops

**Files:**
- Create: `beacon/channels.py`, `tests/test_channels.py`, `experiments/beacon_channels.py`

**Interfaces:**
- Produces:
  - `channels_from_cir(a: np.ndarray) -> np.ndarray`: input is a Sionna RT `cir(out_type="numpy")` coefficient array, shape `[n_rx, 1, n_tx, n_tx_ant, n_paths, n_time]`. Output is `a[:, 0, :, :, :, 0].sum(axis=-1)`, shape `(n_rx, n_tx, n_tx_ant)` complex64. Never `.squeeze()`.
  - Saved file `results/<date>_beacon_channels/channels.npz` with keys: `h` `(n_drops, n_vsat, 3, 64)` complex64, `vsat_pos` `(n_drops, n_vsat, 3)`, `bs_pos` `(n_drops, 3)`, `sat_az_deg`/`sat_el_deg` `(n_drops,)`, `extent` `(4,)`, `point_type` (2D grid of drop 0, for the map background).

- [ ] **Step 1: Write failing tests:**

```python
def test_sums_paths():
    a = np.zeros((2, 1, 3, 64, 5, 1), complex); a[1, 0, 2, 7, :, 0] = 0.1
    h = channels_from_cir(a); assert h.shape == (2, 3, 64) and np.isclose(h[1, 2, 7], 0.5)
def test_single_rx_keeps_shape():
    assert channels_from_cir(np.ones((1, 1, 3, 64, 1, 1))).shape == (1, 3, 64)
```

- [ ] **Step 2: Run** `pytest tests/test_channels.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** `beacon/channels.py`.
- [ ] **Step 4: Run** `pytest tests/test_channels.py -v`. Expected: 2 passed.
- [ ] **Step 5: Implement** `experiments/beacon_channels.py` with `argparse`: `--n-drops` (default 100), `--n-vsat` (default 300), `--seed` (default 0), `--smoke` (1 drop, 5 VSATs). Per drop, mirroring `Lambda_CDF_det.ipynb` cell 8:
  - `np.random.seed(seed + drop)`; az ~ U[0, 360), el ~ U[25, 90]
  - `SceneConfigSionna(scene, nbs=1, nsect=3, fc=9.999e9)`
  - `compute_positions(ntn_rx=n_vsat, tn_rx=0, azimuth=az, elevation=el, centerBS=True, bs_dist_min=35, bs_dist_max=3000)`
  - `compute_paths(tx_rows=4, tx_cols=16, tn_rx_rows=1, tn_rx_cols=1, max_depth=3)`
  - `h = channels_from_cir(cfg.a_ntn)`

  If `tn_rx=0` fails inside `compute_positions`, use `tn_rx=1` (TN paths are not used) and note it in `params.json`.
- [ ] **Step 6: Smoke run** `python -m experiments.beacon_channels --smoke`. Expected: `channels.npz` with `h.shape == (1, 5, 3, 64)` and at least one non-zero VSAT. Time it per drop, extrapolate to 100 drops × 300 VSATs, and **ask the user** before the full run if it exceeds 10 min.
- [ ] **Step 7: Full run** (after approval if needed).
- [ ] **Step 8: Commit** `beacon/channels.py tests/test_channels.py experiments/beacon_channels.py` with message `Add ray-traced gNB-to-VSAT channel drops`.

---

### Task 6: Single-LoS hand link budget check

**Files:**
- Create: `experiments/los_check.py`

**Interfaces:**
- Consumes: `beacon.channels.channels_from_cir`. Importing `SceneConfigSionna` registers the `"vsat"` antenna pattern.

- [ ] **Step 1: Implement** `experiments/los_check.py`:
  - Empty scene `load_scene()`, `frequency = 9.999e9`.
  - TX array 4×16, spacing 0.5, `"tr38901"`, `"V"`, at `[0, 0, 10]`, orientation `[0, -0.1745329252, 0]` (sector 0, as in `compute_paths`).
  - RX array 1×1 `"vsat"` `"V"` at `[500, 0, 10]`, `look_at([0, 0, 10])`.
  - `PathSolver` with `max_depth=0`, `los=True`, `synthetic_array=True`.
  - Compare the measured mean per-element `|h|²` in dB with the hand value `20·log10(λ/(4π·500)) + (8 − 12·(10/65)²) + 38.1 = −60.60 dB` (FSPL −106.42, element 7.72 dBi at 10° off the tilted boresight, dish 38.1 dBi at boresight). Print both. Exit with code 1 if `|Δ| > 0.5 dB`.
- [ ] **Step 2: Run** `python -m experiments.los_check`. Expected: `|Δ| ≤ 0.5 dB` and exit code 0. If it fails, stop and report. Do not tune the tolerance.
- [ ] **Step 3: Commit** with message `Add single-LoS hand link budget check`.

---

### Task 7: Maps, aggregate fractions and sanity checks

**Files:**
- Create: `experiments/beacon_maps.py`

**Interfaces:**
- Consumes: `channels.npz` (Task 5), `curves.npz` (Task 4), `beacon.linkbudget.*`, `beacon.precoder.zc_precoder`, `beacon.io.*`.
- Produces: `results/<date>_beacon_maps/` with `classes.npz` (`classes` `(n_T, n_P, n_drops, n_vsat)` int8, `t_values`, `p_dbm`, `t_rule`, `t_rule_bound`), figures `maps_grid` and `fractions`, and `params.json` (including both input run dirs).

- [ ] **Step 1: Implement** `experiments/beacon_maps.py` with `argparse` `--channels <path>` and `--curves <path>`:
  - `w = zc_precoder()`; `gamma_req = 10**(gamma_req_db/10)`; `inr_th = 10**(-0.6)`.
  - `T_rule = max over P ∈ {30, 40, 45} and all drops of t_rule(beacon_snr(h, w, P, 1.0, 2.0), inr_worst(h, P, 1e8, 2.0), gamma_req, inr_th)`, rounded up to the next 10 µs. Report it alongside `t_rule_bound(gamma_req, 64, 3, inr_th, 1e8)`.
  - Classify every (T ∈ {20e-6, T_rule}, P, drop).
  - `maps_grid`: 3 rows (P) × 2 cols (T) for drop 0. `point_type` background in light greys, VSAT dots green `#2e7d32` / red `#c62828` / grey `#9e9e9e`, BS as a black triangle, and a per-panel title with the counts of each class.
  - `fractions`: green / red / grey fraction vs P (x = 30, 40, 45 dBm), solid lines for 20 µs and dashed for T_rule, pooled over drops.
- [ ] **Step 2: Implement the sanity checks** in the same script. Print `PASS`/`FAIL` for each and exit code 1 on any FAIL:
  - (a) For each T, the green fraction is non-decreasing in P.
  - (b) At T_rule the red count is 0.
  - (c) Print `T_rule / t_rule_bound`. A ratio > 1 means the precoder ripple exceeds the `‖h‖²/N_tx` assumption. Print it as `NOTE`, not a FAIL; the spec says to report it.
- [ ] **Step 3: Smoke run** on the smoke channels and curves: `python -m experiments.beacon_maps --channels results/<smoke ch> --curves results/<smoke curves>`. Expected: both figures written and checks printed. Only one power level is meaningful here, so check that the script runs end to end.
- [ ] **Step 4: Full run** on the full channels and curves. Expected: checks (a) and (b) PASS. Report T_rule, the bound ratio, and the red fraction at 20 µs for each P.
- [ ] **Step 5: Commit** with message `Add STEP 1 beacon detection maps and sanity checks`.
