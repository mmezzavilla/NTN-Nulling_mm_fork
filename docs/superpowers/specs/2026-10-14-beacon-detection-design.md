# STEP 1 design: gNB → VSAT beacon detection

Date: 2026-10-14. Status: draft for approval.

## Intent

The thesis authenticates the TN–NTN nulling handshake. The gNB broadcasts a
beacon that carries a random nonce (the challenge). Each VSAT that hears it
replies with a ZKP-authenticated response (STEP 2). The gNB then nulls only
toward authenticated VSATs.

A VSAT that does not hear the beacon, or cannot decode its nonce, never
responds, so it is never nulled. STEP 1 asks whether a realistic beacon
reaches every VSAT that needs protection, for gNB TX power in {30, 40, 45} dBm.

This flips the detection problem of the baseline paper (Jia et al.,
GLOBECOM 2025, `docs/Joint_Detection_...pdf`). There the VSAT transmits and the
gNB detects with 64 antennas. Here the gNB transmits and a single-antenna VSAT
detects.

## Success criterion (decided)

The beacon succeeds when **every VSAT that the gNB's data transmission could
harm detects the beacon and decodes the nonce correctly.** VSATs that can
never be harmed do not need to hear it.

"Harmed" uses the **worst case, summed over all three sectors** (decided):
every sector's full data beam pointed straight at the VSAT at once,

    INR_worst = Σ_s P · ‖h_s‖² / N0_vsat   ≥  INR_th = −6 dB

where `h_s` (64×1) is the ray-traced channel from sector `s` to the VSAT, `P` is
the per-sector TX power, and `N0_vsat = kT · B · NF_vsat`. Summing matches the
baseline notebook, which adds interference over sectors. Using the worst-case
beam is deliberately conservative: it does not depend on which TN-UE happens
to be served.

"Success" means **detection AND correct nonce decoding**, not just detection.

## Why beacon length is the key lever (design rationale)

The beacon cannot be beamformed toward VSATs, because their positions are
unknown. It is sent quasi-omni from the full 64-element array, so a VSAT
receives about the per-element gain, `|wᴴh|² ≈ ‖h‖²/N_tx`. A data beam can
deliver up to `N_tx` (18 dB) more. Beam sweeping does not help: K beams of
duration T/K deliver the same total energy per VSAT.

If the beacon and data use the same power `P`, both the beacon SNR and the
worst-case INR scale with `P`. Therefore:

- **Power** changes how many VSATs are at risk (and hear the beacon).
- **Beacon length T** alone decides whether missed-but-harmed VSATs exist.
  Red dots vanish when

      T · B  ≥  γ_req · N_tx / INR_th

  where `γ_req` is the total received beacon energy-to-noise ratio needed for
  success (Es,tot/N0).
  - Detection only (scalar, P_FA = 10⁻⁸): γ ≈ ln(10⁸) ≈ 18.4 → T ≳ 47 µs.
  - Nonce decoding (128 bits at ~3 dB Eb/N0, rough guess): γ ≈ 256 →
    T ≳ 650 µs at 100 MHz. **Decoding dominates.** The exact `γ_req` comes from
    Layer A below.

Because harm is summed over three sectors, the bound needs up to 3× (4.8 dB)
more in the worst case. `T_rule` is set from the simulated maps, not only from
this formula.

The paper's 20 µs preamble falls short of this rule, so red dots are expected
at 20 µs.

## What gets built

Two layers, both decided.

### Layer A: beacon link-level curves (Sionna, done once)

An OFDM beacon at 10 GHz, 100 MHz, made of:
- a known preamble for detection (scalar energy / matched-filter test,
  threshold set by P_FA = 10⁻⁸ per hypothesis, as in the paper);
- a 128-bit nonce, encoded with a low-rate Sionna channel code, with
  repetition if needed.

Beacon structure (*my call*): `T` is the **total** beacon length, preamble
plus nonce. The split is fixed: a preamble that takes a fixed fraction of the
resources, and a nonce codeword repeated to fill the rest. That way the total
received energy maps to a single `γ_req` at any T. The "20 µs" case means the
whole beacon fits in the paper's 20 µs preamble budget.

One false-alarm hypothesis (*my call*) is one timing offset in a continuous
search, so P_FA = 10⁻⁸ = 1/B per sample offset. That matches `p_fa = 1/B` in
`Lambda_CDF_det.ipynb`, i.e. about one false alarm per second of search.

The beacon is simulated over AWGN (single-antenna receiver) and gives:
- P_det vs SNR at the target P_FA;
- the measured P_FA under noise only, which must match 10⁻⁸ within Monte
  Carlo error (use a scaled-down P_FA check if 10⁻⁸ is too costly to measure
  directly, and state it);
- nonce block error rate vs SNR;
- `γ_req`: the received energy-to-noise ratio at which
  P_det · (1 − BLER) ≥ 0.99 (*my call*).

### Layer B: per-VSAT maps (analytic over ray-traced channels)

- Reuse the root `SceneConfigSionna` with `Denver_Scene/Boulder/Boulder.xml`:
  1 BS, 3 sectors, 4×16 URA per sector, VSAT dish pattern pointed at the
  satellite, paper deployment (80% rooftop / 20% ground, satellite
  azimuth U[0°,360°], elevation U[25°,90°], 500 km). The channels `h_s` per
  VSAT and sector come from `a_ntn` summed over paths, as in
  `Lambda_CDF_det.ipynb`. No new ray tracing method is needed.
- Beacon precoder (*my call*): a fixed, constant-modulus, quasi-omni
  precoder that uses all 64 PAs at full power, e.g. a 2D Zadoff-Chu phase
  profile (Kronecker of length-4 and length-16 ZC), ‖w‖ = 1. Its real
  pattern ripple is captured automatically because the received SNR uses the
  exact `|wᴴh_s|²`.
- Sectors (*my call*): each sector sends its own beacon on orthogonal
  resources (time-multiplexed). Each sector's beacon is tested separately;
  the VSAT succeeds if any one decodes.
- For each VSAT, sector, P ∈ {30, 40, 45} dBm and T ∈ {20 µs, T_rule}:
  - beacon SNR: `γ_s = P · T · |wᴴh_s|² / (kT · NF_vsat)`;
  - success_s = `γ_s ≥ γ_req`;
  - harmed = `INR_worst ≥ −6 dB` (summed over sectors, see above).
- Colour per VSAT:
  - **green**: decoded at least one sector's beacon (decided: one is enough,
    because all sectors belong to the same co-located gNB; whether the reply
    reaches the harming sector is a STEP 2 question);
  - **red**: harmed and decoded no sector's beacon;
  - **grey**: missed and harmless.
- `T_rule` is the smallest T meeting the rule with Layer A's `γ_req`.

### Outputs

- **Headline figure:** a 3×2 grid of maps (power × beacon length) for one
  drop, green/red/grey dots on the scene footprint with the BS marked.
- Aggregate over drops: fraction green / red / grey vs power for each T.
- Layer A curves (P_det, BLER vs SNR).

### Sanity checks (required, from CLAUDE.md)

1. The green fraction does not drop as power rises.
2. The measured P_FA matches the target.
3. A single LoS link agrees with a hand link budget (FSPL + element gain +
   VSAT sidelobe gain + noise).
4. The red fraction at `T_rule` is ~0 at every power level. A non-zero red
   fraction means the precoder ripple exceeds the `‖h‖²/N_tx` assumption,
   and should be reported, not hidden.

## Parameters

From the paper (Table I) unless noted:
- fc = 10 GHz (code uses 9.999 GHz), B = 100 MHz, kT = −174 dBm/Hz.
- P per sector ∈ {30, 40, 45} dBm, the same for beacon and data.
- NF_vsat = 2 dB (*my call*: Table I says 2 dB, but `Lambda_CDF_det.ipynb`
  uses 3 dB; the two look swapped with the BS value. Record both in
  `params.json`.)
- P_FA = 10⁻⁸ per hypothesis, INR_th = −6 dB.
- Nonce = 128 bits (*my call*).
- Paper's preamble is 20 µs (Table I; the text near Fig. 6 says 20 ms; the
  code uses 20 µs).

## Goals / non-goals / anti-goals

- **Goals:** green/red/grey maps at the three power levels; showing that beacon
  length, not power, eliminates missed-but-harmed VSATs; a value for
  `T_rule` with a decoded nonce.
- **Non-goals (later steps):** the VSAT → gNB proof response (STEP 2),
  capacity vs proof size (STEP 3), other bands (sub-7 GHz, mmWave), adversary
  / replay simulation, synchronisation and CFO search beyond what the P_FA
  budget accounts for.
- **Anti-goals:** a map that looks good because "harmed" was defined loosely;
  calling a VSAT green on detection alone; a figure that depends on the
  random TN-UE choice.

## Constraints (from CLAUDE.md)

- Experiments are `.py` scripts in `experiments/`; shared logic goes in a
  module. Do not edit the original notebooks.
- Each run saves to `results/<YYYY-MM-DD>_<name>/`: `params.json` (with git
  hash), `.npz` arrays, figures as `.pdf` + `.png`.
- Smoke run first (few VSATs, one power level). Ask before anything longer
  than ~10 minutes.
- Use the **root** modules (not `NTN-NULLING-SYS-BP/`). The patched
  `radio_map.py` is not needed for this.

## Left to the builder

Module and file names; the exact Sionna code (LDPC vs polar, rate, repetition)
as long as it carries 128 bits; OFDM numerology; the number of drops and
VSATs per drop (paper: 300 NTN-UEs, 100 drops); plotting style; whether to add
the optional **elevation-fan precoder** (a wide azimuth beam with the 4 rows
focused near the horizon) as a comparison variant.
