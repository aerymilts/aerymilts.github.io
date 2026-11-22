---
title: "When Offline Metrics Mislead: Auditing Covariate Shift, Data Leakage, and Broken Feedback Loops in Operational ML"
layout: post
date: 2026-11-21 14:00
tag:
  - machine learning
  - evaluation
  - computer vision
  - mlops
category: blog
author: kennethho
hidden: false
description: "Two operational ML case studies on auditing geographic covariate shift, cross-pipeline data leakage, inference latency cliffs, and closed-world evaluation mirages."
---

In high-consequence machine learning, the most dangerous failure mode is not a model with low accuracy. It is a model with high offline accuracy that silently breaks in deployment.

Over the past year leading applied ML research and production-readiness audits in high-assurance operational environments, I repeatedly saw offline benchmarks fail for two distinct reasons:

1. **Pipeline and distribution failures**, where cross-pipeline data leakage, geographic covariate shift, and naive inference scaling mask how a model behaves on real-world inputs.
2. **Taxonomy and feedback-loop failures**, where closed-world evaluation schemas and offline human-in-the-loop workflows produce flattering F1 scores on paper while generating near-total rejection in live operations.

Below are two anonymised case studies: **Aerial Terrain Segmentation** (where I directed the research roadmap) and **Airspace Object Classification** (where I served in a technical advisory and production-readiness audit capacity). Together, they illustrate how these failures surface, how we measured them, and the evaluation hygiene rules we instituted in response.

---

## Case Study 1: Aerial Terrain Segmentation (Covariate Shift, Leakage, and the Inference Latency Wall)

### The Problem: Urban Bias and Covariate Shift

This system automates terrain analysis from high-resolution aerial and satellite imagery across four land-cover classes (*Classes A, B, C, and D*). After pivoting away from an early monolithic multi-class prototype, our initial urban-biased 4× single-class baseline (`Baseline v1`) still degraded severely in rural and forested terrain: a classic geographic covariate shift problem.

To address this, I led a team of 2 to 3 data scientists in retraining terrain-specialised single-class segmentation ensembles. We evaluated several segmentation backbones, including `SegFormer` and `DeepLabV3+` as comparative baselines alongside `YOLO11-seg` and `YOLO26-seg`, and standardised the production ensemble on single-class `YOLO-seg` models after heavier architectures such as `DeepLabV3+` hit memory limits on constrained hardware. Across training iterations, we paired multi-source public datasets (`OpenEarthMap`, `LoveDA`, and `LandCover.ai`) with orientation-invariant augmentations (`flipud=0.5`, rotation), classification-loss tuning, and self-labelled hard-negative background imagery to suppress urban false positives.

### The Evaluation Trap: Micro-Averaging and Cross-Pipeline Leakage

When benchmarking the new specialised models against `Baseline v1`, two measurement traps surfaced immediately:

* **Macro vs. Micro Volatility:** Because individual terrain classes are sparse or completely absent in many aerial tiles, macro-averaging across images was heavily skewed by empty tiles. We standardised on **pixel-level micro-averaged IoU, Precision, Recall, and F1** across a 30% holdout spanning `LandCover.ai` (0.25–0.50 m/px), `LoveDA` (rural/urban, 0.30–0.60 m/px), and `OpenEarthMap` (global generalisation proxy, 0.25–0.50 m/px).
* **Silent Cross-Pipeline Leakage:** During our quantitative audit, we caught a critical caveat. Because earlier training iterations and the holdout split had been constructed by slightly different data pipelines, portions of the test split could overlap with training tiles whenever a model had trained on that source dataset, inflating raw scores. Rather than reporting unverified gains as clean generalisation, we explicitly flagged potentially leaked cells in our benchmark reports (`†` below) and instituted a **uniform, stratified holdout pipeline** to eliminate cross-pipeline leakage going forward.

### Quantitative Benchmark Results (Public Aerial Datasets)

| Terrain Class | Dataset (Zoom, n tiles) | Baseline v1 (IoU / P / R / F1) | Specialised v2 (IoU / P / R / F1) | Tuned v2.1 (IoU / P / R / F1) | Key Takeaway |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Class A** | `LoveDA (Z18, 501)` | 0.194 / 0.206 / 0.762 / **0.325** | 0.241 / 0.541 / 0.302 / **0.388** | N/A | **Leak-free OOD:** +33.5 pts precision; cut false-positive noise |
| **Class A** | `OpenEarthMap (Z19, 2,100)` | 0.382 / 0.584 / 0.526 / **0.553** | 0.465 / 0.761 / 0.544 / **0.635**† | N/A | +8.2 pts F1 on flagged in-domain split† |
| **Class B** | `LoveDA (Z18, 501)` | 0.296 / 0.349 / 0.658 / **0.456** | 0.314 / 0.620 / 0.390 / **0.478** | 0.397 / 0.437 / 0.812 / **0.568** | **Leak-free OOD:** v2 over-pruned recall; v2.1 threshold tuning recovered coverage |
| **Class B** | `OpenEarthMap (Z18, 1,043)` | 0.331 / 0.374 / 0.744 / **0.498** | 0.317 / 0.758 / 0.352 / **0.481**† | 0.651 / 0.735 / 0.850 / **0.788**† | +29.0 pts F1 after threshold calibration (`0.35 → 0.10`)† |
| **Class B** | `OpenEarthMap (Z19, 2,100)` | 0.336 / 0.410 / 0.651 / **0.503** | 0.417 / 0.815 / 0.461 / **0.589**† | 0.641 / 0.739 / 0.829 / **0.781**† | Consistent gains across zoom scales on flagged in-domain split† |
| **Class C** | `OpenEarthMap (Z18, 1,043)` | 0.212 / 0.234 / 0.698 / **0.350** | N/A | 0.461 / 0.593 / 0.674 / **0.631**† | +28.1 pts F1; reduced false positives† |
| **Class C** | `LoveDA (Z19, 2,003)` | 0.373 / 0.487 / 0.614 / **0.543** | N/A | 0.482 / 0.763 / 0.566 / **0.650**† | Strong precision gain (+27.6 pts) on flagged in-domain split† |

<p class="table-note"><sup>†</sup> <strong>Leakage provenance note:</strong> Cells marked with <code>†</code> reflect datasets that were also included in that model's training mix under an earlier pipeline, creating potential tile-level split overlap; catching this overlap during internal auditing directly triggered our migration to a uniform, stratified holdout pipeline. Unmarked <code>LoveDA</code> rows for Class A and Class B represent strictly unseen out-of-distribution (OOD) test splits.</p>

### The Systems Reality: Tiled Inference vs. Latency

Even after improving tile-level F1, deploying onto real operational areas (`1.44` to `4.0 sq km` high-resolution rasters) exposed a second gap: passing full-scene imagery into fixed-resolution segmentation heads caused severe small-object degradation on large scenes.

We ran a controlled local end-to-end runtime ablation across six representative high-resolution test scenes comparing slicing, edge padding, and **overlapping alpha-blending** (to eliminate tile-seam artefacts):

| Test Stage | Slicing | Padding | Alpha Blend (Overlap) | Avg. Local Runtime (s) | Delta vs. Baseline | Mask Quality Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Test 0** (Staging Baseline) | No | No | None | **10.13 s** | Baseline | Severe small-object loss on large scenes |
| **Test 1** | Yes | No | None | **48.41 s** | +378% | Recovers detail, introduces grid-seam cuts |
| **Test 2** | Yes | Yes | None | **48.57 s** | +379% | Negligible cost; cleaner border predictions |
| **Test 3** | Yes | Yes | Yes (0.10) | **136.70 s** | +1,249% | **Best visual continuity** across tile boundaries |
| **Test 4** | Yes | Yes | Yes (0.25) | **136.87 s** | +1,251% | Marginal gain over 0.10 overlap |

<figure class="post-figure">
  <img src="{{ site.url }}/assets/images/posts/tiled-inference-latency-ablation.png" alt="Bar chart showing aerial segmentation tiled inference latency rising from 10.1s at baseline to 136.7s with 0.10 overlap alpha-blending" />
  <figcaption>Figure 1: Average local end-to-end runtime across six high-resolution scenes (1.44–4.0 sq km) as slicing, edge padding, and overlapping alpha-blending are added.</figcaption>
</figure>

Improving mask continuity via overlapping alpha-blending multiplied inference latency by **13.5×** (`10.13s → 136.70s`). That empirical finding forced an engineering pivot before production: we introduced **ONNX Runtime serving** (with graph optimisations and bounded session memory/threading controls) and **in-place array memory optimisation** during mask post-processing and multi-class winner resolution so the serving layer could absorb tiled inference without exceeding compute or memory budgets.

---

## Case Study 2: Airspace Object Classification (When Closed-World Proxies Hide a 94% Rejection Rate)

If the aerial segmentation study illustrates how data pipelines and inference geometry distort offline benchmarks, our second case (an **airspace object classification** POC covering classes *X, Y, and Z*) illustrates a far subtler failure: **evaluation framing and broken human-in-the-loop feedback**.

On this project, I operated in an advisory and production-readiness audit capacity alongside the project's data scientist (who owned the model training, trial log extraction, and confusion-matrix analysis).

### The Benchmark Mirage

Representative early operational trials looked deceptively healthy on paper: operators accepted **87%–94%** of model predictions on interacted tracks. Yet a deeper audit of the trial logs and training repository uncovered three compounding flaws:

1. **Temporal Split Leakage in Training:** Auditing the original CatBoost training codebase revealed **non-chronological shuffle splitting** across radar track segments. This violated time-independence by allowing temporally adjacent points from the same flight track to appear in both train and validation splits. I recommended shifting to grouped track splits with `TimeSeriesSplit`.
2. **Closed-World Taxonomy Blindness:** Those headline 87%–94% acceptance numbers were computed over only **1.4%–3.9%** of total tracks in those sessions. Across all early trials combined, operators interacted with only **2.97%** of **30,686** observed tracks (with another early trial dropping to **39.1%** acceptance). Worse, unmodelled **Out-of-Taxonomy (OOT)** background traffic was silently absorbed into `Class Y`, while `Class X` and `Class Z` predictions suffered **96%–100% rejection**.
3. **UI Change Exposed Reality:** When the interface was updated to allow explicit operator corrections, overall rejection spiked to **~94% (305 of 323 interacted tracks)**, driven by operators manually relabelling predictions into two unmodelled **Out-of-Taxonomy (OOT)** background categories (`297` OOT corrections in total: `276` in OOT Subtype 1 and `21` in OOT Subtype 2). Critically, these actionable labels were being captured **offline outside the model's training loop**.

### How the Same Model Scores 0.80 or 0.02 F1

All F1 scores below were computed at the **track-update row level** (`7,306` / `153,448` / `103,953` rows, matching the table order below). Notably, the two post-UI-update trials (`Options 1b` and `2a`) contained only **1 true `Class X` track (`350` update rows)**, meaning a single framing choice over how OOT false positives are counted moves `Class X` F1 across nearly the entire `[0, 1]` interval:

| Evaluation Framing | Class X F1 | Class Y F1 | Class Z F1 | What the Metric Actually Measures |
| :--- | :--- | :--- | :--- | :--- |
| **Option 1b** (Accepted tracks only, OOT removed; n = 15 tracks, 7,306 rows) | **0.803** | **0.931** | **0.995** | Best-case closed-world ceiling (cherry-picked; 1 Class X track) |
| **Option 3b** (7-trial adjusted baseline, ~85.1k FP rows dropped; 153,448 rows) | **0.573** | **0.965** | **0.719** | Adjusted stakeholder reporting baseline |
| **Option 2a** (Accepted + OOT corrections mapped to Class Y; n = 315 tracks, 103,953 rows) | **0.021** (1.06% prec.) | **0.701** | **0.433** | **True operational false-positive burden** |

Mechanically, the gap across these three framings comes down to how rejected OOT false positives are handled:

* **Option 1b (`0.803` F1, survivorship bias):** Evaluates only the `15` operator-accepted tracks after deleting all `305` rejected tracks (`~29k` OOT false-positive rows vanish), inflating Class X precision to `73.2%`.
* **Option 3b (`0.573` F1, asymmetric row deletion):** Pools all 7 trials after dropping `~85.1k` FP rows in total (`~43.4k` OOT-induced Class X and Class Z FP rows from the last 2 trials plus `~41.7k` Class X FP-on-Class Y rows from the first 5 trials, where the legacy UI lacked OOT labels), artificially lifting Class X F1 without fixing the underlying false-positive burden.
* **Option 2a (`0.021` F1, true operational burden):** Retains the `297` rejected OOT tracks (mapped to Class Y). Because the model repeatedly predicted `Class X` on OOT traffic (`~29k` FP rows against only `1` true Class X track of `350` rows), Class X recall remains `88.9%` while precision collapses from `73.2%` to `1.06%`.

<figure class="post-figure">
  <img src="{{ site.url }}/assets/images/posts/airspace-eval-framing.png" alt="Grouped bar chart comparing airspace classification Class X, Class Y, and Class Z F1 scores across Options 1b, 3b, and 2a" />
  <figcaption>Figure 2: Identical underlying model predictions evaluated under three different track-update row-level framings. Class X F1 collapses from 0.80 to 0.02 once operational out-of-taxonomy false positives are retained.</figcaption>
</figure>

### The Go/No-Go Audit Decision

During the pre-deployment review, I red-teamed the adjusted evaluation framings (which asymmetrically dropped **~29k to ~85.1k rows** across Options 2b and 3b to reconcile legacy labels). While those numbers validated the old collapsed taxonomy, they provided zero evidence that the model could reliably separate in-taxonomy classes from out-of-taxonomy background traffic in live airspace.

That audit contributed directly to a leadership **no-go decision** against presenting inflated metrics or pushing to premature deployment. When subsequent retraining naively added OOT background tracks to the dataset and collapsed into majority-class OOT predictions, I diagnosed the failure as imbalance laziness vs. feature insufficiency and recommended pivoting to a **two-stage cascaded architecture** (a dedicated In-Taxonomy vs. Out-of-Taxonomy filter upstream of sub-classification) paired with operator UX journey mapping and kinematic-invariant track signatures / Multi-Hypothesis Tracking (MHT) for fragmented track alignment.

---

## Four Evaluation Hygiene Rules for High-Stakes ML

1. **Audit the Split Before the Score:** Cross-pipeline tile overlap (Case Study 1) and non-chronological track shuffling (Case Study 2) both manufacture artificial confidence. Treat evaluation-set construction as a first-class engineering artefact with strict provenance.
2. **Never Average Away Operational Asymmetry:** Always report micro-averaged and per-class precision/recall alongside the exact denominator (`n` evaluated vs. `n` total, and row-level vs. entity-level granularity).
3. **Measure the Full Inference Loop, Not Just the Model Head:** A model that passes offline tile benchmarks but takes 136 seconds per scene, or relies on operator corrections that never re-enter the training loop, is not production-ready.
4. **Reward Evidence-Based "No-Go" Decisions:** The highest-leverage intervention in applied ML is often proving *why a system should not ship yet* before a misleading benchmark hardens into an operational liability.
