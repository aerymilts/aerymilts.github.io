---
title: "Monitoring Adaptive Adversaries: Graph Attribution, Sequential Pattern Mining, and Safe Risk-Rule Lifecycles"
layout: post
date: 2026-11-21 13:00
tag:
  - adversarial ml
  - risk modelling
  - graph analytics
  - observability
category: blog
author: kennethho
hidden: false
description: "Architectural lessons from TikTok Live AML/refund fraud and Binance Web3 risk on hub-gated graph attribution, DeepSHAP + PrefixSpan discovery, and fail-open monitor observability."
---

In standard machine learning, the data distribution shifts slowly and passively. In adversarial risk control, the data distribution actively probes your decision boundaries, splits forbidden actions across multiple accounts, and adapts the moment a rule goes live.

Over four years leading risk data science and security analytics across **TikTok Live (ByteDance)** and **Binance Web3**, I found that catching adaptive adversaries at scale is rarely about training a single stronger classifier. Instead, durable defence rests on three architectural pillars:

1. **Separating high-recall perception from high-precision defence**, using graph topology and device priors to resolve hub ambiguity.
2. **Treating model interpretability as a discovery and compression engine**, rather than post-hoc reporting.
3. **Treating risk rules and monitors like production software**, with shadow deployments, incremental coverage accounting, and silent-failure telemetry.

---

## Pillar 1: Two-Tier Detection and Hub-Aware Graph Re-Ranking (TikTok Live AML & Refund Fraud)

At ByteDance, our Business Risk unit safeguarded TikTok Live against organised refund fraud (user-streamer collusion) and anti-money laundering (placing and layering illicit funds through invalid livestreams).

| Layer | Operating Objective | Precision / Recall Profile | Primary Methods |
| :--- | :--- | :--- | :--- |
| **Risk Perception** (Proactive) | Surface novel attack vectors and early anomalies before losses compound | **High Recall, Lower Precision** | Statistical Process Control (Nelson Rules), Isolation Forest, DBSCAN, Sequence Mining |
| **Risk Defence** (Reactive) | Block withdrawals, freeze accounts, and trigger automated enforcement | **High Precision, Lower Recall** | Calibrated risk rules, hub-gated graph scores, device-fingerprint confirmation |

### The Anomaly Ensemble + Bipartite Graph Re-Ranking

We built an ensemble combining four complementary signals:

* **Business heuristics:** Flagging structural impossibilities, such as disproportionate gifting income on a brand-new account with minimal unique viewer engagement.
* **Isolation Forest:** Unsupervised isolation of multi-dimensional revenue and transaction outliers.
* **Statistical Process Control (Nelson Rules):** Specifically **Rule 1** (points > 3σ from the mean), **Rule 4** (14+ consecutive alternating oscillations, signalling synthetic pacing), and **Rule 8** (8 consecutive points outside ±1σ on both sides of the mean, indicating non-random bimodal switching).
* **Behavioural Clustering (DBSCAN):** Clustering a streamer's livestreams by audience engagement statistics and flagging broadcasts whose revenue fell outside 1σ of their engagement cluster.

To suppress false positives from the raw ensemble, we layered **seed-based bipartite community detection** on top:

1. **Seed Initialisation:** Streamers positively confirmed through manual case review form the initial seed set.
2. **Interaction-Weighted Propagation:** Users interacting with seed streamers inherit a suspicion weight scaled by interaction strength. Conceptually, a high-commitment financial edge (direct gifting/coin transfer) carries far more signal than a passive view or chat message, preventing casual spectators from polluting the graph.
3. **Downstream Re-Ranking:** Other streamers accumulating weighted risk flow from those users receive a graph proximity score that acts as a re-ranking boost over the raw unsupervised ensemble output. This promotes correlated collusion networks to the top of the investigation queue while demoting isolated, one-off revenue spikes.

<figure class="post-figure">
  <img src="{{ site.url }}/assets/images/posts/tiktok-bipartite-reranking.svg" alt="Flowchart showing confirmed malicious streamer seeds propagating weighted suspicion through interacting users to re-rank raw anomaly ensemble outputs via hub-gated behavioural and device priors" />
  <figcaption>Figure 1: Two-tier perception and hub-gated bipartite re-ranking architecture used to separate organised laundering rings from isolated anomalies.</figcaption>
</figure>

### Solving the Edge Case: Disambiguating Legitimate "Whales" from Laundering Hubs

Graph centrality alone breaks down on high-degree nodes: a legitimate "whale" gifter looks topologically identical to a money-laundering mule. We gated high-degree hub flags with **behavioural priors and device telemetry**:

* **Account-history priors:** Authentic whales exhibit established account age and near-zero historical refund rates. Brand-new accounts (particularly those newly provisioned in developing regions suddenly gifting disproportionate volumes) carry high adversarial prior probability.
* **Emulator & device-environment fingerprints:** Organised fraud rings spin up virtualised farms. Checking low-level device telemetry exposed reliable synthetic signatures: **nearly full disk space** combined with a **near-zero time delta between OS installation and app installation** strongly indicated freshly cloned emulators rather than organic consumer devices.

**Operational Impact:** Uncovered **3 distinct laundering networks** with **$2M USD in fund inflows** over a 3-month window (**$800k USD still in circulation** at discovery). On refund fraud, accelerated detection lead time by **2–7 days**, intercepted tens of fraudulent streamers daily (~$50k USD in attempted withdrawals), and drove the overall refund rate down from **10% to 2%**.

---

## Pillar 2: Interpretability as a Discovery and Compression Engine (DeepSHAP + PrefixSpan)

Sequential deep models capture multi-step refund fraud well, but suffer from high event-vocabulary cardinality and opacity. We integrated **DeepSHAP** into our sequential risk pipeline:

* **Model Compression via Global Attribution:** Aggregating position-level DeepSHAP attributions against a null reference sequence quantified each event type's marginal log-odds contribution to fraud risk. Safely pruning noisy long-tail UI and telemetry events with near-zero global attribution reduced input cardinality and **halved model training time** without sacrificing detection power.
* **Attack-Vector Discovery:** Clustering sequence embeddings of confirmed refunders and running **`PrefixSpan` sequential pattern mining** surfaced recurring multi-step exploit trajectories accounting for **~50% of observed fraudulent behaviours**. For example, exploratory subsequence analysis revealed a distinct behavioural-provenance pattern: fraudulent high-value gifters disproportionately entered target livestreams via **DM (private messaging) or external app links** rather than organically through the **FYP (For You Page) recommendation feed**. Surfacing these interpretable entry-and-gifting trajectories directly inside an internal rule debugger cut rule creation and debugging time by **50%**.

---

## Pillar 3: On-Chain Attribution and Safe Monitor Deployment (Binance Web3 Risk)

At **Binance**, transactions are public and immutable, but adversaries rotate disposable wallets and fraudulent dApp contracts in seconds.

### Seed-Based On-Chain Attribution for Phishing & Malicious dApps

Starting from an internal blacklist of confirmed scam and phishing dApp wallets, we traced newly observed dApp wallets interacting with user approval and transfer flows. Linking a new wallet directly or indirectly to the blacklist (via shared cash-out sinks transferring ill-gotten gains to the same aggregation wallet, shared contract creator/deployer addresses, or common funding wallets) exposed associated phishing infrastructure proactively. Combined with automated case-assessment rules and investigation dashboards, this improved coverage of risky domains by **20%** and fake tokens by **16%**, increased loss prevention by **$2M USD**, and cut case-resolution time by **20%**.

### The Risk Strategy Lifecycle: Preventing Silent Monitor Degradation

I built a standardised three-stage **Risk Strategy Lifecycle** (`Backtest → Grayscale Shadow A/B → Production Live Enforcement`) and 0-to-1 analytics layer with Prometheus real-time alerting.

<figure class="post-figure">
  <img src="{{ site.url }}/assets/images/posts/binance-risk-lifecycle.svg" alt="Flowchart of the three-stage Risk Strategy Lifecycle from Backtest to Grayscale Shadow A/B to Production Enforcement with Prometheus real-time observability gates and rollback loop" />
  <figcaption>Figure 2: Three-stage monitor release lifecycle with real-time Prometheus observability gates to catch silent fail-open regressions.</figcaption>
</figure>

Two measurement practices proved essential:

| Measurement Practice | What It Tracks | Why Standard Metrics Miss It |
| :--- | :--- | :--- |
| **Incremental vs. Independent Precision/Recall** | **Incremental:** net new coverage and FPs added on top of the existing stack. **Independent:** non-overlapping coverage and FPs unique to the new strategy. | A new model can show high standalone recall while adding zero incremental coverage because existing rules already catch the easy cases, leaving only its false positives behind. |
| **Unclassified (`Null`) Rate Observability** | Real-time Prometheus alerts on (1) overall traffic volume, (2) shifts in High/Medium/Low risk ratios, and (3) **spikes in `Null` / unclassified responses**. | When an upstream feature table stalls, an RPC node times out, or a malformed payload crashes a parser, risk classifiers fail open into `Null`. Monitoring only flagged volume misses silent fail-open outages. |

---

## Transferable Principles for Monitoring Autonomous Systems

1. **Single-turn signals are always gameable:** Combine point-in-time anomalies with sequence mining (`PrefixSpan`) and interaction-graph attribution.
2. **Topology without context creates false positives:** Gate high-degree hubs with behavioural history and environment telemetry.
3. **Interpretability should close the loop:** Attribution (`DeepSHAP`) should both compress the monitor and translate latent patterns into auditable failure modes.
4. **Always monitor the monitor:** Track incremental coverage and alert on `Null`/unclassified fail-open spikes.
