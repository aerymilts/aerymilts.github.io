---
title: "Solving Zero-Label Physical Segmentation with Classical CV and Information Theory"
layout: post
date: 2026-11-21 12:00
tag:
  - computer vision
  - information theory
  - first principles
category: blog
author: kennethho
hidden: false
description: "How a physics-first 2D-to-1D reduction, CIELAB superpixels, and Jensen-Shannon Divergence solved zero-label soil texture segmentation under glare and marker-line artifacts."
---

When you have zero labelled training data, reaching for a deep neural network is usually the fastest way to build a brittle system. Often the highest-leverage move is to avoid machine learning altogether and exploit the physics of the problem first.

At Yara Digital Farming, I worked on **SmartJar**, a smartphone-based soil texture analysis prototype designed to help smallholder farmers estimate the **sand, silt, and clay** composition of their soil without laboratory equipment. Because we had no labelled dataset of soil-jar segmentation masks, training a supervised CNN was off the table. Instead, solving the problem required stepping back, reasoning from the physics of the measurement, and combining classical computer vision with information theory.

---

## 1. First-Principles Reduction: Turning 2D Segmentation into 1D Physics

The classic "jar test" for soil texture relies on **Stokes' Law**: when soil is shaken with water in a straight-sided glass jar and left to settle, heavier particles sink first. Over time, the soil separates into three vertically stacked, density-ordered horizontal layers beneath a supernatant water column:

1. **Sand** (bottom layer: heaviest, settles within minutes)
2. **Silt** (middle layer: settles over hours)
3. **Clay** (top soil layer: finest, settles last)

Because the jar is a uniform cylinder, the relative volume fraction of each soil component reduces to its **vertical layer height**. We did not need pixel-perfect 2D semantic segmentation of arbitrary shapes; we needed a reliable 1D vertical profile of homogeneous color-texture bands inside the jar.

---

## 2. Where Standard Unsupervised Pipelines Break

Our initial zero-label computer vision pipeline followed a textbook classical sequence:

1. **Background Removal:** Isolate the foreground glass jar from cluttered farm and table backgrounds.
2. **Superpixel Over-Segmentation (SLIC in `LABxy` space):** Partition the jar image into perceptually uniform superpixels using CIELAB color channels (`L*`, `a*`, `b*`) alongside spatial coordinates (`x`, `y`).
3. **Region Adjacency Graph (RAG) & Hierarchical Merging:** Connect neighboring superpixels weighted by mean color distance and merge adjacent regions iteratively.

In controlled illustrations, SLIC + RAG works cleanly. On real smartphone photos of glass jars, **three physical imaging artifacts broke the adjacency and mean-color assumptions**:

* **Specular Glass Reflections:** Vertical glare streaks across curved glass created high-luminance barriers that sliced a single continuous soil band into disconnected left and right segments.
* **Hand-Drawn Jar Markings:** Measurement lines drawn on the outside of the jar physically bisected homogeneous soil layers, preventing standard RAG merging (which only merges *spatially adjacent* regions) from reuniting the split halves.
* **Lighting Gradients:** Uneven ambient illumination across the curved jar introduced intra-layer color drift that caused naive mean-color adjacency thresholds to over-fragment a single soil stratum.

---

## 3. The Custom Fix: Distribution-Based Merging (Jensen-Shannon Divergence) + Vertical Projection

To overcome specular reflections, marker lines, and lighting gradients without labelled training data, we replaced naive mean-color adjacency merging with a **distribution-similarity and spatial-proximity merge**:

<figure class="post-figure">
  <img src="{{ site.url }}/assets/images/posts/smartjar-pipeline.svg" alt="SmartJar pipeline schematic showing a glare- and marker-split soil jar photo processed via SLIC LABxy superpixels, JSD plus vertical proximity non-adjacent merging, 1D y-axis modal projection, and USDA soil texture mapping" />
  <figcaption>Figure 1: End-to-end SmartJar zero-label segmentation pipeline, reuniting non-adjacent band fragments across glare and marker lines before collapsing to a 1D vertical profile.</figcaption>
</figure>

* **Color-Distribution Similarity (CIELAB + Jensen-Shannon Divergence):** Working in `CIELAB` separates specular glare (`L*` luminance) from underlying soil chrominance (`a*`, `b*`). Comparing full region color distributions with **Jensen-Shannon Divergence (JSD)** (which is symmetric, bounded, and finite even when histogram bins have zero support in one region) avoids both the brittleness of mean-color thresholds and the singularities of raw KL divergence.
* **Non-Adjacent Reunification via Vertical Proximity:** Two regions separated horizontally by a glare streak or a vertical mark still share almost identical vertical (`y`-axis) coordinates and color-distribution entropy. By combining low JSD with vertical spatial proximity, the algorithm reunited fragmented bands across reflections and markings into coherent soil layers.
* **1D Modal Height Extraction & USDA Soil Texture Mapping:** Once regions were merged, we collapsed the 2D mask onto the vertical axis by taking the **modal (most frequent) segment label per horizontal row (`y`-coordinate)**. This acted as a robust 1D filter against residual edge glare, yielding clean vertical pixel heights per stratum. Excluding the top water column and normalizing the three soil band heights (`h_sand`, `h_silt`, `h_clay`) by total settled soil height converts directly into volume percentages on the ternary **USDA Soil Texture Triangle**. For example, in one illustrative prototype jar measuring `358 px` water (excluded), `190 px` clay, `280 px` silt, and `569 px` sand (`1,039 px` total soil), normalizing yields **54.8% Sand, 26.9% Silt, and 18.3% Clay**, classifying the sample as **Sandy Loam**.

---

## 4. Scope, Limitations, and Lessons

Because this work was delivered as a **Checkpoint 1 zero-label algorithmic prototype** (with formal quantitative laboratory validation scoped for a later phase after my departure, and with patent filing initiated separately), performance during my tenure was evaluated visually across diverse smartphone samples rather than against a ground-truth laboratory benchmark; the numerical example above is an illustrative single-jar prototype output rather than a lab-validated accuracy metric.

Even as a prototype, SmartJar reinforced a principle that carries directly into modern AI engineering: **understand the physical and structural invariants of your domain before reaching for `model.fit()`**. When you exploit geometry (1D settling order) and information theory (JSD over color distributions), you can solve real-world measurement artifacts transparently, debug every failure mode visually, and build a strong baseline before collecting a single training label.
