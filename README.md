# Qwen3.5-9B Inference Benchmark

## Overview
This repository contains the benchmarking scripts, test assets, and result logs for evaluating the unquantized `Qwen/Qwen3.5-9B` vision-language model. The primary objective is to establish a reliable, steady-state baseline inference latency for a Browser Automation Agent tasked with UI element tag extraction.

## Hardware & Environment
* **GPU:** NVIDIA A100-SXM4-80GB (85.1 GB VRAM)
* **Precision:** `bfloat16` (Unquantized)
* **Framework:** Hugging Face `transformers`

## Repository Structure
* **`Qwen3.5-9B.ipynb`**: The complete execution notebook containing the benchmark logic, image processing, and latency calculations.
* **`/Screenshots`**: Contains terminal outputs validating the benchmark execution and final latency calculations.
* **`/Tests`**: Contains the visual assets used for the prompt (Normal and Taggified screenshots), as well as the raw `latencies_100_calls.txt` data log.

## Methodology
To ensure stability and rule out initialization or cold-start variances, the model was subjected to a comprehensive **100-call continuous benchmark** split across two distinct task types.

**Test Cases Evaluated:**
1. *Click Cart* (Navigation action - identifying a single icon)
2. *Fill Last Name* (Input action - identifying a specific text field)

**Process:** The GPU was subjected to a 3-call warmup phase prior to recording. `torch.cuda.synchronize()` was strictly utilized to guarantee accurate, blocking latency measurements. A streamlined system prompt was utilized to isolate true hardware generation capabilities and model response times without artificially inflating prefill latency overhead.

## Results (100-Call Stability Test)
The continuous loop validated the baseline latency across both navigation and text-input extraction tasks on an A100. 

**Test 1: Click Cart (Navigation Action)**
* **Average Time:** 2.968s
* **90th Percentile:** 3.000s

**Test 2: Fill Last Name (Input Action)**
* **Average Time:** 4.437s
* **90th Percentile:** 4.514s

**Combined Overall Stability**
* **Average Latency:** 3.703s
* **Fastest Call:** 2.902s
* **Slowest Call:** 4.601s

*Raw latency data for all iterations can be found in `Tests/latencies_100_calls.txt`.*
