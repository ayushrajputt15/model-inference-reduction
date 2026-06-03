# Model Inference Reduction
Benchmarking Qwen3.5-9B vision-language model inference speed for a Browser Automation Agent.

**Hardware:** NVIDIA A100-SXM4-80GB | **Precision:** bfloat16 | **Runs:** 100 calls per test case

---

## Repo Structure
```
model-inference-reduction/
└── qwen/
    └── qwen3.5-9B/
        ├── Qwen3.5-9B-100call.ipynb    ← full benchmark notebook
        └── latencies_100_calls.txt      ← raw timing data (100 values)
```

---

## Benchmark Results

### Test 1 — Click the shopping cart icon
| Metric | Value | What it means |
|--------|-------|---------------|
| Average Time | 2.968s | Typical call duration |
| 90th Percentile | 3.000s | 90% of calls finish within this |

### Test 2 — Fill the Last Name field
| Metric | Value | What it means |
|--------|-------|---------------|
| Average Time | 4.437s | Typical call duration |
| 90th Percentile | 4.514s | 90% of calls finish within this |

### Overall Stability (100 calls, Test 1 only)
| Metric | Value | What it means |
|--------|-------|---------------|
| Average Latency | 2.868s | Mean across all 100 calls |
| Fastest Call | 2.830s | Best case — short output, warm GPU |
| Slowest Call | 2.913s | Worst case — slight GPU scheduling delay |

> Latency is extremely stable — only 83ms difference between fastest and slowest call across 100 runs.

---

## Time Breakdown — Where does each second go?

### Test 1 — Click the shopping cart icon (avg over 100 runs)
| Stage | Time | % of Total | What happens here |
|-------|------|------------|-------------------|
| Image loading | 124.72ms | 4.3% | PNG files read from disk, converted to RGB |
| Tokenization | 81.0ms | 2.8% | Images converted to patches, text tokenized |
| **Inference (GPU)** | **2710.32ms** | **92.9%** | **Model generates output tokens one by one on GPU** |
| Decoding | 0.2ms | 0.0% | Output token IDs converted back to text |
| **TOTAL** | **2916.23ms** | **100%** | |

### Test 2 — Fill the Last Name field (avg over 100 runs)
| Stage | Time | % of Total | What happens here |
|-------|------|------------|-------------------|
| Image loading | 98.97ms | 2.3% | PNG files read from disk, converted to RGB |
| Tokenization | 83.69ms | 1.9% | Images converted to patches, text tokenized |
| **Inference (GPU)** | **4110.88ms** | **95.7%** | **Model generates output tokens one by one on GPU** |
| Decoding | 0.21ms | 0.0% | Output token IDs converted back to text |
| **TOTAL** | **4293.76ms** | **100%** | |

### Overall Combined Breakdown
| Stage | Time | % of Total |
|-------|------|------------|
| Image loading | 111.8ms | 3.1% |
| Tokenization | 82.3ms | 2.3% |
| **Inference (GPU)** | **3410.6ms** | **94.6%** |
| Decoding | 0.2ms | 0.0% |
| **TOTAL** | **3605.0ms** | **100%** |

---

## What each metric means

**Image Loading (~100-125ms)**
Time to read PNG screenshot files from disk and convert to RGB format.
Accounts for ~3-4% of total time. Could be reduced by keeping images in memory.

**Tokenization (~81-84ms)**
Time to convert both screenshots into image patches and text into tokens.
Includes vision encoder preprocessing. Accounts for ~2-3% of total time.

**Inference / GPU (~2700-4100ms)**
The actual model computation on GPU — generating output tokens one by one (autoregressive decoding).
This is **93-96% of total time** and the only stage worth optimizing.
Varies between test cases because longer outputs take more tokens to generate.

**Decoding (~0.2ms)**
Converting output token IDs back into readable text.
Completely negligible — not worth optimizing.

---

## Key Finding
**Inference is 93-96% of total latency.**
Image loading and tokenization together account for less than 7%.
Any meaningful optimization must target the GPU inference stage directly.

The two test cases differ by ~1.4s average because Test 2 generates a longer JSON response (fill action with value field) vs Test 1 (simple click action).

---

## Optimization Attempts

| Method | Avg Time | vs Baseline | Notes |
|--------|----------|-------------|-------|
| BF16 unquantized ✅ | 3.605s | baseline | Stable, works perfectly |
| NF4 4-bit (BitsAndBytes) ❌ | ~22s | 6x slower | Dequantization overhead negates savings on A100 |
| AWQ 4-bit ❌ | blocked | — | gptqmodel kernel incompatible with torch 2.8 |
| vLLM ❌ | blocked | — | C extension conflict with torch 2.8 on Lightning.ai |

**Root cause of AWQ/vLLM failures:** Lightning.ai uses torch 2.8 which is too new for current AWQ kernels and vLLM stable release. Both require torch ≤ 2.6.

**Next step:** Run on clean environment with torch 2.6 + vLLM + AWQ for proper quantization benchmark.
