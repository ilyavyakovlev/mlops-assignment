# MLOps Assignment Report

## 1. Serving Configuration

vLLM was deployed on a Nebius Cloud H100 SXM 80 GB instance serving
`Qwen/Qwen3-30B-A3B-Instruct-2507` (MoE, ~3B active parameters per token).

| Flag | Value | Rationale |
|---|---|---|
| `--dtype` | `bfloat16` | H100 native dtype; better numerical stability than fp16 at no throughput cost |
| `--max-model-len` | `8192` | Covers 3 K schema + prompt + output with headroom; smaller context = less KV pressure per slot |
| `--gpu-memory-utilization` | `0.92` | Leaves ~6 GB buffer for MoE expert weight residency |
| `--max-num-seqs` | `8` (tuned from 64) | Limits concurrent sequences; see Phase 6 |
| `--enable-prefix-caching` | ✓ | DB schemas repeat across requests — prefix cache hit rate was consistently high |
| `--enable-chunked-prefill` | ✓ | Improves batching across variable-length prompts |
| `--trust-remote-code` | ✓ | Required to load Qwen3 tokenizer correctly with transformers 4.47 |

**Note on compatibility.** The pinned `transformers` version must be `<5.0`. Version 5.x
removed `all_special_tokens_extended` from the slow tokenizer path; vLLM 0.10.2 calls it
unconditionally. A direct patch to `vllm/transformers_utils/tokenizer.py` was applied on the VM
(`getattr` fallback), and `pyproject.toml` now pins `transformers>=4.46.0,<5.0`.

---

## 2. Baseline Evaluation

Evaluation was run on 30 questions drawn from the BIRD development set across 9 databases.
The agent used the `generate → execute → verify → [revise → execute →]* verify` loop
with `MAX_ITERATIONS = 3`.

| Metric | Value |
|---|---|
| Questions | 30 |
| Overall pass rate | **23.33%** (7 / 30) |
| Pass rate at iteration 0 | 23.33% |
| Pass rate at iteration 1 | 23.33% |
| Pass rate at iteration 2 | 23.33% |
| Median latency (per question) | ~0.8 s |

**Does the loop earn its keep?** Not in the baseline. The per-iteration pass rate is completely
flat: every question answered correctly was already correct on the first attempt, and no
failing question was recovered by revise cycles. Root cause analysis identified two failure
modes:

1. **Verifier false positives.** 16 of the 23 failing questions stopped at iteration 1
   (`verify_ok = True`), meaning the verifier accepted a wrong answer. With no ground-truth
   reference, the verifier can only reason about plausibility, and a syntactically valid result
   looks plausible even if semantically wrong.

2. **Wrong string literals.** Many failures stemmed from SQLite's case-sensitive comparisons:
   the model emitted `gender = 'm'` instead of `'M'`, `label = 'carcinogenic'` instead of
   `'+'`, `element = 'Calcium'` instead of `'ca'`, `Admission = 'outpatient clinic'` instead
   of `'-'`. The model had no way to know the actual stored values.

---

## 3. SLO Optimization

**Target:** P95 end-to-end agent latency < 5 s, sustained over 5 minutes.

### Baseline load test (before tuning) — 10 RPS, `--max-num-seqs 64`

| Metric | Value |
|---|---|
| Requested RPS | 10 |
| Success rate | 3.5 % (106 / 3 000) |
| Timeouts | 64 % |
| Latency P50 | 12.9 s |
| Latency P95 | **94.1 s** |
| Latency P99 | 109.9 s |

**Observation.** The queue was completely saturated. With `--max-num-seqs 64`, vLLM accepted
up to 64 concurrent sequences. The agent makes 2–3 serial LLM calls per question; at ~2–5 s
per call, each request holds a slot for 6–15 s. At 10 RPS, 60–150 requests are in flight
simultaneously, far exceeding the 64-slot budget, and the KV cache was exhausted within
seconds of the test starting.

**Hypothesis.** Reducing `--max-num-seqs` limits queue growth and allows the requests that
do run to complete within their timeout window. The true sustainable throughput of a 30 B MoE
model serving multi-call agent requests on one H100 is ~0.15–0.5 RPS; a test at 2 RPS
is a fairer comparison of per-request latency.

**Change:** `--max-num-seqs 64 → 8`

### After-tuning load test — 2 RPS, `--max-num-seqs 8`

| Metric | Value |
|---|---|
| Requested RPS | 2 |
| Success rate | 7.3 % (44 / 600) |
| Timeouts | 68 % |
| Latency P50 | **2.0 s** |
| Latency P95 | 115.9 s (timeout-dominated) |
| Latency P99 | 117.5 s |

**Result.** P50 improved from 12.9 s to **2.0 s** — requests that complete finish quickly.
P95 remains timeout-dominated because even 2 RPS exceeds sustainable throughput (~0.15 RPS).
The 5 s P95 SLO is not achievable with this model and single-GPU configuration at any RPS
above ~0.5.

**What would hit the SLO.** Prefix caching (already enabled) helps on repeated schemas.
Tensor parallelism across multiple H100s, a distilled 7 B model, or speculative decoding
would be needed to reach 5 s P95 at meaningful throughput.

---

## 4. Agent Value

The verify → revise loop did not improve the pass rate in either the baseline or the
after-tuning run — per-iteration accuracy was flat in both cases (23.33 % and 36.67 %
respectively across all iterations). The loop's current limitation is that the verifier
lacks ground-truth output to compare against, making it dependent on semantic plausibility
reasoning which is unreliable for edge-case encodings and schema conventions.

The **36.67 % improvement over baseline 23.33 %** came entirely from prompt and schema
changes: adding inline sample values to the schema rendering (`-- e.g.: 'M', 'F'`) so the
model sees the actual stored values, and strengthening the VERIFY and REVISE prompts with
explicit case-sensitivity warnings. This is a +57 % relative improvement in pass rate.

The loop architecture itself is sound. Given a stronger verifier — one that executes both
the candidate and a reference query, or uses few-shot execution-based comparison — the revise
cycles would add measurable value.

---

## 5. What I'd Do With More Time

- **Execution-based verifier.** Run a reference query (simplified or paraphrased gold SQL)
  and compare rows rather than asking the LLM to judge plausibility. This removes the false-
  positive problem entirely.
- **Schema enrichment beyond sample values.** Add column cardinality, min/max, and foreign-key
  join paths to the schema context. Many failures involved ambiguous column selection across
  joins.
- **Multi-GPU serving.** Tensor-parallel vLLM across 2–4 H100s would bring per-request
  latency under 2 s and allow the SLO to be met at 2–5 RPS with a 30 B model.
- **Model quantisation.** AWQ or GPTQ 4-bit would halve memory usage, enabling larger
  `--max-num-seqs` and higher throughput on the same hardware.
- **Eval set expansion.** 30 questions is too small to draw statistically robust conclusions
  about per-database or per-complexity-tier performance. BIRD dev has 1 534 questions; running
  the full set would give reliable signal for prompt iteration.
