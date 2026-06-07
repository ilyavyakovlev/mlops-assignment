# Implementation Plan: LLM Inference + Observability

## Infrastructure strategy

| Stage | Where | LLM backend |
|---|---|---|
| Phases 0–4 (setup, agent, tracing, eval harness) | Local laptop | Nebius AI Studio API — Qwen3 hosted, no GPU needed |
| Phases 1, 5, 6 (vLLM, real evals, SLO testing) | Nebius Cloud H100 VM | vLLM serving Qwen3-30B-A3B locally |

The o11y stack (Prometheus + Grafana + Langfuse) runs via `docker compose` in both environments.

---

## What's already complete

- `agent/server.py` — FastAPI server + Langfuse callback wiring
- `agent/execution.py` — SQL execution helper
- `agent/schema.py` — DB schema renderer
- `load_test/driver.py` — async load driver
- `scripts/load_data.py` — BIRD data downloader
- `docker-compose.yml` — full o11y stack
- `infra/grafana/provisioning/` — datasource + dashboard provider already configured
- `serving.json` — 2 starter panels (requests running, gen tokens/s)

---

## Phase 0 — Setup

### Local (do this first)

1. Get a Nebius AI Studio API key at **studio.nebius.ai → API Keys**
2. Paste it as `LLM_API_KEY` in `.env`
3. `uv sync`
4. `docker compose up -d`
5. Confirm: Prometheus at http://localhost:9090, Grafana at http://localhost:3000 (admin/admin), Langfuse at http://localhost:3001
6. `uv run python scripts/load_data.py` — downloads ~500 MB BIRD dev set, creates `evals/eval_set.jsonl` and `load_test/perf_pool.jsonl`

### Nebius Cloud VM (provision when ready for Phases 1/5/6)

1. Provision H100 VM via Nebius console (H100 SXM 80 GB, Ubuntu 22.04)
2. SSH with 5 port forwards:
   ```bash
   ssh -L 3000:localhost:3000 -L 9090:localhost:9090 \
       -L 3001:localhost:3001 -L 8000:localhost:8000 \
       -L 8001:localhost:8001 <user>@<vm-ip>
   ```
3. Clone repo, `uv sync`, `docker compose up -d`, `uv run python scripts/load_data.py`
4. In `.env` on the VM: comment out the Studio block, uncomment the local vLLM block

**Deliverables:** Five ports reachable in browser, BIRD data under `data/bird/`, `.env` configured.

---

## Phase 1 — vLLM Configuration (Nebius VM)

**File to edit:** `scripts/start_vllm.sh`

Initial config for Qwen3-30B-A3B (MoE) on H100 80 GB, workload = 1.5–3K prompt tokens, short structured outputs, ~2–3 sequential LLM calls per user request:

| Flag | Value | Rationale |
|---|---|---|
| `--max-model-len` | `8192` | Covers 3K schema+prompt+output; smaller = less KV cache pressure per slot |
| `--enable-prefix-caching` | ✓ | DB schemas repeat across requests → big cache hit rate boost |
| `--gpu-memory-utilization` | `0.92` | Leave buffer; MoE expert weights stay resident |
| `--max-num-seqs` | `64` | Start high for throughput; dial down if P95 latency blows out |
| `--dtype` | `bfloat16` | H100 native; better numerical stability than FP16 |
| `--enable-chunked-prefill` | ✓ | Better batching of variable-length prompts |
| `--disable-log-requests` | ✓ | Reduces per-request logging overhead at high RPS |

No `--tensor-parallel-size` needed (single GPU). This is a starting point — Phase 6 iterates on it.

Manual smoke test: fire 3–5 questions from `evals/eval_set.jsonl` with curl and confirm sensible SQL comes back.

**Deliverables:** vLLM at http://localhost:8000, `screenshots/vllm_manual_query.png`, config flags + rationale in `REPORT.md`.

---

## Phase 2 — Grafana Dashboard

**File to edit:** `infra/grafana/provisioning/dashboards/serving.json`

Extend the 2 existing panels with 6 more (8 total):

| # | Panel | PromQL | Category |
|---|---|---|---|
| 1 | Requests running *(existing)* | `vllm:num_requests_running` | Throughput |
| 2 | Gen tokens/s *(existing)* | `rate(vllm:generation_tokens_total[1m])` | Throughput |
| 3 | E2E latency P50/P95/P99 | `histogram_quantile(0.95, rate(vllm:e2e_request_latency_seconds_bucket[1m]))` | Latency |
| 4 | Time-to-first-token P50/P95 | `histogram_quantile(0.95, rate(vllm:time_to_first_token_seconds_bucket[1m]))` | Latency |
| 5 | Time-per-output-token P95 | `histogram_quantile(0.95, rate(vllm:time_per_output_token_seconds_bucket[1m]))` | Latency |
| 6 | Request queue depth | `vllm:num_requests_waiting` | Throughput |
| 7 | Request success rate | `rate(vllm:request_success_total[1m])` | Throughput |
| 8 | KV cache usage % | `vllm:gpu_cache_usage_perc` | KV Cache |

Can be built and committed before the VM is up; panels will light up once vLLM scraping starts.

**Deliverables:** `serving.json` committed, `screenshots/grafana_serving.png`.

---

## Phase 3 — Agent Implementation (local, Nebius AI Studio)

### `agent/prompts.py`

Six template strings to fill in:

- `GENERATE_SQL_SYSTEM` — SQL expert; output only a ` ```sql ``` ` block; SQLite dialect; no prose
- `GENERATE_SQL_USER` — placeholders: `{schema}`, `{question}`
- `VERIFY_SYSTEM` — return exactly `{"ok": bool, "issue": str}`; mark `ok=false` on: SQL error, zero rows when question implies rows, columns don't answer the question
- `VERIFY_USER` — placeholders: `{question}`, `{sql}`, `{execution}`
- `REVISE_SYSTEM` — SQL expert fixing a broken query; output only ` ```sql ``` ` block
- `REVISE_USER` — placeholders: `{schema}`, `{question}`, `{previous_sql}`, `{execution}`, `{issue}`

### `agent/graph.py`

Three functions to implement:

**`verify_node`** — call LLM with VERIFY prompts; parse JSON defensively (handle markdown fences and prose wrapping); return `{"verify_ok": bool, "verify_issue": str}`.

**`revise_node`** — call LLM with REVISE prompts (pass prior SQL + execution result + verifier issue); extract SQL with `_extract_sql`; bump `iteration` counter; return same shape as `generate_sql_node`.

**`route_after_verify`** — return `"end"` if `state.verify_ok or state.iteration >= MAX_ITERATIONS`, else `"revise"`.

Test by running `uvicorn agent.server:app --port 8001` and curling with questions from `evals/eval_set.jsonl`. Confirm at least one question triggers a revise cycle.

**Deliverables:** Agent server running at :8001, `verify → revise` loop wired and capped.

---

## Phase 4 — Langfuse Tracing (local)

`server.py` already conditionally initialises `CallbackHandler` when env vars are set.

1. Open http://localhost:3001, sign up, create a project
2. Copy public + secret keys into `.env` (`LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`)
3. Restart the agent server (it picks up env vars on startup)
4. Add metadata tags to the `config` dict in `server.py` (e.g. `db`, `model`) for Phase 6 filtering
5. Fire 10 questions, inspect the waterfall trace in Langfuse UI

**Deliverables:** `screenshots/langfuse_trace.png`, `screenshots/langfuse_tags.png`.

---

## Phase 5 — Eval Runner

**File to implement:** `evals/run_eval.py` — two functions.

### `eval_one(question, agent_url)`

1. Run gold SQL → get gold rows
2. POST to agent → get `history` (list of per-iteration dicts with `sql`) + final `sql` + `iterations`
3. For each entry in `history`: execute that SQL, compare canonicalized rows against gold → `correct_at[i]`
4. Execute final `sql` → `final_correct`
5. Return `{question, db_id, gold_sql, agent_sql, iterations, correct_at, final_correct, latency_seconds}`

### `summarize(results)`

1. Find `max_iter` across all results
2. For `k` in 0..max_iter: apply carry-forward — if agent stopped at `j < k`, treat iteration-k correctness as identical to iteration-j
3. Return `{n, overall_pass_rate, per_iteration_pass_rate: {"0": x, "1": y, ...}}`

Test the harness end-to-end against Nebius AI Studio first. Run the real baseline against H100 vLLM; save to `results/eval_baseline.json`.

**Deliverables:** `results/eval_baseline.json`, `screenshots/grafana_eval_run.png`.

---

## Phase 6 — SLO Optimization (Nebius VM)

**Target: P95 end-to-end agent latency < 5 s, 10+ RPS sustained over 5 minutes.**

```bash
uv run python load_test/driver.py --rps 10 --duration 300
```

For each iteration: observe Grafana → form one hypothesis → change one flag → re-run → record in `REPORT.md`:

> *"saw X → hypothesized Y → changed Z → result W"*

Common levers to reach for (in rough order of impact for this workload):
- `--max-num-seqs` — controls queue depth vs. latency tradeoff
- `--max-model-len` — smaller = less KV pressure, more headroom for concurrent requests
- Prefix caching on/off — schemas should make this a clear win; confirm with KV cache panel
- `--enable-chunked-prefill` — helps when prompt lengths are mixed

After final config: re-run evals → `results/eval_after_tuning.json`. Check quality survived.

**Deliverables:** `screenshots/grafana_before.png`, `screenshots/grafana_after.png`, `results/eval_after_tuning.json`, iteration log in `REPORT.md`.

---

## Phase 7 — REPORT.md (≤ 3 pages)

Sections:
1. **Serving config** — flags table + one-line rationale each
2. **Baseline eval** — overall pass rate, per-iteration pass rate, does the loop earn its keep?
3. **SLO iteration log** — baseline numbers, each iteration, final numbers
4. **Agent value** — one paragraph citing per-iteration pass rate as evidence
5. **What I'd do with more time** — specific (not "add Kubernetes")

---

## Final deliverables checklist

| File | Phase |
|---|---|
| `REPORT.md` | 7 |
| `infra/grafana/provisioning/dashboards/serving.json` | 2 |
| `agent/graph.py` | 3 |
| `agent/prompts.py` | 3 |
| `evals/run_eval.py` | 5 |
| `results/eval_baseline.json` | 5 |
| `results/eval_after_tuning.json` | 6 |
| `scripts/start_vllm.sh` | 1 |
| `screenshots/vllm_manual_query.png` | 1 |
| `screenshots/grafana_serving.png` | 2 |
| `screenshots/langfuse_trace.png` | 4 |
| `screenshots/langfuse_tags.png` | 4 |
| `screenshots/grafana_eval_run.png` | 5 |
| `screenshots/grafana_before.png` | 6 |
| `screenshots/grafana_after.png` | 6 |
