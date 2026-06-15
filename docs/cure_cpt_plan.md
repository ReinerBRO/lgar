# CURE-CPT Implementation Plan

Counterfactual Utility Retrieval-Head Enhancement for Qwen2.5-0.5B Continued Pretraining

Date: 2026-05-28
Base model: Qwen2.5-0.5B (24 layers, 14 Q heads, 2 KV heads per layer, hidden_size=896)
Note: read model.config at runtime; do not hardcode head counts.
Data: FineWebEdu doc-aware, Qwen tokenizer
Remote root: `/gemini/space/private/zjc/goals/lgar`

---

## 1. Motivation

LGAR learns a query-level router to decide which positions get global attention. This is access control — it limits who can see remote context, but does not make the model better at using it. CURE-CPT instead identifies the attention heads that causally carry long-context retrieval and concentrates training signal on them.

Key insight: long-context failures concentrate at a few high-utility positions, and the retrieval work is done by a small set of heads that already exist in the pretrained model. CURE finds these heads via counterfactual ablation and trains them with dedicated losses.

---

## 2. Architecture Overview

```
Qwen2.5-0.5B backbone (frozen or CE-finetuned)
    |
    +-- All heads: normal dense attention, full causal mask
    |
    +-- Selected retrieval heads: additional LoRA adapters (rank=8, zero-init)
        |
        +-- L_CE:      standard next-token CE, all params
        +-- L_RH-CE:   extra CE on high-utility queries, grad -> retrieval adapters only
        +-- L_RH-KD:   KL(full || RH-only), grad -> retrieval adapters only
        +-- L_cov:     coverage loss for multi-evidence, grad -> retrieval adapters only
```

---

## 3. Reusable Components from LGAR

| LGAR module | Reuse in CURE | Changes needed |
|---|---|---|
| `mining.py:compute_long_short_logp` | Stage A: query utility | Same logic: `u_q = NLL_local - NLL_full` |
| `mining.py:build_lsd_label_batch_from_scores` | Stage A: high-utility filtering | Rename to query_utility, same percentile+NLL filter |
| `mining.py:mine_lsd_labels` | Stage A entry point | Rename output to query_utility |
| `mining.py:document_causal_attention_mask` | Stage B ablation masks | Reuse directly |
| `mining.py:local_document_attention_mask` | Stage A local-view, Stage B local-only baseline | Reuse directly |
| `mining.py:lgar_routed_attention_mask` | Stage B: per-head remote ablation mask | Need new per-head mask function |
| `mining.py:select_topk_global_queries` | Stage A: utility filtering | Reuse for identifying high-utility positions |
| `mining.py:lsd_audit_examples` | Stage A/B audit | Reuse directly |
| `config.py:LGARParams` | CURE config | Extend with CURE-specific params |
| `config.py:routed_layer_indices` | Stage B: candidate head selection | Reuse for upper-layer targeting |
| `data.py:all` | Data pipeline | Reuse directly |
| `modeling.py:all` | Model loading | Reuse directly |
| `hard_lgar.py` | Not reused | CURE uses standard Qwen forward |
| `router.py` | Not reused | CURE has no query router |
| `sharding.py` | Stage B sharding | Reuse for ablation computation distribution |
| `train_stage0.py:distributed setup` | Training harness | Reuse patterns |
| `evaluate.py` | Evaluation | Reuse directly |

---

## 4. New Components

### 4.1 `curcpt/config.py` — CURE parameters

```python
@dataclass(frozen=True)
class CUREParams:
    # Stage A: utility mining
    seq_len: int = 8192
    local_window: int = 1024
    min_remote_margin: int = 256
    utility_top_fraction_ablation: float = 0.05   # top 5% for head ablation
    utility_top_fraction_training: float = 0.10    # top 10% for training
    full_nll_max_quantile: float = 0.80

    # Stage B: head ablation
    ablation_candidates: str = "upper_third"       # which layers to probe
    ablation_top_k_fraction: float = 0.05          # top 5% of candidate heads
    ablation_min_delta: float = 0.01               # minimum NLL drop to qualify
    ablation_batch_size: int = 4
    ablation_calibration_sequences: int = 512      # per half; 1024 total if sparse
    ablation_split_half: bool = True               # require stability across both halves

    # Stage C: training
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lambda_rh_ce: float = 0.1
    lambda_rh_kd: float = 0.05
    lambda_cov: float = 0.005
    cov_warmup_tokens: int = 10_000_000

    # Training
    context_len: int = 8192
    tokens_per_run: int = 200_000_000
    lr: float = 1.0e-5
    min_lr: float = 1.0e-6
    warmup_fraction: float = 0.03
    weight_decay: float = 0.1
    grad_clip: float = 1.0
```

### 4.2 `curcpt/head_ablation.py` — Stage B core

New module. Head unit is **(layer_id, q_head_id)** — not KV head.

For each candidate head (layer, q_head):
1. Run full-attention forward, get baseline NLL at high-utility positions
2. Run modified forward where this Q head can only attend local_window
3. Compute delta = NLL_ablated - NLL_full per position
4. Aggregate across calibration set
5. Rank heads by mean delta, select top-k

Calibration set: 512 sequences from held-out FineWebEdu doc-aware split.
Same distribution as CPT training, not RULER/NoLiMa.
Split into two halves; only select heads with stable positive ablation scores in both halves.
Increase to 1024 sequences if high-utility positions are sparse.

Key function:
```python
def ablate_head_remote_attention(
    model, input_ids, doc_ids, high_utility_mask,
    layer_idx, q_head_id, local_window,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Mask remote attention for one Q head, return per-position NLL delta."""
```

Implementation approach:
- Patch the attention mask for the specific Q head at the specific layer
- Qwen2.5-0.5B has 14 Q heads and 2 KV heads (read from model.config, do not hardcode)
- Each Q head maps to a KV head via GQA grouping; ablation operates at Q head level
- Construct per-head mask: broadcast [B, 1, T, T] for the target head, local-only for keys beyond local_window

### 4.3 `curcpt/adapters.py` — Head-specific LoRA

```python
class HeadSliceLoRA(nn.Module):
    """LoRA adapter for a single Q head's slice of q_proj or o_proj.

    Operates on head_dim slice, not full projection.
    """
    def __init__(self, head_dim: int, rank: int = 8, alpha: float = 16.0):
        self.lora_A = nn.Parameter(torch.zeros(rank, head_dim))
        self.lora_B = nn.Parameter(torch.zeros(head_dim, rank))
        self.scaling = alpha / rank

    def forward(self, x_slice: torch.Tensor) -> torch.Tensor:
        # x_slice: [..., head_dim]
        return (x_slice @ self.lora_A.T @ self.lora_B.T) * self.scaling
```

Apply to selected retrieval heads' **q_proj** and **o_proj** slices only.
Do NOT modify k_proj/v_proj in v1 — GQA KV heads are shared across multiple Q heads and would confound the mechanism.
Zero-init so training starts from identity (no-op).
Optional later ablation: add group-shared K/V LoRA for the selected Q head's KV group.

### 4.4 `curcpt/losses.py` — CURE loss functions

```python
def cure_loss(
    logits_full: torch.Tensor,        # [B, T, V] from normal forward
    logits_rh_only: torch.Tensor,     # [B, T, V] from RH-bottleneck forward
    labels: torch.Tensor,             # [B, T]
    high_utility_mask: torch.Tensor,  # [B, T] bool
    valid_mask: torch.Tensor,         # [B, T] bool
    params: CUREParams,
    step_tokens: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute CURE-CPT total loss."""
    # L_CE: all tokens, all params
    l_ce = cross_entropy(logits_full, labels, valid_mask)

    # L_RH-CE: high-utility tokens, retrieval-head adapters only
    l_rh_ce = cross_entropy(logits_full, labels, high_utility_mask & valid_mask)

    # L_RH-KD: KL(full || RH-only) on high-utility tokens
    l_rh_kd = kl_divergence(logits_full, logits_rh_only, high_utility_mask & valid_mask)

    # L_cov: coverage (optional, warmup)
    l_cov = coverage_loss(...) if step_tokens > warmup else 0.0

    total = l_ce + lambda_rh_ce * l_rh_ce + lambda_rh_kd * l_rh_kd + lambda_cov * l_cov
    return total, metrics
```

### 4.5 `curcpt/forward.py` — CURE forward pass

Two forward paths:
1. **Full forward**: standard Qwen forward, all heads attend full context. Produces `logits_full`.
2. **RH-bottleneck forward**: retrieval heads get full attention, non-retrieval heads get local-only attention. Produces `logits_rh_only`.

```python
def cure_forward(
    model, input_ids, attention_mask, doc_ids,
    retrieval_heads: set[tuple[int, int]],  # {(layer_id, q_head_id), ...}
    local_window: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (logits_full, logits_rh_only)."""
```

### 4.6 `curcpt/train.py` — CURE training loop

```
for step in dataloader:
    1. Load batch (reuse PackedSequenceSignalDataset or compute on-the-fly)
    2. Identify high-utility positions (from offline cache or online mining)
    3. Forward pass: get logits_full and logits_rh_only
    4. Compute cure_loss
    5. Gradient rules:
       - L_CE grad -> all trainable params (base model or base LoRA)
       - L_RH-CE, L_RH-KD, L_cov grad -> retrieval-head adapters only
    6. Optimizer step, logging
```

Gradient isolation via `torch.autograd.grad` or `retain_graph` + selective `.backward()`.

### 4.7 `curcpt/run_head_ablation.py` — Stage B CLI

```
python -m curcpt.run_head_ablation \
    --model-path ... \
    --checkpoint-path ... \        # CE-CPT checkpoint or base
    --cache-dir ... \
    --signal-dir ... \             # Stage A offline query_utility cache
    --output-dir ... \
    --ablation-sequences 512 \     # per half (1024 total)
    --top-k-fraction 0.05 \
    --local-window 1024 \
    --seq-len 8192
```

Outputs `ablation_results.json`:
```json
{
  "retrieval_heads": [[18, 7], [19, 3], [20, 11], ...],
  "head_scores": {"18_7": 0.045, "19_3": 0.038, ...},
  "head_scores_half1": {"18_7": 0.042, ...},
  "head_scores_half2": {"18_7": 0.048, ...},
  "num_selected": 12,
  "calibration_sequences_per_half": 512,
  "split_half_stable": true
}
```

---

## 5. Implementation Phases

All phases use real Qwen2.5-0.5B and real FineWebEdu data. No synthetic/toy experiments.

### Phase 0: Infrastructure (1-2 days)

- [ ] Create `curcpt/` package with `__init__.py`
- [ ] `curcpt/config.py`: CUREParams dataclass
- [ ] `curcpt/forward.py`: dual forward path (full + RH-bottleneck)
- [ ] `curcpt/adapters.py`: HeadLoRA module + injection/removal helpers
- [ ] Unit tests: adapter zero-init produces identity, dual forward matches baseline
- [ ] 100-step real-data smoke test: no NaN, loss decreases, gradient isolation correct

### Phase 1: Stage A — Utility Mining (0.5 day)

- [ ] `curcpt/utility_mining.py`: wrap existing `mining.py` functions, rename signal to `query_utility`
- [ ] Store both `query_pos` and `target_pos` to avoid off-by-one bugs
- [ ] `u_q = NLL_local(x_{q+1}) - NLL_full(x_{q+1})` — same computation, explicit naming
- [ ] Verify: utility labels match LGAR LSD labels (same signal, new name)
- [ ] Run offline utility mining on CE-CPT checkpoint (reuse current 8-GPU mahiro job)

### Phase 2: Stage B — Head Ablation (2-3 days)

- [ ] `curcpt/head_ablation.py`: per-Q-head remote-attention masking
- [ ] Head unit: (layer_id, q_head_id), read head geometry from model.config
- [ ] Calibration: 512 sequences per half from held-out FineWebEdu split (same distribution as CPT)
- [ ] Split-half stability: only select heads positive in both halves
- [ ] `curcpt/run_head_ablation.py`: CLI for calibration run
- [ ] Run ablation on mahiro
- [ ] Analyze: layer distribution, head stability across halves
- [ ] Output `ablation_results.json` with selected retrieval heads

### Phase 3: Stage C — CURE Training (2-3 days)

- [ ] `curcpt/losses.py`: L_CE, L_RH-CE, L_RH-KD, L_cov
- [ ] `curcpt/train.py`: training loop with gradient isolation
- [ ] Gradient routing: ensure L_RH-CE/KD/cov only update adapters
- [ ] Smoke run: 100 steps on real data, verify no NaN, loss decreases, gradient isolation correct
- [ ] Pilot run: 50M-100M tokens, seq_len=4K, 4 groups (CE-CPT, Adapter-CE, LongCE-CPT, CURE-CPT)

### Phase 4: Evaluation (1-2 days)

- [ ] RULER13 4K/8K
- [ ] NoLiMa-style evaluation
- [ ] high-utility CE, normal val CE, short MC
- [ ] Post-training head ablation sensitivity
- [ ] Attention coverage analysis on multi-evidence examples

---

## 6. Experiment Design (Stage1)

### 6.0 Non-negotiable rules

Do not run toy experiments. Only run real-data smoke tests and real-model pilot experiments.
Use Qwen2.5-0.5B base and Qwen-tokenized FineWebEdu doc-aware sequences from the beginning.
The minimum allowed debug is a 100-step smoke test on real data, not a synthetic toy task.
Do not report toy results as evidence. Toy results are not useful for deciding whether CURE works.

### 6.1 Smoke test (100 steps, real data)

Purpose: engineering validation only, not evidence.

Checks:
- [ ] Forward/backward no NaN
- [ ] Selected head LoRA mask correct (only target Q/O slices receive adapter updates)
- [ ] Utility cache alignment correct (query_pos vs target_pos)
- [ ] Head ablation score nonzero for at least one head
- [ ] Gradient isolation: L_RH-CE/KD/cov do not update non-adapter params

### 6.2 Pilot runs (50M-100M tokens, seq_len=4K)

```
Base:              Qwen2.5-0.5B base
Data:              FineWebEdu doc-aware, Qwen tokenizer
Seq len:           4096
Calibration:       512 held-out real sequences (same distribution)
Train budget:      50M-100M tokens per run
```

| Run | Name | Setup | Purpose |
|---|---|---|---|
| A | `CE_CPT` | Standard CE, full attention | Baseline |
| B | `Adapter_CE` | Same retrieval-head adapter params, standard CE only | Param-matched baseline |
| C | `LongCE_CPT` | LongCE scalar reweight, full attention | Related baseline |
| D | `CURE_CPT` | Full CURE method | Main method |

Optional (if compute allows):
| E | `CURE_no_KD` | CURE without L_RH-KD | Ablation |
| F | `CURE_no_cov` | CURE without L_cov | Ablation |

### 6.3 Budget

```
Tokens per run:    50M-100M (pilot); 200M+ only after pilot passes
Context length:    4096 (pilot); 8192 for full Stage1
Batch tokens:      128K-256K
Optimizer:         AdamW
LR:                1e-5
Min LR:            1e-6
Warmup:            3%
Weight decay:      0.1
Grad clip:         1.0
Precision:         bf16
```

### 6.4 Evaluation

```
RULER13 2K / 4K
NoLiMa-style long dependency
high-utility CE
normal validation CE
short MC accuracy
post-training head ablation sensitivity
```

### 6.5 Pass/Fail Criteria

Pass requires ALL safety criteria AND at least 3 long-context criteria.

**Safety:**
1. `normal_val_CE(CURE) <= normal_val_CE(CE_CPT) + 0.02`
2. `short_MC(CURE) >= short_MC(CE_CPT) - 0.5 absolute`
3. No persistent loss spike
4. `CURE > Adapter-CE` on at least one long-context metric

**Long-context (need >= 3):**
1. RULER13 2K/4K avg > CE-CPT by >= 1.0 absolute
2. multikey/multivalue/multiquery improve over CE-CPT
3. NoLiMa improves over CE-CPT
4. high-utility CE improves over CE-CPT
5. CURE > LongCE-CPT on at least one important metric

**Retrieval-head criteria:**
1. Post-training ablation shows selected heads became more causally important
2. Retrieval heads' attention mass on high-utility positions increased vs pre-training

---

## 7. File Structure

```
curcpt/
    __init__.py
    config.py           # CUREParams
    forward.py          # cure_forward: dual-path forward
    adapters.py         # HeadSliceLoRA, inject/remove
    head_ablation.py    # Stage B: per-Q-head ablation with split-half
    losses.py           # L_CE, L_RH-CE, L_RH-KD, L_cov
    utility_mining.py   # Stage A: query_utility, wrap lgar_cpt.mining
    train.py            # Stage C: CURE training loop
    run_head_ablation.py # Stage B CLI
    evaluate.py         # Reuse lgar_cpt.evaluate patterns

scripts/
    run_cure_stage_a.sh     # Offline query_utility mining (reuse signal cache)
    run_cure_stage_b.sh     # Head ablation on mahiro
    run_cure_smoke.sh       # 100-step real-data smoke test
    run_cure_pilot.sh       # 50M-100M token pilot, 4 runs

specs/
    03_CURE_CPT_Qwen0.5B.md  # Full spec (to write)

tests/
    test_cure_adapters.py
    test_cure_ablation.py
    test_cure_losses.py
    test_cure_forward.py
```

---

## 8. GQA Consideration

Qwen2.5-0.5B has 14 Q heads and 2 KV heads per layer (GQA ratio 7:1).
Read `model.config.num_attention_heads` and `model.config.num_key_value_heads` at runtime; do not hardcode.

Head unit for ablation and training: **(layer_id, q_head_id)**.

Ablation operates at Q head level:
- For each candidate (layer, q_head), construct a mask that restricts that specific Q head to local-window keys
- The corresponding KV head is shared with other Q heads, so the ablation mask must be applied at the attention weight level after the GQA expansion
- 14 candidates per upper layer, ~112 candidates for top 1/3 layers (8 layers * 14 Q heads)

LoRA adapters: q_proj and o_proj slices for the selected Q head only.
Do not touch k_proj/v_proj in v1 since KV heads are shared across 7 Q heads.

---

## 9. Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Head ablation noisy at 512 sequences | Unreliable head selection | Split-half stability filter; increase to 1024 if sparse |
| Retrieval heads not stable across data | Selected heads don't generalize | Use CE-CPT checkpoint for ablation (better features than base) |
| RH-bottleneck forward is 2x compute | Training throughput drops | Only compute RH-bottleneck on high-utility positions (sparse) |
| Gradient isolation is tricky | L_CE leaks into adapters or L_RH-CE updates base | Use `torch.autograd.grad` with explicit `inputs=` for each loss |
| Coverage loss too complex for v1 | Delayed start | Make L_cov optional, warmup after 10M tokens |
| Adapter-CE already strong | CURE advantage unclear | CURE must beat Adapter-CE to pass |
| Q-head ablation mask applied after GQA expansion | Implementation complexity in Qwen attention | Hook into attention weights post-expansion, not raw Q/K/V |
