# L-GAR-CPT: LSD-Guided Global Attention Router for Qwen2.5-0.5B Continued Pretraining

## 0. One-line summary

Train a Qwen2.5-0.5B long-context CPT model with an LSD-supervised router that predicts which query positions need global context. The method uses ordinary CE for language modeling and uses LongCE-style long-short difference only as a routing signal, not as CE loss reweighting.

---

## 1. Core hypothesis

Only a minority of next-token predictions truly benefit from remote context. Instead of spending full global attention on every query position, train a router to identify high-LSD positions and allocate global attention primarily to those positions.

For target token `x_i`:

```text
long_logp_i  = log p_teacher(x_i | full prefix)
short_logp_i = log p_teacher(x_i | last short_window tokens)
LSD_i        = long_logp_i - short_logp_i
```

A query position `i-1` is a global-attention candidate when the next token `x_i` has high LSD and is not just a random hard/noisy token.

Main claim to test:

```text
LSD-supervised global query routing improves long-context CPT efficiency and long-context downstream ability while preserving normal CE and short-task performance.
```

---

## 2. Fixed experimental setup

### Base model

```text
Base checkpoint: Qwen/Qwen2.5-0.5B base
Tokenizer: Qwen2.5 tokenizer
Training type: continued pretraining on raw text
Precision: bf16
Attention: FlashAttention-2 if available
Gradient checkpointing: on
```

### Data

Use FineWebEdu-shuffled raw documents, retokenized with the Qwen tokenizer.

Required metadata:

```text
doc_id
doc_token_start
doc_token_end
intra_doc_offset
is_special_token
is_padding
```

Packing is allowed, but all LSD labels and long-context evaluations must respect document boundaries. Do not treat tokens from another packed document as remote evidence.

### Context and windows

```text
Stage1 context_len: 8192
Stage2 context_len: 8192 by default; 16384 only if stable
short_window: 1024
local_window: 1024 or 2048
router target fraction: 5% to 10% positive labels
initial global query budget: 100%
final global query budget: 25% by default; do not go below 20% in Stage1
```

---

## 3. Method definition

### 3.1 LSD label miner

Use a frozen teacher or no-grad full-attention teacher pass to compute labels.

For every valid target token `x_i`:

```text
long_nll_i  = -log p_teacher(x_i | full prefix)
short_nll_i = -log p_teacher(x_i | last short_window tokens)
LSD_i       = short_nll_i - long_nll_i
```

This is equivalent to:

```text
LSD_i = log p_long - log p_short
```

A positive label for query position `i-1` requires:

```text
same_document_prefix_len >= short_window + 256
target is not padding / special / isolated EOS
LSD_i >= batch_or_shard_percentile(90 or 95)
long_nll_i <= batch_or_shard_percentile(70)
```

The long-NLL filter is important: high LSD alone can include noisy tokens where both long and short predictions are bad. We want tokens where long context genuinely helps.

Log these statistics every eval interval:

```text
lsd_mean
lsd_p50 / p90 / p95 / p99
positive_label_fraction
long_nll_positive_mean
short_nll_positive_mean
labels_per_sequence
labels_per_document
```

### 3.2 Router

For selected upper layers, add a small router:

```text
r_l,t = sigmoid(MLP_l(RMSNorm(h_l,t)))
```

Options:

```text
shared_router_across_upper_layers = true by default
router_hidden_dim = min(512, d_model)
router_input = layer input hidden state before attention
```

Router supervision:

```text
L_router = BCE(r_l,t, y_key_t)
```

Budget regularization:

```text
actual_budget_l = mean(topk_or_soft_selection_l)
L_budget = (actual_budget_l - target_budget)^2
```

Optional entropy regularization:

```text
L_entropy = -mean(r * log r + (1-r) * log(1-r))
```

Use a small entropy term only if router collapses to all-zero or all-one.

### 3.3 RouterAux mode

RouterAux mode keeps full attention unchanged and trains only:

```text
L = L_CE + lambda_router * L_router + lambda_budget * L_budget
```

This mode tests whether the LSD signal is learnable from model states without changing the attention pattern.

### 3.4 L-GAR mode: progressive local/global attention

L-GAR changes only selected upper layers.

Recommended default:

```text
num_layers_total: use model config
routed_layers: top 1/3 of layers
lower layers: unchanged full attention
upper routed layers: local/global query routing
```

For a routed layer:

```text
if query position t is selected as global:
    it can attend to the full prefix within the same document/sequence mask
else:
    it can attend only to the local_window prefix
```

Selection:

```text
global_queries = topk(router_score, target_budget_fraction)
```

Use straight-through top-k for masking, while BCE trains the router. Keep the router loss even when the top-k decision is non-differentiable.

Progressive schedule:

```text
0M-10M tokens:
  full attention everywhere
  train router only

10M-30M tokens:
  keep full attention or use very high global budget, e.g. 75%-100%
  monitor router AUC / precision@budget

30M+ tokens:
  apply routing in upper layers
  target_budget schedule: 75% -> 50% -> 25%
```

If the implementation cannot support dynamic per-query masks efficiently, implement RouterAux only and report that hard L-GAR is not available in this environment. Do not fake sparse attention with unrelated dropout.

### 3.5 Final objective

For RouterAux:

```text
L = L_CE
  + lambda_router * L_router
  + lambda_budget * L_budget
```

For L-GAR:

```text
L = L_CE_routed
  + lambda_router * L_router
  + lambda_budget * L_budget
  + optional lambda_kl * KL(full_teacher_logits || routed_student_logits)
```

Recommended initial values:

```text
lambda_router: 0 -> 0.02 ramp over first 10M tokens
lambda_budget: 0.005
lambda_kl: 0.0 by default; 0.02 only if routed CE becomes unstable
```

Do not apply LongCE CE weighting inside the L-GAR method run.

---

## 4. Stage 0: implementation and review checklist

Before Stage1, complete all checks below.

### 4.1 Dataset checks

- [ ] Qwen tokenizer is used.
- [ ] Packed sequences preserve document boundary metadata.
- [ ] LSD labels never cross document boundaries.
- [ ] Padding, special tokens, and isolated EOS targets are excluded.
- [ ] A printed audit shows `doc_id`, target text, short-window text, and remote prefix span.

### 4.2 Label miner checks

Run on at least 10k sequences without training.

Required outputs:

```text
positive_label_fraction
lsd_p50 / p90 / p95 / p99
mean long_nll on positive labels
mean short_nll on positive labels
number of labels per batch
number of documents with labels
```

Manual audit for 20 examples:

```text
target token / target text
full-prefix excerpt
short-window excerpt
long_nll
short_nll
LSD
label yes/no
```

### 4.3 Router checks

Run 100 optimizer steps in RouterAux mode.

Pass requirements:

```text
no NaN
router loss decreases or is stable
positive labels are nonzero
router AUC > 0.55 after warmup, or precision@budget > random baseline
CE does not spike by more than 0.05 relative to CE-CPT smoke run
```

### 4.4 Routed attention checks

For hard L-GAR mode:

- [ ] Mask shape is correct for batch, heads, query length, key length.
- [ ] Causal mask is preserved.
- [ ] Query positions cannot attend to future tokens.
- [ ] Query positions cannot attend across packed document boundaries if boundary masks are used.
- [ ] Low-router-score queries see only local_window keys.
- [ ] High-router-score queries see full allowed prefix.
- [ ] Actual global budget matches target budget within ±2%.
- [ ] Routed layers are only the intended upper layers.
- [ ] Full attention mode and 100% budget mode numerically match baseline within small tolerance.

### 4.5 Smoke runs

Run 500-1000 update steps for:

```text
CE-CPT smoke
RouterAux-CPT smoke
L-GAR-CPT smoke with high budget
```

Save plots:

```text
train CE
val CE
router BCE
router AUC / precision@budget
actual global budget
high-LSD CE
throughput including label mining
```

---

## 5. Stage 1: four-run CPT screen

All four runs start from the same Qwen2.5-0.5B base checkpoint.

### Stage1 budget

```text
Tokens per run: 150M CPT tokens
Context length: 8192
Global batch tokens: choose a stable value, e.g. 128K-256K
Minimum optimizer updates: target >= 800 if possible
Optimizer: AdamW
LR: 1e-5 default; allow 5e-6 if unstable
Min LR: 1e-6
Warmup: 3%
Weight decay: 0.1
Grad clip: 1.0
Precision: bf16
```

### Runs

| Run | Name | Objective | Attention |
|---|---|---|---|
| A | `Qwen_CE_CPT` | standard CE | full attention |
| B | `Qwen_LongCE_CPT` | LongCE-style CE weighting | full attention |
| C | `Qwen_RouterAux_CPT` | CE + LSD-router auxiliary | full attention |
| D | `Qwen_LGAR_CPT` | CE + LSD-router auxiliary + progressive routed upper-layer attention | local/global routed attention |

### LongCE baseline settings

Use safe clipped weights:

```text
weight_i = clip(exp(LSD_i), min=0.25, max=4.0)
normalize weights to mean 1.0 within batch
```

LongCE is a related baseline. The main baseline is CE-CPT.

### L-GAR run settings

```text
routed_layers: top 1/3 layers
local_window: 1024 first; 2048 if 1024 is too harmful
target_budget final: 25%
first 10M tokens: router-only full attention
next 20M tokens: high budget 75%-100%
remaining: progressive to 25%-50%
```

If `Qwen_LGAR_CPT` is unstable, rerun one fix with:

```text
local_window = 2048
final_budget = 50%
routed_layers = top 1/4 layers
lambda_router = 0.01
```

Do not run more than one fix without writing a review note.

---

## 6. Stage 1 evaluation

Evaluate every run on the same checkpoints and token budgets.

### Required metrics

```text
normal validation CE
normal validation PPL
high-LSD-token CE
Long 1024 target CE
Long 4096 target CE
RULER-mini 4K
RULER-mini 8K if available
NoLiMa or NoLiMa-style long dependency
needle/key-value retrieval as secondary only
short MC average accuracy
router AUC / precision@budget for router runs
actual global budget by layer
train tokens/sec excluding eval
true wall-clock tokens/sec including label mining and routing overhead
```

### Router-specific metrics

```text
router_auc
router_precision_at_5pct
router_precision_at_10pct
router_recall_at_budget
actual_budget_per_layer
positive_label_fraction
mean LSD of routed tokens
mean LSD of non-routed tokens
```

### Plots

Save:

```text
train CE vs tokens
val CE vs tokens
high-LSD CE vs tokens
router AUC vs tokens
actual global budget vs tokens
RULER/NoLiMa summary table
throughput table
```

---

## 7. Stage 1 pass/fail rule

L-GAR-CPT passes only if all safety criteria and at least one strong long-context criterion are met.

### Safety criteria

```text
normal_val_CE(LGAR) <= normal_val_CE(CE_CPT) + 0.02
short_MC_avg(LGAR) >= short_MC_avg(CE_CPT) - 1.0 absolute point
no persistent loss spike after routing begins
true wall-clock overhead is reported
```

### Long-context criteria

At least two of the following must hold:

```text
high-LSD CE improves over CE-CPT
Long 4096 target CE improves over CE-CPT
RULER-mini average improves over CE-CPT
NoLiMa-style score improves over CE-CPT
LGAR beats RouterAux on at least one important long-context metric
```

### Router criteria

```text
router precision@budget > random baseline by a clear margin
routed tokens have higher mean LSD than non-routed tokens
actual global budget matches target budget
```

### Non-sufficient criteria

The following are not enough by themselves:

```text
beating LongCE-CPT only
router loss decreasing only
local/global attention being faster only
needle retrieval improving while high-LSD CE and RULER do not improve
```

---

## 8. Stage 2: two-run CPT scale-up

Only run Stage2 if Stage1 passes.

### Stage2 runs

| Run | Name | Setup |
|---|---|---|
| A | `Qwen0.5B_CE_CPT_2B` | Qwen2.5-0.5B base + standard CE CPT |
| B | `Qwen0.5B_LGAR_CPT_2B` | Qwen2.5-0.5B base + best L-GAR-CPT variant from Stage1 |

### Stage2 budget

```text
Tokens per run: 2B CPT tokens
Context length: 8192 default; 16384 only if Stage1 showed stable 16K behavior
Same data order for both runs
Same tokenizer
Same optimizer family
Same LR schedule
Same eval schedule
```

### Stage2 success rule

```text
normal_val_CE(method) <= normal_val_CE(CE_CPT) + 0.01 to 0.02
high-LSD CE improves over CE-CPT
RULER/NoLiMa average improves over CE-CPT
short MC does not drop >1 absolute point
true wall-clock overhead is acceptable and reported
method is not merely a LongCE improvement; it must beat CE-CPT on long-context metrics
```

---

## 9. Common bugs to catch

- [ ] Label off-by-one: router at position `i-1` should predict whether target token `x_i` needs long context.
- [ ] Using target token itself in the short/long prefix.
- [ ] Computing LSD across packed document boundaries.
- [ ] Treating high-NLL noisy tokens as positive high-LSD labels.
- [ ] Forgetting to normalize LongCE baseline weights to mean 1.
- [ ] Applying routed attention in all layers too early.
- [ ] Breaking the causal mask when constructing local/global masks.
- [ ] Reporting only step tokens/sec while ignoring label mining/routing overhead.
- [ ] Comparing L-GAR to LongCE only; CE-CPT is the real baseline.
- [ ] Letting the router choose all tokens or no tokens due to missing budget control.

---

## 10. Suggested command templates

Adapt paths to the local training codebase.

```bash
# Stage 0 label audit
python train.py \
  --config configs/qwen05b_cpt.yaml \
  --method lgar \
  --stage label_audit \
  --base_model Qwen/Qwen2.5-0.5B \
  --data finewebedu_qwen_docaware \
  --context_len 8192 \
  --short_window 1024 \
  --output_dir runs/lgar_cpt/label_audit

# Stage 1 CE-CPT
python train.py --config configs/qwen05b_cpt.yaml \
  --method ce_cpt \
  --max_tokens 150000000 \
  --output_dir runs/lgar_cpt/Qwen_CE_CPT

# Stage 1 RouterAux-CPT
python train.py --config configs/qwen05b_cpt.yaml \
  --method lgar_router_aux \
  --max_tokens 150000000 \
  --output_dir runs/lgar_cpt/Qwen_RouterAux_CPT

# Stage 1 L-GAR-CPT
python train.py --config configs/qwen05b_cpt.yaml \
  --method lgar_routed \
  --max_tokens 150000000 \
  --context_len 8192 \
  --short_window 1024 \
  --local_window 1024 \
  --final_global_budget 0.25 \
  --output_dir runs/lgar_cpt/Qwen_LGAR_CPT
```

---

## 11. Final report template

The final Stage1 report must include:

```text
1. Exact base checkpoint and tokenizer
2. Dataset path and document-boundary handling
3. Number of CPT tokens per run
4. Normal CE table
5. High-LSD CE table
6. RULER/NoLiMa table
7. Short MC table
8. Router quality table
9. True wall-clock throughput table
10. Pass/fail decision
11. If fail: whether failure is label quality, routing instability, CE safety, or downstream non-transfer
12. If pass: exact Stage2 config to launch
```
