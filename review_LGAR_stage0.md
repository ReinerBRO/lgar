# L-GAR Stage 0 Implementation Review

Date: 2026-05-24
Remote root: `/gemini/space/private/zjc/goals/lgar`
Environment: `/gemini/space/private/zjc/envs/zjc_env`
Compute: `yui`

Status: PASS for official Stage 0 audit launch. Stage 1 is allowed only after the official Stage 0 summary passes all checks.

## Mechanism Review

- LSD labels are assigned to query position `i-1` for target token `x_i`.
- Long and short teacher passes both use document-aware causal masks.
- Short pass uses `short_window`; packed documents cannot attend across boundaries.
- Positive labels require remote same-document prefix length, positive high LSD, and long-NLL filtering.
- Padding, BOS/EOS/PAD special tokens are excluded from positive labels.
- RouterAux objective is implemented as `CE + lambda_router * BCE + lambda_budget * budget`.
- Router input uses selected upper-layer hidden states before the corresponding layer.
- Local/global routed mask construction is tested.
- Hard L-GAR uses a custom Qwen forward path that injects routed masks only into selected upper layers.

## Fixes Made During Review

- Found that `sdpa` with required 4D document-aware additive masks produced NaN CE in real Qwen smoke.
- Changed Stage 0 default attention implementation to `eager` and documented the reason.
- Found bf16 hidden states hitting fp32 router weights without autocast.
- Fixed router forward to train the small router in fp32 from fp32-cast hidden states.
- Fixed objective weighting so budget regularization is not accidentally scaled by `lambda_router`.

## Verification

Remote `yui` review:

```text
python -m py_compile lgar_cpt/*.py
python -m unittest discover -s tests -p 'test_lgar_*.py'
Ran 7 tests in 0.050s
OK
qwen_config_ok 896 24
```

Real Qwen smoke:

```text
seq_len=512
short_window=128
steps=1
attention=eager
paths: CE, RouterAux, hard L-GAR high-budget
result: completed, no NaN, hard L-GAR actual budget matched target
```

Stage 1 smoke:

```text
seq_len=512
tokens_per_run=512
runs: Qwen_CE_CPT, Qwen_LongCE_CPT, Qwen_RouterAux_CPT, Qwen_LGAR_CPT
result: completed, summary written
```

The 1-step smokes did not pass the router-quality Stage 0 gate because `router_beats_random=false`; that is expected for a 1-step smoke and is not used as the official decision.

## Launch Decision

Launch official Stage 0 audit on `yui`.

Launch Stage 1 only after the official Stage 0 summary passes all checks.
