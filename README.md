# L-GAR CPT Validation Project

This project implements the Stage 0 validation harness for `L-GAR-CPT: LSD-Guided Global Attention Router for Qwen2.5-0.5B`.

It follows the spec in `specs/02_L_GAR_Qwen0.5B_CPT.md`:

- Base checkpoint: `/gemini/space/private/zjc/models/Qwen2.5-0.5B`
- Tokenizer: Qwen2.5 tokenizer from the same checkpoint
- Data source: `/gemini/space/private/zjc/data/fineweb_edu_100BT-shuffled`
- Remote project root: `/gemini/space/private/zjc/goals/lgar`
- Environment: `/gemini/space/private/zjc/envs/zjc_env`
- Compute node: yui only

Stage 0 validates document-aware Qwen tokenization, LSD label mining, manual label audit, RouterAux 100-step training, and local/global routed mask tests.

Stage 0 uses Qwen `eager` attention because the real run smoke found `sdpa` produces NaN with the required 4D document-aware additive masks. Do not switch back to `sdpa` unless the mask path is revalidated.

Hard L-GAR uses a custom Qwen forward path that keeps lower layers on full document attention and injects per-query local/global masks only into the selected upper routed layers. `scripts/run_stage1.sh` requires a passing Stage 0 summary before launch.

Run on yui after syncing:

```bash
cd /gemini/space/private/zjc/goals/lgar
bash scripts/review_stage0_impl.sh
bash scripts/run_stage0.sh
```
