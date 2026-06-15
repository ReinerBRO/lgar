from __future__ import annotations

from curcpt.model_eval_utils import infer_lora_alpha


def test_infer_lora_alpha_prefers_checkpoint_summary() -> None:
    assert infer_lora_alpha({"summary": {"lora_alpha": 32.0}}, default=16.0) == 32.0


def test_infer_lora_alpha_falls_back_to_default() -> None:
    assert infer_lora_alpha({"summary": {}}, default=16.0) == 16.0
