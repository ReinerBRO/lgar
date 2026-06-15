from __future__ import annotations

from curcpt.ruler_answer_probe import build_score_items


class TinyTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(ch) % 251 for ch in text]


def test_build_score_items_scores_target_tokens_after_answer_prefix() -> None:
    row = {
        "index": 7,
        "input": "question?",
        "answer_prefix": " Answer:",
        "outputs": ["blue"],
        "token_position_answer": 123,
    }
    item = build_score_items(TinyTokenizer(), "qa_1", row, seq_len=128)[0]

    # Label positions predict ids[pos + 1]. The first scored label is the
    # leading separator before the gold output because the prefix ends with ':'.
    first_scored = item.target_mask.index(True)
    assert item.ids[first_scored + 1] == ord(" ") % 251
    assert sum(item.target_mask) == len(" blue")
    assert item.index == 7
    assert item.answer_position == 123


def test_build_score_items_left_crops_prompt_but_keeps_target_mask() -> None:
    row = {
        "index": 1,
        "input": "x" * 40,
        "answer_prefix": " Answer:",
        "outputs": ["abc"],
    }
    item = build_score_items(TinyTokenizer(), "niah_single_1", row, seq_len=10)[0]

    assert len(item.ids) == 10
    assert sum(item.target_mask) == len(" abc")
    assert all(item.target_mask[-len(" abc") :])
