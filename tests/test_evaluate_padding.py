from __future__ import annotations

import unittest

from lgar_cpt.evaluate import _pad_to_tokens


class _CharTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[str]:
        del add_special_tokens
        return list(text)

    def decode(self, ids: list[str]) -> str:
        return "".join(ids)


class EvaluatePaddingTests(unittest.TestCase):
    def test_pad_to_tokens_reaches_target_length(self) -> None:
        tokenizer = _CharTokenizer()
        prefix = "facts:\n"
        suffix = "\nquestion?"
        text = _pad_to_tokens(tokenizer, prefix, suffix, target_tokens=128)
        self.assertGreaterEqual(len(tokenizer.encode(text)), 128)
        self.assertTrue(text.startswith(prefix))
        self.assertTrue(text.endswith(suffix))

    def test_pad_to_tokens_preserves_suffix_when_prefix_is_too_long(self) -> None:
        tokenizer = _CharTokenizer()
        prefix = "a" * 64
        suffix = "XYZ"
        text = _pad_to_tokens(tokenizer, prefix, suffix, target_tokens=16)
        self.assertEqual(text[-len(suffix) :], suffix)
        self.assertEqual(len(tokenizer.encode(text)), 16)


if __name__ == "__main__":
    unittest.main()
