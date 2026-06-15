from __future__ import annotations

import json
from pathlib import Path
import sys
import types

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.modules.setdefault(
    "torch",
    types.SimpleNamespace(
        bfloat16=object(),
        float16=object(),
        float32=object(),
        manual_seed=lambda seed: None,
        cuda=types.SimpleNamespace(is_available=lambda: False),
        device=lambda name: name,
        __version__="fake",
    ),
)

from lgar_cpt import data as data_mod


class _FakeTokenizer:
    eos_token_id = None

    def __call__(self, texts, **kwargs):
        del kwargs
        return {"input_ids": [[ord(ch) % 257 for ch in text] for text in texts]}


class _FakeAutoTokenizer:
    @staticmethod
    def from_pretrained(*args, **kwargs):
        del args, kwargs
        return _FakeTokenizer()


def _write_shard(path: Path, rows: list[tuple[str, str, str]]) -> None:
    table = pa.table(
        {
            "id": [row[0] for row in rows],
            "text": [row[1] for row in rows],
            "url": [row[2] for row in rows],
        }
    )
    pq.write_table(table, path)


def test_parallel_fineweb_cache_preserves_shard_order_and_doc_metadata(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(data_mod, "AutoTokenizer", _FakeAutoTokenizer)
    raw = tmp_path / "raw"
    data_dir = raw / "data"
    data_dir.mkdir(parents=True)
    _write_shard(
        data_dir / "train-00000-of-00100.parquet",
        [
            ("a", "abc", "u0"),
            ("b", "defg", "u1"),
        ],
    )
    _write_shard(
        data_dir / "train-00001-of-00100.parquet",
        [
            ("c", "hijkl", "u2"),
            ("d", "mnopqr", "u3"),
        ],
    )

    out = tmp_path / "cache"
    info = data_mod.prepare_qwen_fineweb_cache(
        raw_data_dir=raw,
        model_path=tmp_path / "fake-model",
        cache_dir=out,
        target_tokens=10,
        min_doc_tokens=1,
        batch_size=2,
        add_eos=False,
        workers=2,
    )

    assert info["builder"] == "parallel_shard_tokenizer"
    assert info["actual_tokens"] == 12
    assert np.load(out / "doc_lengths.npy").tolist() == [3, 4, 5]
    assert np.load(out / "doc_offsets.npy").tolist() == [0, 3, 7]
    assert np.load(out / "tokens.npy").tolist() == [ord(ch) % 257 for ch in "abcdefghijkl"]

    docs = [json.loads(line) for line in (out / "docs.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [doc["doc_row"] for doc in docs] == [0, 1, 2]
    assert [doc["id"] for doc in docs] == ["a", "b", "c"]
    assert [doc["source_shard"] for doc in docs] == [
        "train-00000-of-00100.parquet",
        "train-00000-of-00100.parquet",
        "train-00001-of-00100.parquet",
    ]
