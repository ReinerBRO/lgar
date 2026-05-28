from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer

from .config import Paths
from .utils import ensure_dir, write_json


def parquet_files(raw_data_dir: str | Path, max_shards: int | None = None) -> list[Path]:
    data_dir = Path(raw_data_dir) / "data"
    files = sorted(data_dir.glob("train-*.parquet"))
    if max_shards is not None:
        files = files[: int(max_shards)]
    if not files:
        raise FileNotFoundError(f"no parquet shards found under {data_dir}")
    return files


def prepare_qwen_fineweb_cache(
    raw_data_dir: str | Path,
    model_path: str | Path,
    cache_dir: str | Path,
    target_tokens: int,
    max_shards: int | None = None,
    min_doc_tokens: int = 256,
    batch_size: int = 256,
    add_eos: bool = True,
) -> dict[str, Any]:
    """Tokenize raw FineWebEdu documents with the Qwen tokenizer.

    The cache keeps explicit document starts and lengths so packing can still
    enforce same-document LSD mining and long-context evaluation.
    """

    cache_dir = ensure_dir(cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    eos_id = tokenizer.eos_token_id
    append_eos = add_eos and eos_id is not None

    all_tokens: list[np.ndarray] = []
    offsets: list[int] = []
    lengths: list[int] = []
    metadata_path = cache_dir / "docs.jsonl"
    total = 0
    docs_kept = 0
    docs_seen = 0

    with metadata_path.open("w", encoding="utf-8") as meta_f:
        for shard in parquet_files(raw_data_dir, max_shards=max_shards):
            pf = pq.ParquetFile(shard)
            for rb in pf.iter_batches(batch_size=batch_size, columns=["id", "text", "url"]):
                batch = rb.to_pydict()
                texts = [str(x or "") for x in batch["text"]]
                encoded = tokenizer(
                    texts,
                    add_special_tokens=False,
                    padding=False,
                    truncation=False,
                    return_attention_mask=False,
                )["input_ids"]
                for local_idx, ids in enumerate(encoded):
                    docs_seen += 1
                    if append_eos:
                        ids = list(ids) + [int(eos_id)]
                    if len(ids) < min_doc_tokens:
                        continue
                    arr = np.asarray(ids, dtype=np.int32)
                    offsets.append(total)
                    lengths.append(int(arr.size))
                    all_tokens.append(arr)
                    meta_f.write(
                        json.dumps(
                            {
                                "doc_row": docs_kept,
                                "source_shard": shard.name,
                                "id": batch["id"][local_idx],
                                "url": batch["url"][local_idx],
                                "num_tokens": int(arr.size),
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    total += int(arr.size)
                    docs_kept += 1
                    if total >= target_tokens:
                        break
                if total >= target_tokens:
                    break
            if total >= target_tokens:
                break

    if not all_tokens:
        raise RuntimeError("tokenization produced no usable documents")

    tokens = np.concatenate(all_tokens).astype(np.int32, copy=False)
    np.save(cache_dir / "tokens.npy", tokens)
    np.save(cache_dir / "doc_offsets.npy", np.asarray(offsets, dtype=np.int64))
    np.save(cache_dir / "doc_lengths.npy", np.asarray(lengths, dtype=np.int32))
    info = {
        "raw_data_dir": str(raw_data_dir),
        "model_path": str(model_path),
        "target_tokens": int(target_tokens),
        "actual_tokens": int(tokens.size),
        "docs_seen": int(docs_seen),
        "docs_kept": int(docs_kept),
        "min_doc_tokens": int(min_doc_tokens),
        "add_eos": bool(append_eos),
        "eos_token_id": eos_id,
        "max_shards": max_shards,
    }
    write_json(cache_dir / "cache_info.json", info)
    return info


class PackedFineWebDataset:
    def __init__(
        self,
        cache_dir: str | Path,
        seq_len: int,
        pad_token_id: int,
        split: str,
        seed: int = 1337,
        val_docs: int = 1024,
        min_doc_tokens: int = 256,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.seq_len = int(seq_len)
        self.pad_token_id = int(pad_token_id)
        self.tokens = np.load(self.cache_dir / "tokens.npy", mmap_mode="r")
        self.doc_offsets = np.load(self.cache_dir / "doc_offsets.npy", mmap_mode="r")
        self.doc_lengths = np.load(self.cache_dir / "doc_lengths.npy", mmap_mode="r")
        self.rng = np.random.default_rng(seed)
        self.split = split
        valid_docs = np.nonzero(self.doc_lengths >= int(min_doc_tokens))[0]
        val_docs = min(int(val_docs), max(1, valid_docs.size // 10), valid_docs.size)
        if split == "val":
            self.doc_indices = valid_docs[:val_docs]
        elif split == "train":
            self.doc_indices = valid_docs[val_docs:] if valid_docs.size > val_docs else valid_docs
        else:
            raise ValueError(f"unsupported split: {split}")
        if self.doc_indices.size == 0:
            raise RuntimeError(f"no docs available for split {split}")
        self.min_doc_tokens = int(min_doc_tokens)
        self.max_segments = int(math.ceil((self.seq_len + 1) / max(1, self.min_doc_tokens))) + 1

    def _sample_doc_span(self, remaining: int) -> tuple[int, int, int]:
        doc_row = int(self.rng.choice(self.doc_indices))
        doc_len = int(self.doc_lengths[doc_row])
        take = min(int(remaining), doc_len)
        if self.split == "train" and doc_len > take:
            start_in_doc = int(self.rng.integers(0, doc_len - take + 1))
        else:
            start_in_doc = 0
        return doc_row, start_in_doc, int(take)

    def _load_doc_piece(self, doc_row: int, start_in_doc: int, take: int) -> np.ndarray:
        start = int(self.doc_offsets[int(doc_row)]) + int(start_in_doc)
        return np.asarray(self.tokens[start : start + int(take)], dtype=np.int64)

    def sample_layout_batch(self, batch_size: int) -> dict[str, np.ndarray]:
        segment_doc_rows = np.full((batch_size, self.max_segments), -1, dtype=np.int32)
        segment_start_offsets = np.full((batch_size, self.max_segments), -1, dtype=np.int32)
        segment_lengths = np.zeros((batch_size, self.max_segments), dtype=np.int32)
        full_len = self.seq_len + 1
        for b in range(batch_size):
            pos = 0
            segment = 0
            while pos < full_len:
                doc_row, start_in_doc, take = self._sample_doc_span(full_len - pos)
                if segment >= self.max_segments:
                    raise RuntimeError("sample exceeded configured max_segments")
                segment_doc_rows[b, segment] = int(doc_row)
                segment_start_offsets[b, segment] = int(start_in_doc)
                segment_lengths[b, segment] = int(take)
                pos += int(take)
                segment += 1
        return {
            "segment_doc_rows": segment_doc_rows,
            "segment_start_offsets": segment_start_offsets,
            "segment_lengths": segment_lengths,
        }

    def materialize_layout_batch(self, layout: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        segment_doc_rows = np.asarray(layout["segment_doc_rows"], dtype=np.int32)
        segment_start_offsets = np.asarray(layout["segment_start_offsets"], dtype=np.int32)
        segment_lengths = np.asarray(layout["segment_lengths"], dtype=np.int32)
        batch_size = int(segment_doc_rows.shape[0])
        full_len = self.seq_len + 1
        ids = np.full((batch_size, full_len), self.pad_token_id, dtype=np.int64)
        doc_rows = np.full((batch_size, full_len), -1, dtype=np.int64)
        source_doc_rows = np.full((batch_size, full_len), -1, dtype=np.int64)
        offsets_in_doc = np.full((batch_size, full_len), -1, dtype=np.int64)
        for b in range(batch_size):
            pos = 0
            segment = 0
            while segment < segment_doc_rows.shape[1]:
                doc_row = int(segment_doc_rows[b, segment])
                take = int(segment_lengths[b, segment])
                if doc_row < 0 or take <= 0:
                    break
                start_in_doc = int(segment_start_offsets[b, segment])
                piece = self._load_doc_piece(doc_row, start_in_doc, take)
                n = int(piece.size)
                pack_doc_id = b * 1_000_000 + segment
                ids[b, pos : pos + n] = piece
                doc_rows[b, pos : pos + n] = pack_doc_id
                source_doc_rows[b, pos : pos + n] = doc_row
                offsets_in_doc[b, pos : pos + n] = np.arange(start_in_doc, start_in_doc + n)
                pos += n
                segment += 1
        input_ids = ids[:, :-1]
        labels = ids[:, 1:]
        input_doc = doc_rows[:, :-1]
        target_doc = doc_rows[:, 1:]
        loss_mask = (input_doc == target_doc) & (target_doc >= 0) & (labels != self.pad_token_id)
        is_padding = ids == self.pad_token_id
        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "loss_mask": loss_mask,
            "doc_ids_full": doc_rows,
            "source_doc_rows_full": source_doc_rows,
            "doc_offsets_full": offsets_in_doc,
            "is_padding_full": is_padding,
        }
        if "sequence_indices" in layout:
            batch["sequence_indices"] = np.asarray(layout["sequence_indices"], dtype=np.int64)
        return batch

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        return self.materialize_layout_batch(self.sample_layout_batch(batch_size))


def packed_sequence_count_for_tokens(target_tokens: int, seq_len: int) -> int:
    return max(1, int(math.ceil(int(target_tokens) / max(1, int(seq_len)))))


def prepare_packed_sequence_layout_cache(
    cache_dir: str | Path,
    output_dir: str | Path,
    seq_len: int,
    pad_token_id: int,
    split: str,
    num_sequences: int,
    seed: int = 1337,
    min_doc_tokens: int = 256,
    val_docs: int = 1024,
    write_batch_size: int = 1024,
) -> dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    dataset = PackedFineWebDataset(
        cache_dir=cache_dir,
        seq_len=seq_len,
        pad_token_id=pad_token_id,
        split=split,
        seed=seed,
        val_docs=val_docs,
        min_doc_tokens=min_doc_tokens,
    )
    shape = (int(num_sequences), int(dataset.max_segments))
    doc_rows_mm = np.lib.format.open_memmap(output_dir / "segment_doc_rows.npy", mode="w+", dtype=np.int32, shape=shape)
    starts_mm = np.lib.format.open_memmap(output_dir / "segment_start_offsets.npy", mode="w+", dtype=np.int32, shape=shape)
    lengths_mm = np.lib.format.open_memmap(output_dir / "segment_lengths.npy", mode="w+", dtype=np.int32, shape=shape)
    for start in range(0, int(num_sequences), int(write_batch_size)):
        n = min(int(write_batch_size), int(num_sequences) - start)
        layout = dataset.sample_layout_batch(n)
        doc_rows_mm[start : start + n] = layout["segment_doc_rows"]
        starts_mm[start : start + n] = layout["segment_start_offsets"]
        lengths_mm[start : start + n] = layout["segment_lengths"]
    del doc_rows_mm, starts_mm, lengths_mm
    info = {
        "cache_dir": str(cache_dir),
        "split": str(split),
        "seq_len": int(seq_len),
        "pad_token_id": int(pad_token_id),
        "num_sequences": int(num_sequences),
        "seed": int(seed),
        "min_doc_tokens": int(min_doc_tokens),
        "val_docs": int(val_docs),
        "max_segments": int(dataset.max_segments),
    }
    write_json(output_dir / "layout_meta.json", info)
    return info


class PackedSequenceLayoutDataset:
    def __init__(
        self,
        cache_dir: str | Path,
        signal_dir: str | Path,
        pad_token_id: int,
        seed: int = 1337,
    ) -> None:
        self.signal_dir = Path(signal_dir)
        meta = json.loads((self.signal_dir / "layout_meta.json").read_text(encoding="utf-8"))
        self.base = PackedFineWebDataset(
            cache_dir=cache_dir,
            seq_len=int(meta["seq_len"]),
            pad_token_id=int(pad_token_id),
            split=str(meta["split"]),
            seed=seed,
            val_docs=int(meta.get("val_docs", 1024)),
            min_doc_tokens=int(meta.get("min_doc_tokens", 256)),
        )
        self.seq_len = int(meta["seq_len"])
        self.rng = np.random.default_rng(seed)
        self.num_sequences = int(meta["num_sequences"])
        self.segment_doc_rows = np.load(self.signal_dir / "segment_doc_rows.npy", mmap_mode="r")
        self.segment_start_offsets = np.load(self.signal_dir / "segment_start_offsets.npy", mmap_mode="r")
        self.segment_lengths = np.load(self.signal_dir / "segment_lengths.npy", mmap_mode="r")

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        indices = self.rng.integers(0, self.num_sequences, size=int(batch_size), dtype=np.int64)
        layout = {
            "segment_doc_rows": np.asarray(self.segment_doc_rows[indices], dtype=np.int32),
            "segment_start_offsets": np.asarray(self.segment_start_offsets[indices], dtype=np.int32),
            "segment_lengths": np.asarray(self.segment_lengths[indices], dtype=np.int32),
            "sequence_indices": np.asarray(indices, dtype=np.int64),
        }
        return self.base.materialize_layout_batch(layout)


class PackedSequenceSignalDataset(PackedSequenceLayoutDataset):
    def __init__(
        self,
        cache_dir: str | Path,
        signal_dir: str | Path,
        pad_token_id: int,
        seed: int = 1337,
    ) -> None:
        super().__init__(cache_dir=cache_dir, signal_dir=signal_dir, pad_token_id=pad_token_id, seed=seed)
        self.offline_labels = np.load(self.signal_dir / "offline_lsd_labels.npy", mmap_mode="r")
        self.offline_valid = np.load(self.signal_dir / "offline_lsd_valid.npy", mmap_mode="r")
        offline_lsd_path = self.signal_dir / "offline_lsd.npy"
        self.offline_lsd = np.load(offline_lsd_path, mmap_mode="r") if offline_lsd_path.exists() else None

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        batch = super().sample(batch_size)
        indices = np.asarray(batch["sequence_indices"], dtype=np.int64)
        batch["offline_lsd_labels"] = np.asarray(self.offline_labels[indices], dtype=bool)
        batch["offline_lsd_valid"] = np.asarray(self.offline_valid[indices], dtype=bool)
        if self.offline_lsd is not None:
            batch["offline_lsd"] = np.asarray(self.offline_lsd[indices], dtype=np.float32)
        return batch


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Qwen-tokenized FineWebEdu document cache.")
    defaults = Paths()
    parser.add_argument("--raw-data-dir", default=defaults.raw_data_dir)
    parser.add_argument("--model-path", default=defaults.model_path)
    parser.add_argument("--cache-dir", default=defaults.cache_dir)
    parser.add_argument("--target-tokens", type=int, default=64_000_000)
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--min-doc-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--no-eos", action="store_true")
    args = parser.parse_args()
    info = prepare_qwen_fineweb_cache(
        raw_data_dir=args.raw_data_dir,
        model_path=args.model_path,
        cache_dir=args.cache_dir,
        target_tokens=args.target_tokens,
        max_shards=args.max_shards,
        min_doc_tokens=args.min_doc_tokens,
        batch_size=args.batch_size,
        add_eos=not args.no_eos,
    )
    print(json.dumps(info, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
