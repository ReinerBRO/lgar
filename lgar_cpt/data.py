from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import islice
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from .config import Paths
from .utils import ensure_dir, write_json

AutoTokenizer: Any | None = None


def _auto_tokenizer() -> Any:
    global AutoTokenizer
    if AutoTokenizer is None:
        from transformers import AutoTokenizer as _AutoTokenizer

        AutoTokenizer = _AutoTokenizer
    return AutoTokenizer


def parquet_files(raw_data_dir: str | Path, max_shards: int | None = None) -> list[Path]:
    data_dir = Path(raw_data_dir) / "data"
    files = sorted(data_dir.glob("train-*.parquet"))
    if max_shards is not None:
        files = files[: int(max_shards)]
    if not files:
        raise FileNotFoundError(f"no parquet shards found under {data_dir}")
    return files


def _tokenize_shard_to_temp(payload: dict[str, Any]) -> dict[str, Any]:
    shard = Path(payload["shard"])
    shard_index = int(payload["shard_index"])
    model_path = str(payload["model_path"])
    tmp_dir = Path(payload["tmp_dir"])
    min_doc_tokens = int(payload["min_doc_tokens"])
    batch_size = int(payload["batch_size"])
    add_eos = bool(payload["add_eos"])

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    tokenizer = _auto_tokenizer().from_pretrained(model_path, trust_remote_code=True)
    eos_id = tokenizer.eos_token_id
    append_eos = add_eos and eos_id is not None

    token_path = tmp_dir / f"shard_{shard_index:05d}.tokens.i32"
    length_path = tmp_dir / f"shard_{shard_index:05d}.lengths.npy"
    docs_path = tmp_dir / f"shard_{shard_index:05d}.docs.jsonl"
    lengths: list[int] = []
    docs_seen = 0
    token_count = 0

    with token_path.open("wb") as tok_f, docs_path.open("w", encoding="utf-8") as meta_f:
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
                arr.tofile(tok_f)
                lengths.append(int(arr.size))
                token_count += int(arr.size)
                meta_f.write(
                    json.dumps(
                        {
                            "source_shard": shard.name,
                            "id": batch["id"][local_idx],
                            "url": batch["url"][local_idx],
                            "num_tokens": int(arr.size),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )

    np.save(length_path, np.asarray(lengths, dtype=np.int32))
    return {
        "shard_index": shard_index,
        "source_shard": shard.name,
        "tokens_path": str(token_path),
        "lengths_path": str(length_path),
        "docs_path": str(docs_path),
        "docs_seen": int(docs_seen),
        "docs_kept": int(len(lengths)),
        "token_count": int(token_count),
        "add_eos": bool(append_eos),
        "eos_token_id": eos_id,
    }


def _parallel_context() -> mp.context.BaseContext | None:
    try:
        return mp.get_context("fork")
    except ValueError:
        return None


def prepare_qwen_fineweb_cache_parallel(
    raw_data_dir: str | Path,
    model_path: str | Path,
    cache_dir: str | Path,
    target_tokens: int,
    max_shards: int | None = None,
    min_doc_tokens: int = 256,
    batch_size: int = 256,
    add_eos: bool = True,
    workers: int = 64,
    copy_chunk_tokens: int = 16_000_000,
) -> dict[str, Any]:
    """Build the Qwen FineWeb cache with shard-level multiprocessing.

    Each worker tokenizes a parquet shard into temporary binary chunks. Rank-0
    then concatenates selected documents in deterministic shard order into the
    canonical cache files consumed by training.
    """

    cache_dir = ensure_dir(cache_dir)
    tmp_dir = cache_dir / "_tmp_tokenize"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    files = parquet_files(raw_data_dir, max_shards=max_shards)
    workers = max(1, min(int(workers), len(files)))
    selected: list[tuple[dict[str, Any], int, int]] = []
    docs_seen = 0
    docs_kept = 0
    total_tokens = 0
    eos_id: int | None = None
    append_eos_actual = False

    payloads = [
        {
            "shard": str(shard),
            "shard_index": idx,
            "model_path": str(model_path),
            "tmp_dir": str(tmp_dir),
            "min_doc_tokens": int(min_doc_tokens),
            "batch_size": int(batch_size),
            "add_eos": bool(add_eos),
        }
        for idx, shard in enumerate(files)
    ]

    stop = False
    ctx = _parallel_context()
    executor_kwargs: dict[str, Any] = {"max_workers": workers}
    if ctx is not None:
        executor_kwargs["mp_context"] = ctx
    with ProcessPoolExecutor(**executor_kwargs) as executor:
        for start in range(0, len(payloads), workers):
            round_payloads = payloads[start : start + workers]
            results = list(executor.map(_tokenize_shard_to_temp, round_payloads))
            for result in sorted(results, key=lambda x: int(x["shard_index"])):
                if eos_id is None:
                    eos_id = result.get("eos_token_id")
                append_eos_actual = append_eos_actual or bool(result.get("add_eos", False))
                docs_seen += int(result["docs_seen"])
                lengths = np.load(result["lengths_path"], mmap_mode="r")
                if lengths.size == 0:
                    print(json.dumps({"event": "tokenized_shard_empty", **result}, sort_keys=True), flush=True)
                    continue
                remaining = int(target_tokens) - int(total_tokens)
                include_docs = int(lengths.size)
                if remaining > 0:
                    cumulative = np.cumsum(lengths, dtype=np.int64)
                    include_docs = int(np.searchsorted(cumulative, remaining, side="left") + 1)
                    include_docs = min(include_docs, int(lengths.size))
                selected_tokens = int(np.sum(lengths[:include_docs], dtype=np.int64))
                selected.append((result, include_docs, selected_tokens))
                docs_kept += int(include_docs)
                total_tokens += int(selected_tokens)
                print(
                    json.dumps(
                        {
                            "event": "selected_shard",
                            "source_shard": result["source_shard"],
                            "include_docs": include_docs,
                            "selected_tokens": selected_tokens,
                            "total_tokens": total_tokens,
                            "target_tokens": int(target_tokens),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                if total_tokens >= int(target_tokens):
                    stop = True
                    break
            if stop:
                break

    if total_tokens <= 0 or docs_kept <= 0:
        raise RuntimeError("tokenization produced no usable documents")

    tokens_mm = np.lib.format.open_memmap(cache_dir / "tokens.npy", mode="w+", dtype=np.int32, shape=(total_tokens,))
    offsets_mm = np.lib.format.open_memmap(cache_dir / "doc_offsets.npy", mode="w+", dtype=np.int64, shape=(docs_kept,))
    lengths_mm = np.lib.format.open_memmap(cache_dir / "doc_lengths.npy", mode="w+", dtype=np.int32, shape=(docs_kept,))

    token_pos = 0
    doc_pos = 0
    with (cache_dir / "docs.jsonl").open("w", encoding="utf-8") as meta_out:
        for result, include_docs, selected_tokens in selected:
            src = np.memmap(result["tokens_path"], dtype=np.int32, mode="r")
            copied = 0
            while copied < selected_tokens:
                n = min(int(copy_chunk_tokens), int(selected_tokens - copied))
                tokens_mm[token_pos + copied : token_pos + copied + n] = src[copied : copied + n]
                copied += n

            lengths = np.load(result["lengths_path"], mmap_mode="r")[:include_docs].astype(np.int64, copy=False)
            local_offsets = np.concatenate(([0], np.cumsum(lengths[:-1], dtype=np.int64)))
            offsets_mm[doc_pos : doc_pos + include_docs] = int(token_pos) + local_offsets
            lengths_mm[doc_pos : doc_pos + include_docs] = lengths.astype(np.int32, copy=False)

            with Path(result["docs_path"]).open("r", encoding="utf-8") as meta_in:
                for line in islice(meta_in, include_docs):
                    row = json.loads(line)
                    row["doc_row"] = int(doc_pos)
                    meta_out.write(json.dumps(row, sort_keys=True) + "\n")
                    doc_pos += 1

            token_pos += int(selected_tokens)

    del tokens_mm, offsets_mm, lengths_mm
    shutil.rmtree(tmp_dir, ignore_errors=True)

    info = {
        "raw_data_dir": str(raw_data_dir),
        "model_path": str(model_path),
        "target_tokens": int(target_tokens),
        "actual_tokens": int(total_tokens),
        "docs_seen": int(docs_seen),
        "docs_kept": int(docs_kept),
        "min_doc_tokens": int(min_doc_tokens),
        "add_eos": bool(append_eos_actual),
        "eos_token_id": eos_id,
        "max_shards": max_shards,
        "workers": int(workers),
        "builder": "parallel_shard_tokenizer",
    }
    write_json(cache_dir / "cache_info.json", info)
    return info


def prepare_qwen_fineweb_cache(
    raw_data_dir: str | Path,
    model_path: str | Path,
    cache_dir: str | Path,
    target_tokens: int,
    max_shards: int | None = None,
    min_doc_tokens: int = 256,
    batch_size: int = 256,
    add_eos: bool = True,
    workers: int = 1,
) -> dict[str, Any]:
    """Tokenize raw FineWebEdu documents with the Qwen tokenizer.

    The cache keeps explicit document starts and lengths so packing can still
    enforce same-document LSD mining and long-context evaluation.
    """

    if int(workers) > 1:
        return prepare_qwen_fineweb_cache_parallel(
            raw_data_dir=raw_data_dir,
            model_path=model_path,
            cache_dir=cache_dir,
            target_tokens=target_tokens,
            max_shards=max_shards,
            min_doc_tokens=min_doc_tokens,
            batch_size=batch_size,
            add_eos=add_eos,
            workers=int(workers),
        )

    cache_dir = ensure_dir(cache_dir)
    tokenizer = _auto_tokenizer().from_pretrained(str(model_path), trust_remote_code=True)
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
        sae_sparse_meta_path = self.signal_dir / "offline_sae_sparse_meta.json"
        self.offline_sae_sparse_meta = None
        self.offline_sae_sparse_offsets = None
        self.offline_sae_sparse_token_positions = None
        self.offline_sae_sparse_full = None
        self.offline_sae_sparse_neg = None
        if sae_sparse_meta_path.exists():
            self.offline_sae_sparse_meta = json.loads(sae_sparse_meta_path.read_text(encoding="utf-8"))
            self.offline_sae_sparse_offsets = np.load(
                self.signal_dir / "offline_sae_sparse_offsets.npy",
                mmap_mode="r",
            )
            self.offline_sae_sparse_token_positions = np.load(
                self.signal_dir / "offline_sae_sparse_token_positions.npy",
                mmap_mode="r",
            )
            self.offline_sae_sparse_full = np.load(
                self.signal_dir / "offline_sae_sparse_full.npy",
                mmap_mode="r",
            )
            self.offline_sae_sparse_neg = np.load(
                self.signal_dir / "offline_sae_sparse_neg.npy",
                mmap_mode="r",
            )
        sae_full_path = self.signal_dir / "offline_sae_feature_full.npy"
        sae_neg_path = self.signal_dir / "offline_sae_feature_neg.npy"
        sae_mask_path = self.signal_dir / "offline_sae_feature_mask.npy"
        self.offline_sae_feature_full = (
            np.load(sae_full_path, mmap_mode="r")
            if sae_sparse_meta_path.exists() is False and sae_full_path.exists()
            else None
        )
        self.offline_sae_feature_neg = (
            np.load(sae_neg_path, mmap_mode="r")
            if sae_sparse_meta_path.exists() is False and sae_neg_path.exists()
            else None
        )
        self.offline_sae_feature_mask = (
            np.load(sae_mask_path, mmap_mode="r")
            if sae_sparse_meta_path.exists() is False and sae_mask_path.exists()
            else None
        )

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        batch = super().sample(batch_size)
        indices = np.asarray(batch["sequence_indices"], dtype=np.int64)
        batch["offline_lsd_labels"] = np.asarray(self.offline_labels[indices], dtype=bool)
        batch["offline_lsd_valid"] = np.asarray(self.offline_valid[indices], dtype=bool)
        if self.offline_lsd is not None:
            batch["offline_lsd"] = np.asarray(self.offline_lsd[indices], dtype=np.float32)
        if self.offline_sae_sparse_meta is not None:
            num_features = int(self.offline_sae_sparse_meta["num_features"])
            full = np.zeros((indices.shape[0], self.seq_len, num_features), dtype=np.float32)
            neg = np.zeros_like(full)
            mask = np.zeros((indices.shape[0], self.seq_len), dtype=bool)
            assert self.offline_sae_sparse_offsets is not None
            assert self.offline_sae_sparse_token_positions is not None
            assert self.offline_sae_sparse_full is not None
            assert self.offline_sae_sparse_neg is not None
            for row, seq_idx in enumerate(indices.tolist()):
                offset_start = int(self.offline_sae_sparse_offsets[int(seq_idx)])
                offset_end = int(self.offline_sae_sparse_offsets[int(seq_idx) + 1])
                if offset_end <= offset_start:
                    continue
                positions = np.asarray(
                    self.offline_sae_sparse_token_positions[offset_start:offset_end],
                    dtype=np.int64,
                )
                full[row, positions] = np.asarray(
                    self.offline_sae_sparse_full[offset_start:offset_end],
                    dtype=np.float32,
                )
                neg[row, positions] = np.asarray(
                    self.offline_sae_sparse_neg[offset_start:offset_end],
                    dtype=np.float32,
                )
                mask[row, positions] = True
            batch["offline_sae_feature_full"] = full
            batch["offline_sae_feature_neg"] = neg
            batch["offline_sae_feature_mask"] = mask
        if self.offline_sae_feature_full is not None:
            batch["offline_sae_feature_full"] = np.asarray(
                self.offline_sae_feature_full[indices], dtype=np.float32
            )
        if self.offline_sae_feature_neg is not None:
            batch["offline_sae_feature_neg"] = np.asarray(
                self.offline_sae_feature_neg[indices], dtype=np.float32
            )
        if self.offline_sae_feature_mask is not None:
            batch["offline_sae_feature_mask"] = np.asarray(
                self.offline_sae_feature_mask[indices], dtype=bool
            )
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
    parser.add_argument("--workers", type=int, default=1)
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
        workers=args.workers,
    )
    print(json.dumps(info, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
