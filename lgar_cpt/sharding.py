from __future__ import annotations


def iter_shard_ranges(num_sequences: int, batch_size: int, rank: int, world_size: int) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    stride = max(1, int(batch_size)) * max(1, int(world_size))
    local_batch = max(1, int(batch_size))
    for global_start in range(0, int(num_sequences), stride):
        start = global_start + int(rank) * local_batch
        if start >= int(num_sequences):
            continue
        end = min(int(num_sequences), start + local_batch)
        ranges.append((start, end))
    return ranges
