from lgar_cpt.sharding import iter_shard_ranges


def test_iter_shard_ranges_covers_all_sequences_without_overlap() -> None:
    num_sequences = 23
    batch_size = 3
    world_size = 4

    covered: list[int] = []
    for rank in range(world_size):
        for start, end in iter_shard_ranges(num_sequences, batch_size, rank, world_size):
            covered.extend(range(start, end))

    assert sorted(covered) == list(range(num_sequences))
    assert len(covered) == len(set(covered))


def test_iter_shard_ranges_handles_more_ranks_than_work() -> None:
    num_sequences = 5
    batch_size = 2
    world_size = 8

    covered: list[int] = []
    active_ranks = 0
    for rank in range(world_size):
        shard_ranges = iter_shard_ranges(num_sequences, batch_size, rank, world_size)
        if shard_ranges:
            active_ranks += 1
        for start, end in shard_ranges:
            covered.extend(range(start, end))

    assert sorted(covered) == list(range(num_sequences))
    assert len(covered) == len(set(covered))
    assert active_ranks == 3
