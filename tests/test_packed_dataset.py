"""Token packing and batching."""

from __future__ import annotations

import pytest
import torch

from tinyllm.data.packed_dataset import PackedBlocks, ShuffledBatchSource, fixed_batches, tokenize_stories


class FakeTokenizer:
    bos_id, eos_id = 1, 2

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = [10 + len(w) for w in text.split()]
        return [self.bos_id, *ids, self.eos_id] if add_bos and add_eos else ids


def test_tokenize_stories_wraps_each_story_and_respects_cap():
    tokens = tokenize_stories(["a bb", "ccc", "dddd"], FakeTokenizer(), max_stories=2)
    assert tokens.tolist() == [1, 11, 12, 2, 1, 13, 2]


def test_blocks_drop_the_ragged_tail():
    blocks = PackedBlocks.from_stream(torch.arange(25), seq_len=8)
    assert len(blocks) == 3
    assert blocks.get(torch.tensor([1]))[0].tolist() == list(range(8, 16))


def test_batches_cover_each_block_once_per_epoch():
    blocks = PackedBlocks.from_stream(torch.arange(80), seq_len=8)  # 10 blocks
    source = ShuffledBatchSource(blocks, batch_size=5, seed=0)
    firsts = [b[:, 0] for b in (source.next_batch(), source.next_batch())]
    assert sorted(torch.cat(firsts).tolist()) == [i * 8 for i in range(10)]
    source.next_batch()
    assert source.epoch == 1


def test_fixed_batches_are_stream_order_and_capped():
    blocks = PackedBlocks.from_stream(torch.arange(80), seq_len=8)
    batches = fixed_batches(blocks, batch_size=3, max_batches=2)
    assert len(batches) == 2 and batches[0][0, 0] == 0 and batches[1][0, 0] == 24


def test_too_little_data_is_rejected():
    blocks = PackedBlocks.from_stream(torch.arange(16), seq_len=8)
    with pytest.raises(ValueError):
        ShuffledBatchSource(blocks, batch_size=3, seed=0)
    with pytest.raises(ValueError):
        fixed_batches(blocks, batch_size=3, max_batches=1)


def test_memory_mapped_cache_blocks_match_in_memory_blocks(tmp_path):
    import numpy as np

    stream = torch.randint(0, 1000, (8 * 12,), dtype=torch.int32)
    np.save(tmp_path / "blocks.npy", stream.numpy().astype(np.uint16).reshape(-1, 8))
    mapped = PackedBlocks(np.load(tmp_path / "blocks.npy", mmap_mode="r"))
    in_memory = PackedBlocks.from_stream(stream, seq_len=8)

    indices = torch.tensor([11, 0, 5, 5])
    assert len(mapped) == len(in_memory) == 12 and mapped.seq_len == 8
    assert mapped.get(indices).dtype == torch.long
    assert torch.equal(mapped.get(indices), in_memory.get(indices))

    a, b = (ShuffledBatchSource(x, 4, seed=3) for x in (mapped, in_memory))
    for _ in range(5):  # batching (and epoch reshuffling) is identical for both backings
        assert torch.equal(a.next_batch(), b.next_batch())


def test_blocks_must_be_2d_with_room_for_a_target():
    with pytest.raises(ValueError):
        PackedBlocks(torch.arange(8))
    with pytest.raises(ValueError):
        PackedBlocks(torch.zeros(4, 1))
