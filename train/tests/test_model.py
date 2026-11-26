"""CPU smoke tests: quantizers, training step, export round trip, reference parity."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer, trainers
from tokenizers import pre_tokenizers as pt

from trit_train import fileformat as ff
from trit_train.export import build, prune_dead_units, quantize_embedding
from trit_train.model import BitLinear, Config, Transformer, activation_quant, weight_quant
from trit_train.reference import Reference
from trit_train.tokenizer import EOS, build_empty, merges, token_bytes

TINY = Config(vocab_size=320, dim=64, n_layers=2, n_heads=2, ffn_dim=128, max_seq_len=64)

CORPUS = [
    "Once upon a time, there was a little dog named Max. Max liked to run.",
    'Lily saw a big red ball. She said, "Can I play?" Her mom smiled.',
    "The sun was warm and the birds sang. Tom and Sue went to the park.",
] * 20


@pytest.fixture(scope="module")
def tok() -> Tokenizer:
    t = build_empty()
    trainer = trainers.BpeTrainer(
        vocab_size=TINY.vocab_size,
        special_tokens=[EOS],
        initial_alphabet=pt.ByteLevel.alphabet(),
    )
    t.train_from_iterator(CORPUS, trainer=trainer)
    return t


def test_weight_quant_is_ternary_with_absmean_scale():
    w = torch.randn(64, 32)
    t, s = weight_quant(w)
    assert set(t.unique().tolist()) <= {-1.0, 0.0, 1.0}
    assert torch.isclose(s, w.abs().mean())


def test_activation_quant_per_token_range():
    x = torch.randn(3, 5, 16) * torch.tensor([0.01, 1.0, 100.0]).view(3, 1, 1)
    q, s = activation_quant(x)
    assert q.abs().max() <= 127
    assert torch.all(q.abs().amax(-1) == 127)  # the absmax element maps to +-127
    assert torch.allclose(q / s, x, atol=(x.abs().amax(-1, keepdim=True) / 127).max().item())


def test_bitlinear_straight_through_gradients():
    lin = BitLinear(16, 8)
    x = torch.randn(4, 16, requires_grad=True)
    lin(x).sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    assert lin.weight.grad is not None and lin.weight.grad.abs().sum() > 0


def test_training_steps_reduce_loss():
    torch.manual_seed(0)
    for ternary in (True, False):
        m = Transformer(Config(**{**TINY.to_dict(), "ternary": ternary}))
        opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
        x = torch.randint(0, TINY.vocab_size, (4, 32))
        first = None
        for _ in range(30):
            _, loss = m(x, x)
            opt.zero_grad()
            loss.backward()
            opt.step()
            first = first if first is not None else loss.item()
        assert loss.item() < first * 0.8, (ternary, first, loss.item())


def test_base3_roundtrip():
    rng = np.random.default_rng(0)
    for cols in (1, 4, 5, 6, 383, 384):
        t = rng.integers(-1, 2, size=(3, cols)).astype(np.int8)
        packed = ff.pack_base3(t)
        assert packed.shape == (3, -(-cols // 5))
        assert packed.max() < 243
        assert np.array_equal(ff.unpack_base3(packed, cols), t)


def test_tokenizer_export_tables(tok):
    tb = token_bytes(tok)
    assert len(tb) == tok.get_vocab_size()
    assert tb[tok.token_to_id(EOS)] == EOS.encode()
    ids = tok.encode("Once upon a time").ids
    assert b"".join(tb[i] for i in ids) == b"Once upon a time"
    for a, b, m in merges(tok):
        assert tb[a] + tb[b] == tb[m]


def test_export_roundtrip_and_reference_parity(tok, tmp_path):
    torch.manual_seed(1)
    model = Transformer(TINY).eval()
    path = tmp_path / "tiny.trit"
    f = build(model, tok)
    ff.write(path, f)
    g = ff.read(path)
    assert (g.dim, g.n_layers, g.vocab_size, g.eos_id) == (64, 2, 320, f.eos_id)
    assert g.tokens == f.tokens and [tuple(m) for m in g.merges] == f.merges
    for name, t in f.tensors.items():
        assert np.array_equal(np.asarray(g.tensors[name].data), np.asarray(t.data)), name
        assert g.tensors[name].scale == pytest.approx(t.scale)

    # The reference over the file must match the training model's own eval
    # forward once the model's embedding is replaced by its int8 version.
    q, s = quantize_embedding(model.embed.weight.detach())
    with torch.no_grad():
        model.embed.weight.copy_(torch.from_numpy(q).float() * torch.from_numpy(s)[:, None])
    idx = torch.randint(0, TINY.vocab_size, (2, 24))
    with torch.no_grad():
        want = model(idx)
    got = Reference(g).forward(idx)
    assert torch.allclose(got, want, atol=1e-4, rtol=1e-4), (got - want).abs().max()


def test_pruning_dead_units_changes_no_output(tok, tmp_path):
    torch.manual_seed(2)
    model = Transformer(TINY).eval()
    f = build(model, tok)
    # Switch some units off the way training does: a whole w_up row of zeros.
    # Unit 9 of layer 0 also keeps a live w_down column, which must not matter.
    dead = {0: [3, 9, 100], 1: [0, 127]}
    for layer, units in dead.items():
        f.tensors[f"layers.{layer}.w_up"].data[units] = 0
    full = tmp_path / "full.trit"
    ff.write(full, f)
    assert ff.version_for(f) == 1

    assert prune_dead_units(f) == [3, 2]
    pruned = tmp_path / "pruned.trit"
    ff.write(pruned, f)
    g = ff.read(pruned)
    assert ff.version_for(g) == 2 and g.ffn_dim == TINY.ffn_dim
    assert ff.ffn_widths(g) == [125, 126]
    assert g.tensors["layers.0.w_down"].shape == (TINY.dim, 125)
    assert g.tensors["layers.1.ffn_sub_norm"].shape == (126,)
    assert full.stat().st_size > pruned.stat().st_size
    # The header says version 2, so a v1-only reader refuses the file.
    assert int.from_bytes(pruned.read_bytes()[4:8], "little") == 2

    idx = torch.randint(0, TINY.vocab_size, (2, 24))
    ref_full, ref_pruned = Reference(ff.read(full)), Reference(g)
    want, got = ref_full.forward(idx), ref_pruned.forward(idx)
    assert torch.allclose(got, want, atol=1e-5, rtol=1e-5), (got - want).abs().max()
    # The zero share is still counted over the trained width.
    # (Equal up to float summation order in the sub-norm, which may round differently.)
    assert ref_pruned.ffn_total == ref_full.ffn_total
    assert abs(ref_pruned.ffn_zeros - ref_full.ffn_zeros) <= ref_full.ffn_total // 1000
    assert prune_dead_units(g) == [0, 0], "pruning twice finds nothing new"
