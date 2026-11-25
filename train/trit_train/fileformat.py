"""Reader and writer for the `.trit` model file. See docs/format.md for the layout.

Everything is little-endian. A 64-byte header, the tokenizer, a tensor table,
then tensor data, each tensor starting on a 64-byte boundary.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

import numpy as np

MAGIC = b"TRIT"
VERSION = 1
ALIGN = 64

DT_F32, DT_I8, DT_TERNARY = 0, 1, 2


@dataclass
class Tensor:
    name: str
    dtype: int
    shape: tuple[int, ...]
    data: np.ndarray  # f32 / i8 values, or for ternary the trits as int8 in {-1,0,1}
    scale: float = 1.0


@dataclass
class TritFile:
    vocab_size: int
    dim: int
    n_layers: int
    n_heads: int
    ffn_dim: int
    max_seq_len: int
    rope_theta: float
    norm_eps: float
    eos_id: int
    tokens: list[bytes]
    merges: list[tuple[int, int, int]]
    tensors: dict[str, Tensor] = field(default_factory=dict)


def pack_base3(trits: np.ndarray) -> np.ndarray:
    """(rows, cols) trits -> (rows, ceil(cols/5)) bytes: 5 columns per byte, little-endian."""
    rows, cols = trits.shape
    nb = -(-cols // 5)
    d = np.ones((rows, nb * 5), dtype=np.uint8)  # padding is trit 0 = digit 1
    d[:, :cols] = (trits + 1).astype(np.uint8)
    d = d.reshape(rows, nb, 5).astype(np.uint16)
    v = d[..., 0] + 3 * d[..., 1] + 9 * d[..., 2] + 27 * d[..., 3] + 81 * d[..., 4]
    return v.astype(np.uint8)


def unpack_base3(packed: np.ndarray, cols: int) -> np.ndarray:
    v = packed.astype(np.int16)
    digits = []
    for _ in range(5):
        digits.append(v % 3)
        v = v // 3
    d = np.stack(digits, axis=-1).reshape(packed.shape[0], -1)
    return (d[:, :cols] - 1).astype(np.int8)


def _header(f: TritFile, n_tensors: int) -> bytes:
    return struct.pack(
        "<4s9I2f3I",
        MAGIC,
        VERSION,
        f.vocab_size,
        f.dim,
        f.n_layers,
        f.n_heads,
        f.ffn_dim,
        f.max_seq_len,
        0,  # reserved (n_kv_heads in a later version)
        0,  # reserved
        f.rope_theta,
        f.norm_eps,
        f.eos_id,
        len(f.merges),
        n_tensors,
    ).ljust(ALIGN, b"\0")


def _tensor_bytes(t: Tensor) -> bytes:
    if t.dtype == DT_F32:
        return np.ascontiguousarray(t.data, dtype="<f4").tobytes()
    if t.dtype == DT_I8:
        return np.ascontiguousarray(t.data, dtype=np.int8).tobytes()
    if t.dtype == DT_TERNARY:
        return pack_base3(np.asarray(t.data, dtype=np.int8).reshape(t.shape)).tobytes()
    raise ValueError(t.dtype)


def write(path, f: TritFile) -> int:
    tensors = list(f.tensors.values())
    tok = bytearray()
    for b in f.tokens:
        assert len(b) < 256
        tok += bytes([len(b)]) + b
    for a, b, m in f.merges:
        tok += struct.pack("<3H", a, b, m)
    blobs = [_tensor_bytes(t) for t in tensors]

    def table(offsets: list[int]) -> bytes:
        out = bytearray()
        for t, off, blob in zip(tensors, offsets, blobs, strict=True):
            name = t.name.encode()
            out += bytes([len(name)]) + name + bytes([t.dtype, len(t.shape)])
            out += struct.pack(f"<{len(t.shape)}I", *t.shape)
            out += struct.pack("<fQQ", t.scale, off, len(blob))
        return bytes(out)

    head = _header(f, len(tensors)) + bytes(tok)
    table_len = len(table([0] * len(tensors)))
    pos = len(head) + table_len
    offsets = []
    for blob in blobs:
        pos = -(-pos // ALIGN) * ALIGN
        offsets.append(pos)
        pos += len(blob)
    body = bytearray(head + table(offsets))
    for off, blob in zip(offsets, blobs, strict=True):
        body += b"\0" * (off - len(body))
        body += blob
    with open(path, "wb") as fh:
        fh.write(body)
    return len(body)


def read(path) -> TritFile:
    with open(path, "rb") as fh:
        buf = fh.read()
    (
        magic,
        version,
        vocab,
        dim,
        layers,
        heads,
        ffn,
        max_seq,
        _r0,
        _r1,
        theta,
        eps,
        eos,
        n_merges,
        n_tensors,
    ) = struct.unpack_from("<4s9I2f3I", buf, 0)
    if magic != MAGIC or version != VERSION:
        raise ValueError(f"not a trit v{VERSION} file")
    pos = ALIGN
    tokens = []
    for _ in range(vocab):
        n = buf[pos]
        tokens.append(bytes(buf[pos + 1 : pos + 1 + n]))
        pos += 1 + n
    merges = [struct.unpack_from("<3H", buf, pos + 6 * i) for i in range(n_merges)]
    pos += 6 * n_merges
    f = TritFile(vocab, dim, layers, heads, ffn, max_seq, theta, eps, eos, tokens, merges)
    for _ in range(n_tensors):
        n = buf[pos]
        name = buf[pos + 1 : pos + 1 + n].decode()
        pos += 1 + n
        dtype, ndim = buf[pos], buf[pos + 1]
        pos += 2
        shape = struct.unpack_from(f"<{ndim}I", buf, pos)
        pos += 4 * ndim
        scale, off, nbytes = struct.unpack_from("<fQQ", buf, pos)
        pos += 20
        raw = np.frombuffer(buf, dtype=np.uint8, count=nbytes, offset=off)
        if dtype == DT_F32:
            data = raw.view("<f4").reshape(shape)
        elif dtype == DT_I8:
            data = raw.view(np.int8).reshape(shape)
        else:
            rows, cols = shape
            data = unpack_base3(raw.reshape(rows, -1), cols)
        f.tensors[name] = Tensor(name, dtype, tuple(shape), data, scale)
    return f
