"""
Minimal flash-attn smoke tests.

Run with:
uv run --extra dev --extra fsdp pytest -q tests/backends/skyrl_train/gpu/test_flash_attn_smoke.py
"""

import math

import pytest


def _require_flash_attn():
    torch = pytest.importorskip("torch")
    flash_attn = pytest.importorskip("flash_attn")
    if not torch.cuda.is_available():
        pytest.skip("flash-attn smoke test requires CUDA")

    from flash_attn import flash_attn_func
    from flash_attn.bert_padding import pad_input, unpad_input

    return torch, flash_attn, flash_attn_func, pad_input, unpad_input


def test_flash_attn_cuda_forward_matches_reference():
    torch, flash_attn, flash_attn_func, _, _ = _require_flash_attn()

    batch_size = 2
    seqlen = 16
    num_heads = 4
    head_dim = 64

    torch.manual_seed(0)
    q = torch.randn(batch_size, seqlen, num_heads, head_dim, device="cuda", dtype=torch.float16, requires_grad=True)
    k = torch.randn(batch_size, seqlen, num_heads, head_dim, device="cuda", dtype=torch.float16, requires_grad=True)
    v = torch.randn(batch_size, seqlen, num_heads, head_dim, device="cuda", dtype=torch.float16, requires_grad=True)

    out = flash_attn_func(q, k, v, dropout_p=0.0, causal=True)
    assert out.shape == q.shape
    assert torch.isfinite(out).all()
    assert flash_attn.__version__

    scale = 1.0 / math.sqrt(head_dim)
    scores = torch.einsum("bqhd,bkhd->bhqk", q.float(), k.float()) * scale
    causal_mask = torch.triu(torch.ones(seqlen, seqlen, device=q.device, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(causal_mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    expected = torch.einsum("bhqk,bkhd->bqhd", probs, v.float())

    assert torch.allclose(out.float(), expected, atol=5e-2, rtol=5e-2)

    loss = out.square().mean()
    loss.backward()
    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None


def test_flash_attn_bert_padding_round_trip():
    torch, _, _, pad_input, unpad_input = _require_flash_attn()

    values = torch.arange(1, 11, device="cuda", dtype=torch.float16).view(2, 5, 1)
    attention_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 1, 1]], device="cuda", dtype=torch.int32)

    packed, indices, cu_seqlens, max_seqlen, _ = unpad_input(values, attention_mask=attention_mask)
    restored = pad_input(packed, indices=indices, batch=values.size(0), seqlen=values.size(1))

    assert packed.shape == (7, 1)
    assert cu_seqlens.tolist() == [0, 3, 7]
    assert max_seqlen == 4
    assert torch.equal(restored[attention_mask.bool()], values[attention_mask.bool()])
