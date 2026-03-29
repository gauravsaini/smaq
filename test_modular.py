"""Lightweight modular smoke tests for the SMAQ package."""

from __future__ import annotations


def run_smoke_tests():
    try:
        import torch
    except ImportError:
        print("PyTorch is not installed in this environment; SMAQ smoke tests skipped.")
        return

    from smaq.block_vq import SMAQBlockVQ
    from smaq.capture import KVCaptureEngine
    from smaq.quantizer import SMAQQuantizer
    from smaq.score import compute_hybrid_attention
    from smaq.store import CompressedKVStore
    from smaq.triton_kernels import smaq_attention_scores
    from smaq.vllm_attn_backend import MODE_ACCUMULATE, MODE_ACTIVE, get_mode, set_mode

    # ------------------------------------------------------------------
    # Test 1: Scalar quantizer roundtrip
    # ------------------------------------------------------------------
    sigma = torch.eye(8)
    quantizer = SMAQQuantizer(dim=8, Sigma_q=sigma, bits=3, device=torch.device("cpu"))
    vectors = torch.randn(2, 3, 8)
    quantized = quantizer.quantize(vectors)
    restored = quantizer.dequantize(quantized)

    assert quantized.indices.shape[:2] == vectors.shape[:2]
    assert quantized.norms.shape == vectors.shape[:-1]
    assert restored.shape == vectors.shape
    print("test_scalar_quantizer_roundtrip ... OK")

    # ------------------------------------------------------------------
    # Test 2: Block VQ fit + roundtrip
    # ------------------------------------------------------------------
    vq = SMAQBlockVQ(head_dim=8, block_dim=8, n_centroids=16, c=5.0)
    cal_k = torch.randn(100, 8)
    cal_q = torch.randn(100, 8)
    vq.fit(cal_k, cal_q, kmeans_iters=5, seed=42)

    test_k = torch.randn(10, 8)
    bvq = vq.quantize(test_k)
    k_hat = vq.dequantize(bvq)

    assert bvq.indices.shape == (10, 1)  # 1 block for 8D / 8D
    assert k_hat.shape == (10, 8)

    # Pre-decoded centroids should differ from E-space centroids
    assert not torch.allclose(vq.centroids, vq.decoded_centroids, atol=1e-6)
    print("test_block_vq_fit_roundtrip ... OK")

    # ------------------------------------------------------------------
    # Test 3: Block VQ logit_mse is finite and positive
    # ------------------------------------------------------------------
    lmse = vq.logit_mse(cal_q[:50], cal_k[:50])
    assert lmse > 0, f"logit_mse should be positive, got {lmse}"
    assert lmse < float("inf"), "logit_mse should be finite"
    print("test_block_vq_logit_mse ... OK")

    # ------------------------------------------------------------------
    # Test 4: Capture + store + score pipeline
    # ------------------------------------------------------------------
    store = CompressedKVStore(
        head_dim=8,
        num_kv_heads=2,
        Sigma_q=sigma,
        key_bits=3,
        value_bits=2,
        value_group_size=4,
        device=torch.device("cpu"),
    )
    engine = KVCaptureEngine(store=store, ring_capacity=2, device=torch.device("cpu"))

    keys = torch.randn(5, 2, 8)
    values = torch.randn(5, 2, 8)
    engine.ingest_prefill(keys, values, 5)

    recent = engine.ring.peek()
    query = torch.randn(1, 2, 8)
    output = compute_hybrid_attention(
        query=query,
        store=store,
        recent_k=recent[0] if recent else None,
        recent_v=recent[1] if recent else None,
        num_query_heads=2,
    )
    assert output.shape == (1, 2, 8)
    print("test_capture_store_score_pipeline ... OK")

    # ------------------------------------------------------------------
    # Test 5: vLLM backend mode switching
    # ------------------------------------------------------------------
    set_mode(MODE_ACTIVE)
    assert get_mode() == MODE_ACTIVE
    set_mode(MODE_ACCUMULATE)
    assert get_mode() == MODE_ACCUMULATE
    print("test_vllm_backend_mode_switch ... OK")

    # ------------------------------------------------------------------
    # Test 6: Triton score path matches Torch reference when available
    # ------------------------------------------------------------------
    if torch.cuda.is_available():
        try:
            import triton  # noqa: F401
        except ImportError:
            print("test_triton_attention_scores ... SKIPPED (triton not installed)")
        else:
            device = torch.device("cuda")
            sigma_cuda = torch.eye(8, device=device)
            q_cuda = SMAQQuantizer(dim=8, Sigma_q=sigma_cuda, bits=3, device=device)
            key_cuda = torch.randn(32, 8, device=device)
            query_cuda = torch.randn(2, 8, device=device)
            qk_cuda = q_cuda.quantize(key_cuda)

            scores_ref = q_cuda.attention_score(query_cuda, qk_cuda, use_kernel=False)
            scores_kernel = smaq_attention_scores(q_cuda, query_cuda, qk_cuda)

            assert torch.allclose(scores_ref.float(), scores_kernel.float(), atol=1e-4, rtol=1e-4)
            print("test_triton_attention_scores ... OK")
    else:
        print("test_triton_attention_scores ... SKIPPED (cuda not available)")

    print("---")
    print("Modular Tests Passing")


if __name__ == "__main__":
    run_smoke_tests()
