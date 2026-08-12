# MR-V100 Qwen small-model speculative decoding probe

This note changes the model family after the earlier 14B/32B speculative
decoding failures. The goal is to test the DSpark-style idea on a model pair
that avoids the previous tokenizer/vocabulary mismatch.

The new candidate pair is:

```text
target: Qwen2.5-3B-Instruct
draft: Qwen2.5-0.5B-Instruct
vocab_size: 151,936 for both
```

This is not a production acceleration target. It is a controlled feasibility
probe: if this pair cannot run speculative decoding, then the blocker is deeper
than just model memory size or tokenizer mismatch.

## Environment

```text
device: Iluvatar MR-V100 32 GiB
CoreX driver: 4.4.0
vLLM: 0.17.0
mode: eager
endpoint: /v1/completions, streaming
prompt target: 256 tokens
max output: 128 tokens
prompt mode: random
concurrency: 1, 4, 8, 16
```

## Experiment 1: Qwen2.5-3B baseline

```text
model: Qwen2.5-3B-Instruct
max_model_len: 512
max_num_seqs: 16
speculative decoding: disabled
```

| Concurrency | Success | Output tok/s | TTFT p50 | TPOT p50 | Total p50 |
|---:|---:|---:|---:|---:|---:|
| 1 | 1/1 | 48.6 | 0.447s | 17.2ms | 2.630s |
| 4 | 4/4 | 198.7 | 0.166s | 18.9ms | 2.568s |
| 8 | 8/8 | 375.7 | 0.210s | 19.7ms | 2.713s |
| 16 | 16/16 | 674.9 | 0.335s | 21.1ms | 3.019s |

This is a healthy non-speculative serving baseline. It also gives a strong
control: if speculative decoding fails, the basic vLLM serving stack is not the
problem.

## Experiment 2: Qwen2.5-3B target + Qwen2.5-0.5B draft

```text
target: Qwen2.5-3B-Instruct
draft: Qwen2.5-0.5B-Instruct
method: draft_model
num_speculative_tokens: 4
max_model_len: 512
max_num_seqs: 16
```

Result:

| Concurrency | Success | Output tok/s | Notes |
|---:|---:|---:|---|
| 1 | 1/1 | 0.0 | First request returned no generated token |
| 4 | 0/4 | 0.0 | EngineCore already dead |
| 8 | 0/8 | 0.0 | EngineCore already dead |
| 16 | 0/16 | 0.0 | EngineCore already dead |

Root cause:

```text
RuntimeError: The expanded size of the tensor (896) must match the existing
size (2048) at non-singleton dimension 1.
```

Interpretation:

```text
Qwen2.5-0.5B hidden_size = 896
Qwen2.5-3B hidden_size = 2048
```

Although the tokenizer vocabulary matches, this vLLM/CoreX draft-model path
uses an EAGLE-like proposer that tries to move hidden states between target and
draft shapes. The hidden-size mismatch makes the pair unusable in this runtime.

## Experiment 3: Qwen2.5-0.5B baseline

```text
model: Qwen2.5-0.5B-Instruct
max_model_len: 512
max_num_seqs: 16
speculative decoding: disabled
```

| Concurrency | Success | Output tok/s | TTFT p50 | TPOT p50 | Total p50 |
|---:|---:|---:|---:|---:|---:|
| 1 | 1/1 | 68.7 | 0.370s | 11.7ms | 1.858s |
| 4 | 4/4 | 302.7 | 0.145s | 12.1ms | 1.677s |
| 8 | 8/8 | 598.4 | 0.111s | 12.5ms | 1.697s |
| 16 | 16/16 | 1064.7 | 0.162s | 13.7ms | 1.904s |

This confirms that small-model serving itself is fast and stable on MR-V100.

## Experiment 4: Qwen2.5-0.5B self-draft sanity check

```text
target: Qwen2.5-0.5B-Instruct
draft: Qwen2.5-0.5B-Instruct
method: draft_model
num_speculative_tokens: 4
```

Result:

| Concurrency | Success | Output tok/s | Notes |
|---:|---:|---:|---|
| 1 | 1/1 | 2.1 | First request streamed only 1 token |
| 4 | 0/4 | 0.0 | EngineCore already dead |
| 8 | 0/8 | 0.0 | EngineCore already dead |
| 16 | 0/16 | 0.0 | EngineCore already dead |

Root cause:

```text
RuntimeError: Expected query.is_contiguous() to be true, but got false.
ixformer::infer::vllm_paged_attention_mtp
/root/code/ixformer/src/ixformer/infer/paged_attention.cu:744
```

This is the same CoreX ixformer MTP paged-attention assertion seen in the
earlier 32B ngram speculative probe.

## Interpretation

- Changing to a same-vocab model pair fixed the earlier tokenizer/vocab blocker,
  but exposed a second blocker: the local draft-model path expects hidden-state
  compatibility between target and draft.
- Even a same-model self-draft sanity check is unstable because the speculative
  path reaches `ixformer::infer::vllm_paged_attention_mtp`, which rejects a
  non-contiguous query tensor.
- The DSpark/Prequal idea remains useful as a research lens, but this CoreX
  vLLM image does not currently provide a stable speculative decoding substrate.
- The strongest reliable result is still non-speculative serving: Qwen2.5-3B
  reaches about 674.9 output tok/s at concurrency 16, and Qwen2.5-0.5B reaches
  about 1064.7 output tok/s at concurrency 16.

## Practical next steps

1. Reduce the `vllm_paged_attention_mtp` contiguous-query failure to a minimal
   CoreX/vLLM repro.
2. Try the same target/draft pair on a standard NVIDIA CUDA vLLM image to
   separate vLLM semantics from CoreX backend behavior.
3. Look for a model family where the draft path is officially supported by
   vLLM on this runtime, such as a packaged EAGLE/Medusa/MTP model rather than
   arbitrary Qwen base models.
4. Keep MR-V100 work focused on non-speculative serving optimization until the
   speculative substrate is stable.

## Artifacts

```text
docs/vllm/artifacts/19_qwen_small_speculative_decode128_summary.json
docs/vllm/artifacts/15_server_benchmark_qwen3b_baseline_eager_random_prefix_decode128_c1_4_8_16.json
docs/vllm/artifacts/15_server_benchmark_qwen3b_draft05b_spec_eager_random_prefix_decode128_c1_4_8_16.json
docs/vllm/artifacts/15_server_benchmark_qwen05b_baseline_eager_random_prefix_decode128_c1_4_8_16.json
docs/vllm/artifacts/15_server_benchmark_qwen05b_selfdraft_spec_eager_random_prefix_decode128_c1_4_8_16.json
docs/vllm/artifacts/15_server_qwen3b_draft05b_spec_eager_random_prefix_decode128_c1_4_8_16.log
docs/vllm/artifacts/15_server_qwen05b_selfdraft_spec_eager_random_prefix_decode128_c1_4_8_16.log
```

## Resume framing

```text
Tested DSpark-inspired speculative decoding on a same-vocab Qwen2.5 small-model
pair on Iluvatar MR-V100/CoreX: Qwen2.5-3B baseline reached 674.9 output tok/s
at concurrency 16, but Qwen2.5-3B + 0.5B draft failed due to hidden-size
mismatch in vLLM's EAGLE-like draft proposer, and even 0.5B self-draft failed
inside CoreX ixformer MTP paged attention with a non-contiguous query assertion.
```
