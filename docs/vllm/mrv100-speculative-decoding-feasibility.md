# MR-V100 speculative decoding feasibility

This note records a first attempt to turn the DSpark/Prequal idea into concrete
MR-V100 experiments.

The goal was not to reproduce DSpark directly. Instead, the goal was to test the
nearest available serving mechanisms in the local vLLM/CoreX stack:

1. ngram speculative decoding on the known-good Qwen2.5-32B-Instruct-AWQ server.
2. draft-model speculative decoding with a smaller Qwen2.5 draft model.
3. a 14B non-speculative baseline to confirm the serving stack itself remains
   healthy.

## Why this matters

DSpark/Prequal focuses on speculative decoding serving, especially the problem
that verification work can waste effective batch capacity. That is exactly the
right mental model for future acceleration work, but it only becomes actionable
after the local stack answers three boring engineering questions:

- Does vLLM speculative decoding initialize on CoreX?
- Does the CoreX attention backend support the speculative execution path?
- Do we have a target/draft model pair with compatible tokenizers and enough
  memory headroom?

## Environment

```text
device: Iluvatar MR-V100 32 GiB
CoreX driver: 4.4.0
vLLM: 0.17.0
attention backend: ixformer / vLLM FlashAttention path
server API: OpenAI-compatible /v1/completions
```

## Experiment 1: 32B-AWQ ngram speculative decoding

Target:

```text
model: Qwen2.5-32B-Instruct-AWQ
quantization: awq_marlin
max_model_len: 512
max_num_seqs: 16
prompt target: 256 tokens
max output: 128 tokens
speculative_config:
  method: ngram
  num_speculative_tokens: 4
  prompt_lookup_max: 4
  prompt_lookup_min: 1
```

### CUDA Graph result

Status: failed before server readiness.

Root cause:

```text
RuntimeError: Expected query.is_contiguous() to be true, but got false.
ixformer::infer::vllm_paged_attention_mtp
/root/code/ixformer/src/ixformer/infer/paged_attention.cu:744
```

This happened during CUDA Graph capture. The important observation is that
speculative decoding triggered the MTP paged-attention path in ixformer, and
that path rejected a non-contiguous query tensor.

### Eager result

Status: server started, first request partially streamed, then EngineCore died.

| Concurrency | Success | Output tok/s | TTFT p50 | TPOT p50 | Notes |
|---:|---:|---:|---:|---:|---|
| 1 | 1/1 | 0.9 | 1.928s | 196.2ms | Only 2 output tokens before failure |
| 4 | 0/4 | 0.0 | n/a | n/a | EngineCore already dead |
| 8 | 0/8 | 0.0 | n/a | n/a | EngineCore already dead |
| 16 | 0/16 | 0.0 | n/a | n/a | EngineCore already dead |

The eager run failed with the same ixformer MTP contiguous-query assertion, but
at request execution time rather than graph-capture time. So this is not only a
CUDA Graph problem.

## Experiment 2: 14B target + 0.5B draft model

Target:

```text
target: Qwen2.5-14B-Instruct
draft: Qwen2.5-0.5B-Instruct
method: draft_model
num_speculative_tokens: 4
max_model_len: 512
max_num_seqs: 4
```

Status: rejected during vLLM config validation.

Root cause:

```text
Target and draft model should have the same vocabulary size.
Target model vocab_size=152064.
Draft model vocab_size=151936.
```

Local Qwen2.5 model vocabulary groups:

| Model | Vocab size |
|---|---:|
| Qwen2.5-0.5B-Instruct | 151,936 |
| Qwen2.5-1.5B-Instruct | 151,936 |
| Qwen2.5-3B-Instruct | 151,936 |
| Qwen2.5-7B-Instruct | 152,064 |
| Qwen2.5-14B-Instruct | 152,064 |
| Qwen2.5-32B-Instruct-AWQ | 152,064 |
| Qwen2.5-72B-Instruct-AWQ | 152,064 |

The only smaller same-vocab draft candidate for 14B/32B is 7B. On a single
32 GiB card, 14B+7B or 32B+7B leaves too little memory headroom to be a
practical first experiment.

## Experiment 3: 14B eager baseline

This run verifies that the vLLM serving stack is healthy when speculative
decoding is not enabled.

```text
model: Qwen2.5-14B-Instruct
mode: eager
max_model_len: 1024
max_num_seqs: 8
prompt target: 256 tokens
max output: 128 tokens
```

| Concurrency | Success | Output tok/s | TTFT p50 | TPOT p50 | Total p50 |
|---:|---:|---:|---:|---:|---:|
| 1 | 1/1 | 18.9 | 0.511s | 49.2ms | 6.758s |
| 4 | 4/4 | 69.6 | 0.485s | 54.0ms | 7.343s |
| 8 | 8/8 | 135.2 | 0.601s | 54.8ms | 7.557s |

The 14B baseline consumed about 31,944 MiB at the end of the run, leaving about
824 MiB free. That confirms the MR-V100 card is already near capacity even for
14B non-speculative serving at this workload shape.

## Interpretation

- DSpark's idea is relevant, but direct speculative decoding is not yet a
  clean acceleration lever on this MR-V100 image.
- ngram speculative decoding reaches the CoreX ixformer MTP paged-attention
  path and fails on a contiguous-query assertion. This is a backend compatibility
  problem, not just a benchmark tuning problem.
- draft-model speculative decoding needs a target/draft pair with matching
  vocabulary size. The locally available small Qwen2.5 draft models do not match
  the 14B/32B tokenizer family.
- The same server stack works without speculative decoding, so the failure is
  specific to speculative paths.

## Next experiment candidates

1. Find or build a same-vocab small draft model for the 152,064-vocab Qwen2.5
   family. This is the cleanest path to a real speculative decoding benchmark.
2. Try speculative decoding on another target/draft family where the small draft
   and larger target share tokenizer vocabulary and both fit in 32 GiB.
3. File or isolate the ixformer MTP contiguous-query issue with a minimal vLLM
   ngram speculative repro.
4. Continue non-speculative scheduling experiments: prefix distribution, long
   decode, chunked prefill, `max_num_seqs`, and `max_num_batched_tokens`.
5. Treat DSpark/Prequal as a scheduler research direction after a stable
   speculative path exists.

## Artifacts

```text
docs/vllm/artifacts/18_speculative_decoding_feasibility_summary.json
docs/vllm/artifacts/15_server_benchmark_qwen32b_awq_marlin_ngram_spec_eager_random_prefix_decode128_c1_4_8_16.json
docs/vllm/artifacts/15_server_qwen32b_awq_marlin_ngram_spec_random_prefix_decode128_c1_4_8_16.log
docs/vllm/artifacts/15_server_qwen32b_awq_marlin_ngram_spec_eager_random_prefix_decode128_c1_4_8_16.log
docs/vllm/artifacts/15_server_qwen14b_draft05b_spec_eager_random_prefix_decode128_c1_4.log
docs/vllm/artifacts/15_server_benchmark_qwen14b_baseline_eager_random_prefix_decode128_c1_4_8.json
```

## Resume framing

```text
Investigated speculative decoding feasibility on Iluvatar MR-V100/CoreX after
reading DSpark/Prequal: ngram speculative decoding on Qwen2.5-32B-AWQ exposed
an ixformer MTP paged-attention contiguous-query backend failure, while
draft-model speculative decoding was blocked by Qwen2.5 tokenizer vocabulary
mismatch; validated a stable 14B eager baseline at 135.2 output tok/s under
concurrency 8.
```
