# vLLM Artifact Index

This directory contains raw output files from MR-V100/CoreX vLLM experiments.
The markdown reports in [`../`](../) are the human-readable conclusions; this
directory is the evidence trail.

## File Types

| File type | Meaning |
|---|---|
| `.json` | Structured benchmark result, failure record, or generated summary. These are the primary machine-readable artifacts. |
| `.log` | Console/server/client log captured during the corresponding run. Use these to inspect startup config, stack traces, and runtime messages. |

## Summary Files

| File | What it summarizes |
|---|---|
| [`07_variants_summary.json`](07_variants_summary.json) | Eager/CUDA-Graph and prefix-cache variant smoke-test summary. |
| [`09_scale_ladder_summary.json`](09_scale_ladder_summary.json) | Small-to-mid Qwen2.5 scale ladder summary. |
| [`10_scale_and_awq_summary.json`](10_scale_and_awq_summary.json) | Scale-ladder plus 32B-AWQ result summary. |
| [`11_scale_awq_72b_summary.json`](11_scale_awq_72b_summary.json) | 72B-AWQ boundary attempt summary. |
| [`15_server_benchmark_qwen32b_awq_marlin_summary.json`](15_server_benchmark_qwen32b_awq_marlin_summary.json) | Initial 32B-AWQ server benchmark summary. |
| [`16_server_benchmark_qwen32b_awq_marlin_full_curve_summary.json`](16_server_benchmark_qwen32b_awq_marlin_full_curve_summary.json) | Fuller 32B-AWQ server concurrency curve summary. |
| [`17_prefix_distribution_decode128_summary.json`](17_prefix_distribution_decode128_summary.json) | Shared-prefix versus random-prefix decode-128 benchmark summary. |
| [`18_speculative_decoding_feasibility_summary.json`](18_speculative_decoding_feasibility_summary.json) | First speculative decoding feasibility summary. |
| [`19_qwen_small_speculative_decode128_summary.json`](19_qwen_small_speculative_decode128_summary.json) | Qwen2.5-3B target plus 0.5B draft speculative decoding probe summary. |
| [`20_prefix_cache_radix_workload_summary.json`](20_prefix_cache_radix_workload_summary.json) | Grouped/random/shared prefix-cache workload summary. |
| [`21_prefix_cache_sweep_summary.json`](21_prefix_cache_sweep_summary.json) | Output-length, prompt-length, and prefix-group sweep summary. |

## Individual Run Groups

| Pattern | What the files are |
|---|---|
| `05_vllm_offline_smoke.json` | First offline `vllm.LLM.generate` smoke result. |
| `06_vllm_offline_smoke_file.json` | Smoke result from a real Python file, avoiding multiprocessing-from-stdin issues. |
| `07_variant_*.json` | Variant runs comparing eager, CUDA Graph, and prefix-cache settings. |
| `08_scale_Qwen2.5-*.json` | Early scale ladder for Qwen2.5 0.5B and 1.5B. |
| `09_extreme_Qwen2.5-*.json` | Larger scale probes for Qwen2.5 3B, 7B, and 14B. |
| `10_awq*.json` and `10_awq*.log` | Qwen2.5-32B-AWQ loading/generation probes, including AWQ-Marlin plus CUDA Graph. |
| `12_72b_awq_marlin_eager_gpu098_precheck_fail.*` | 72B-AWQ AWQ-Marlin precheck failure evidence at high GPU memory utilization. |
| `13_72b_awq_marlin_eager_gpu090_oom.*` | 72B-AWQ AWQ-Marlin OOM evidence at lower GPU memory utilization. |
| `14_72b_awq_eager_gpu090_oom.*` | 72B-AWQ non-Marlin/eager OOM evidence. |
| `15_server_benchmark_qwen32b_awq_marlin_*.json` | 32B-AWQ server benchmark result JSONs for random/shared prefix and speculative variants. |
| `15_server_qwen32b_awq_marlin_*.log` | Server logs corresponding to the 32B-AWQ server benchmark runs. |
| `15_server_benchmark_qwen3b_*.json` | Qwen2.5-3B server benchmark JSONs for baseline, speculative, and prefix-cache workloads. |
| `15_server_qwen3b_*.log` | Server logs corresponding to Qwen2.5-3B server benchmark runs. |
| `15_server_benchmark_qwen05b_*.json` and `15_server_qwen05b_*.log` | Qwen2.5-0.5B baseline/self-draft speculative benchmark evidence. |
| `15_server_benchmark_qwen14b_*.json` and `15_server_qwen14b_*.log` | Qwen2.5-14B baseline/draft benchmark evidence. |
| `21_sweep_qwen3b_g4_trueprefix_decode*_cache_*.log` | Output-length sweep logs for Qwen2.5-3B with 4 prefix groups. |
| `22_sweep_qwen3b_g4_trueprefix_prompt*_cache_*.log` | Prompt-length sweep logs for Qwen2.5-3B with 4 prefix groups. |
| `23_sweep_qwen3b_g*_trueprefix_prompt512_decode32_c16_r4_cache_*.log` | Prefix-group locality sweep logs for Qwen2.5-3B. |

## How To Use This Directory

1. Read the matching report in [`../`](../) first.
2. Open the summary JSON named in that report.
3. Use the raw `.json` and `.log` files only when checking a specific run,
   failure mode, or server-side configuration.

