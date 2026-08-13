# vLLM Tooling Index

These scripts were used on the MR-V100/CoreX rental machine to generate the
experiment evidence under [`../../docs/vllm`](../../docs/vllm).

They assume a Linux host with CoreX libraries, vLLM installed, models under
`/data/models`, and a writable work directory such as
`/data/src/vllm-mrv100-probe`.

## Scripts

| File | Purpose |
|---|---|
| [`mrv100_offline_smoke.py`](mrv100_offline_smoke.py) | Minimal offline vLLM smoke test using `vllm.LLM.generate`; safe for Python spawn multiprocessing. |
| [`run_mrv100_vllm_variant.py`](run_mrv100_vllm_variant.py) | Runs one offline variant and records load/generation timing for eager/CUDA-Graph and prefix-cache settings. |
| [`run_mrv100_vllm_variants.sh`](run_mrv100_vllm_variants.sh) | Wrapper that runs the standard variant set and writes a `07_variants_summary.json`. |
| [`run_mrv100_vllm_scale_probe.py`](run_mrv100_vllm_scale_probe.py) | Offline model scale probe: loads a model, records config, runs warm/repeat generation, and captures `ixsmi` snapshots. |
| [`run_mrv100_vllm_scale_ladder.sh`](run_mrv100_vllm_scale_ladder.sh) | Wrapper for the Qwen2.5 scale ladder across 0.5B, 1.5B, 3B, and 7B. |
| [`run_mrv100_vllm_awq_probe.sh`](run_mrv100_vllm_awq_probe.sh) | Wrapper for AWQ/AWQ-Marlin model probes such as Qwen2.5-32B-Instruct-AWQ. |
| [`run_mrv100_vllm_server_benchmark.py`](run_mrv100_vllm_server_benchmark.py) | Async OpenAI-compatible `/v1/completions` benchmark client; measures TTFT, TPOT, total latency, output throughput, and success rate. |
| [`run_mrv100_vllm_server_benchmark.sh`](run_mrv100_vllm_server_benchmark.sh) | Starts a vLLM server with CoreX environment variables, runs the benchmark client, saves logs/results, and tears down the server. |
| [`create_mrv100_remote_snapshot.sh`](create_mrv100_remote_snapshot.sh) | Packages `/data/src/vllm-mrv100-probe` scripts, artifacts, logs, environment data, manifest, and checksums before instance shutdown. |
| [`create_mrv100_misc_snapshot.sh`](create_mrv100_misc_snapshot.sh) | Packages hardware probes, SGLang-Omni/CoreX logs, LLAISYS reports/tools, top-level probe scripts, manifest, and checksums. |

## Output Layout

By default the run scripts write under `$WORK`, usually
`/data/src/vllm-mrv100-probe`:

| Remote path | Contents |
|---|---|
| `$WORK/artifacts` | JSON benchmark outputs, PID files, and generated summaries. |
| `$WORK/logs` | Server logs, benchmark logs, and command transcripts. |
| `$WORK/scripts` | Script copies used on the remote machine. |
| `$WORK/reports` | Optional generated report files. |

The committed copies live in [`../../docs/vllm/artifacts`](../../docs/vllm/artifacts)
and [`../../docs/vllm/remote-snapshots`](../../docs/vllm/remote-snapshots).

