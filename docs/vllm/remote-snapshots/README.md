# MR-V100 remote snapshots

These folders preserve the remote state from the Iluvatar MR-V100 instance
before shutdown on 2026-08-13.

The snapshots intentionally exclude model weights, Hugging Face cache,
virtualenv package trees, shell history, SSH material, and credentials. They
preserve experiment data, runner scripts, logs, reports, manifests, selected
environment details, and SHA-256 checksums.

## Snapshot folders

### `mrv100_vllm_probe_snapshot_20260813_100522`

Source: `/data/src/vllm-mrv100-probe`

Contents:

- `scripts/`: benchmark/probe runner scripts used on the remote machine.
- `artifacts/`: all JSON outputs and PID files left by the vLLM experiments.
- `logs/`: all server, benchmark, sweep, and runner logs.
- `ENVIRONMENT.txt`: selected OS, CPU, disk, GPU, Python, vLLM, and package
  metadata.
- `MANIFEST.txt`: file tree and count summary.
- `SHA256SUMS.txt`: checksums for preserved files.

Local size: about 7.4 MB, 258 files.

Important files/directories:

| Path | What it is |
|---|---|
| `ENVIRONMENT.txt` | Shutdown-time environment capture: OS, kernel, CPU, disk, GPU, Python, vLLM, Torch, Transformers, and related package metadata. |
| `MANIFEST.txt` | Full file tree and count/size summary for this snapshot. Use this when you need the exact list of every preserved file. |
| `SHA256SUMS.txt` | SHA-256 checksums for integrity verification. |
| `scripts/` | Remote copies of benchmark runners used to generate the vLLM evidence. |
| `artifacts/` | JSON results and PID files from offline probes, scale probes, server benchmarks, speculative-decoding attempts, and prefix-cache sweeps. |
| `logs/` | Console/server/client logs for the same runs; useful for startup config, backend messages, OOM traces, and stack traces. |

### `mrv100_misc_probe_snapshot_20260813_100811`

Sources:

- `/data/src/mrv100-hw-probe`
- `/data/src/sglang-omni-corex-probe/{logs,reports}`
- `/data/src/llaisys/{docs,reports,tools,scripts}`
- `/data/src/*.sh` probe scripts

Contents:

- MR-V100 hardware/interface probes.
- SGLang/SGLang-Omni feasibility logs and reports.
- LLAISYS MR-V100 assignment/hardware notes copied from the remote checkout.
- Top-level remote probe scripts.
- `MANIFEST.txt` and `SHA256SUMS.txt`.

Local size: about 724 KB, 41 files.

Important files/directories:

| Path | What it is |
|---|---|
| `MANIFEST.txt` | Full file tree and count/size summary for this misc snapshot. |
| `SHA256SUMS.txt` | SHA-256 checksums for integrity verification. |
| `top-level-scripts/` | One-off remote probe scripts used for hardware/interface reconnaissance. |
| `data-src/mrv100-hw-probe/` | MR-V100 hardware, programming interface, bandwidth/FLOPS, and deep interface probe outputs. |
| `data-src/sglang-omni-corex-probe/logs/` | SGLang/SGLang-Omni install/import/server attempt logs on CoreX. |
| `data-src/sglang-omni-corex-probe/reports/` | Human-readable SGLang-Omni compatibility report and community issue draft. |
| `data-src/llaisys/docs/` | Remote LLAISYS hardware notes copied before shutdown. |
| `data-src/llaisys/reports/` | Assignment 4 reports for NVIDIA 4090D and Iluvatar MR-V100/CoreX. |
| `data-src/llaisys/tools/` | Hardware probing helper copied from the remote LLAISYS checkout. |

## Packaging scripts

The scripts used to create these snapshots are preserved here:

- `tools/vllm/create_mrv100_remote_snapshot.sh`
- `tools/vllm/create_mrv100_misc_snapshot.sh`

## Sensitive-data check

Before committing, the snapshots were scanned for common password, bearer-token, GitHub-token, Hugging Face-token, and Notebook-token markers. No matches were found outside this README statement.
