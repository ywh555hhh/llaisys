#!/usr/bin/env python3
"""Run a minimal vLLM offline smoke test on MR-V100/CoreX.

The script is intentionally safe for Python's spawn multiprocessing mode. vLLM
V1 starts engine worker processes, so the LLM construction must live behind a
real ``if __name__ == "__main__"`` guard instead of being executed at import
time.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/data/models/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--out", default="/tmp/mrv100_vllm_offline_smoke.json")
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.45)
    args = parser.parse_args()

    result = {
        "model": args.model,
        "env": {
            key: os.environ.get(key)
            for key in [
                "VLLM_TARGET_DEVICE",
                "VLLM_ATTENTION_BACKEND",
                "VLLM_USE_V1",
                "VLLM_WORKER_MULTIPROC_METHOD",
                "VLLM_ENFORCE_CUDA_GRAPH",
            ]
        },
    }

    t0 = time.perf_counter()
    try:
        from vllm import LLM, SamplingParams

        t_import = time.perf_counter()
        llm = LLM(
            model=args.model,
            trust_remote_code=True,
            dtype="float16",
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enforce_eager=True,
        )
        t_load = time.perf_counter()
        prompts = [
            "用一句话解释 vLLM 的 PagedAttention。",
            "What is the main bottleneck in LLM decode?",
        ]
        params = SamplingParams(temperature=0.0, max_tokens=32)
        outputs = llm.generate(prompts, params)
        t_done = time.perf_counter()

        result.update(
            {
                "ok": True,
                "timing_s": {
                    "import": t_import - t0,
                    "load": t_load - t_import,
                    "generate": t_done - t_load,
                    "total": t_done - t0,
                },
                "outputs": [
                    {
                        "prompt": output.prompt,
                        "text": output.outputs[0].text,
                        "finish_reason": output.outputs[0].finish_reason,
                    }
                    for output in outputs
                ],
            }
        )
    except Exception as exc:  # pragma: no cover - diagnostic script.
        result.update(
            {
                "ok": False,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
        )

    print(json.dumps(result, ensure_ascii=False, indent=2))
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
