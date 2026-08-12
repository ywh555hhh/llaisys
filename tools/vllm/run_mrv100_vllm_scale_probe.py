#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import subprocess
import time
import traceback
from pathlib import Path


def run(cmd: list[str], timeout: int = 30) -> dict:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return {"cmd": cmd, "returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
    except Exception as exc:
        return {"cmd": cmd, "error": repr(exc)}


def ixsmi_snapshot() -> dict:
    return {
        "summary": run(["/usr/local/corex/bin/ixsmi"]),
        "csv": run([
            "/usr/local/corex/bin/ixsmi",
            "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,gpu.power.draw,temperature.gpu,clocks.current.sm,clocks.current.memory",
            "--format=csv",
        ]),
    }


def load_config(model_dir: Path) -> dict:
    cfg = json.loads((model_dir / "config.json").read_text())
    fields = [
        "architectures",
        "model_type",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "vocab_size",
        "max_position_embeddings",
        "torch_dtype",
    ]
    return {k: cfg.get(k) for k in fields if k in cfg}


def generate_once(llm, prompts: list[str], max_tokens: int) -> dict:
    from vllm import SamplingParams

    params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    t0 = time.perf_counter()
    outputs = llm.generate(prompts, params)
    t1 = time.perf_counter()
    output_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    input_tokens = sum(len(getattr(o, "prompt_token_ids", []) or []) for o in outputs)
    return {
        "elapsed_s": t1 - t0,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "output_tok_s": output_tokens / max(t1 - t0, 1e-9),
        "sample": outputs[0].outputs[0].text[:220] if outputs else "",
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--model-id", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-model-len", type=int, default=1024)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--dtype", default="float16")
    p.add_argument("--quantization", default=None)
    p.add_argument("--cuda-graph", action="store_true")
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=4)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.cuda_graph:
        os.environ["VLLM_ENFORCE_CUDA_GRAPH"] = "1"
    model_dir = Path(args.model_dir)
    result = {
        "model_id": args.model_id,
        "model_dir": str(model_dir),
        "args": vars(args),
        "env": {
            k: os.environ.get(k)
            for k in [
                "VLLM_TARGET_DEVICE",
                "VLLM_ATTENTION_BACKEND",
                "VLLM_WORKER_MULTIPROC_METHOD",
                "VLLM_ENFORCE_CUDA_GRAPH",
            ]
        },
        "model_config": load_config(model_dir) if (model_dir / "config.json").exists() else {},
        "ixsmi_before": ixsmi_snapshot(),
    }
    t0 = time.perf_counter()
    try:
        from vllm import LLM

        t_import = time.perf_counter()
        llm_kwargs = {
            "model": str(model_dir),
            "trust_remote_code": True,
            "dtype": args.dtype,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "enforce_eager": args.enforce_eager,
            "enable_prefix_caching": True,
        }
        if args.quantization:
            llm_kwargs["quantization"] = args.quantization
        llm = LLM(
            **llm_kwargs,
        )
        t_load = time.perf_counter()
        prompts = [
            f"You are benchmarking vLLM on MR-V100. Give one concise inference optimization insight. Request {i}."
            for i in range(args.batch_size)
        ]
        run1 = generate_once(llm, prompts, args.max_tokens)
        run2 = generate_once(llm, prompts, args.max_tokens)
        t_done = time.perf_counter()
        result.update(
            {
                "ok": True,
                "timing_s": {
                    "import": t_import - t0,
                    "load": t_load - t_import,
                    "generate_total": t_done - t_load,
                    "total": t_done - t0,
                },
                "runs": {"warm": run1, "repeat": run2},
            }
        )
        del llm
        gc.collect()
    except Exception as exc:
        result.update({"ok": False, "error": repr(exc), "traceback": traceback.format_exc()})
    result["ixsmi_after"] = ixsmi_snapshot()
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
