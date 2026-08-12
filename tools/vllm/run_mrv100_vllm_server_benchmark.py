#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import aiohttp
from transformers import AutoTokenizer


def run(cmd: list[str], timeout: int = 30) -> dict[str, Any]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return {
            "cmd": cmd,
            "returncode": p.returncode,
            "stdout": p.stdout,
            "stderr": p.stderr,
        }
    except Exception as exc:
        return {"cmd": cmd, "error": repr(exc)}


def ixsmi_snapshot() -> dict[str, Any]:
    return {
        "summary": run(["/usr/local/corex/bin/ixsmi"]),
        "csv": run(
            [
                "/usr/local/corex/bin/ixsmi",
                "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,gpu.power.draw,temperature.gpu,clocks.current.sm,clocks.current.memory",
                "--format=csv",
            ]
        ),
    }


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize(values: list[float]) -> dict[str, float | None]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else None,
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def make_prompt(
    tokenizer: Any,
    target_prompt_tokens: int,
    request_id: int,
    prompt_mode: str,
    seed: int,
) -> str:
    shared_seed = (
        "You are benchmarking vLLM serving on an Iluvatar MR-V100 accelerator. "
        "Give a concise systems insight about inference throughput, KV cache, "
        "CUDA Graph, quantization, and latency tradeoffs. "
    )
    if prompt_mode == "shared":
        text = shared_seed
    elif prompt_mode == "random":
        digest = hashlib.sha256(f"{seed}:{request_id}".encode("utf-8")).hexdigest()
        text = (
            "You are benchmarking a single request with a mostly unique prefix. "
            f"Unique workload fingerprint {digest}. "
            "Discuss scheduler pressure, prefill cost, decode throughput, and latency. "
        )
    else:
        raise ValueError(f"unknown prompt_mode={prompt_mode!r}")
    while len(tokenizer.encode(text, add_special_tokens=False)) < target_prompt_tokens:
        if prompt_mode == "shared":
            text += shared_seed
        else:
            digest = hashlib.sha256(f"{seed}:{request_id}:{len(text)}".encode("utf-8")).hexdigest()
            text += (
                f"Segment {digest[:16]} {digest[16:32]} {digest[32:48]} {digest[48:]}. "
                "Keep this request-specific context distinct from other concurrent prompts. "
            )
    return f"{text}\nRequest id: {request_id}. Answer in one compact paragraph."


async def request_once(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    tokenizer: Any,
    prompt: str,
    max_tokens: int,
    request_id: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": True,
    }
    t0 = time.perf_counter()
    first_token_s: float | None = None
    text_parts: list[str] = []
    usage: dict[str, Any] | None = None
    status = 0
    error: str | None = None
    raw_events = 0
    try:
        async with session.post(url, json=payload, timeout=None) as resp:
            status = resp.status
            if resp.status != 200:
                body = await resp.text()
                error = body[:1000]
            else:
                buffer = ""
                async for chunk in resp.content.iter_any():
                    buffer += chunk.decode("utf-8", errors="replace")
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            continue
                        raw_events += 1
                        obj = json.loads(data)
                        if obj.get("usage"):
                            usage = obj["usage"]
                        choices = obj.get("choices") or []
                        if not choices:
                            continue
                        delta = choices[0].get("text") or ""
                        if delta and first_token_s is None:
                            first_token_s = time.perf_counter() - t0
                        text_parts.append(delta)
    except Exception as exc:
        error = repr(exc)
    total_s = time.perf_counter() - t0
    output_text = "".join(text_parts)
    output_tokens = None
    input_tokens = None
    if usage:
        output_tokens = usage.get("completion_tokens")
        input_tokens = usage.get("prompt_tokens")
    if output_tokens is None:
        output_tokens = len(tokenizer.encode(output_text, add_special_tokens=False))
    if input_tokens is None:
        input_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
    decode_s = None
    tpot_s = None
    if first_token_s is not None:
        decode_s = max(total_s - first_token_s, 0.0)
        if output_tokens and output_tokens > 1:
            tpot_s = decode_s / (output_tokens - 1)
    return {
        "request_id": request_id,
        "ok": status == 200 and error is None,
        "status": status,
        "error": error,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "ttft_s": first_token_s,
        "total_s": total_s,
        "decode_s": decode_s,
        "tpot_s": tpot_s,
        "output_tok_s": (output_tokens / total_s) if output_tokens and total_s > 0 else None,
        "raw_events": raw_events,
        "sample": output_text[:240],
    }


async def run_concurrency(
    base_url: str,
    model: str,
    tokenizer: Any,
    prompt_tokens: int,
    max_tokens: int,
    concurrency: int,
    requests_total: int,
    prompt_mode: str,
    seed: int,
) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/completions"
    connector = aiohttp.TCPConnector(limit=max(concurrency, 1))
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=30)
    prompts = [
        make_prompt(tokenizer, prompt_tokens, i, prompt_mode, seed)
        for i in range(requests_total)
    ]
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        next_id = 0
        active: set[asyncio.Task[dict[str, Any]]] = set()
        while next_id < requests_total or active:
            while next_id < requests_total and len(active) < concurrency:
                task = asyncio.create_task(
                    request_once(
                        session,
                        url,
                        model,
                        tokenizer,
                        prompts[next_id],
                        max_tokens,
                        next_id,
                    )
                )
                active.add(task)
                next_id += 1
            done, active = await asyncio.wait(active, return_when=asyncio.FIRST_COMPLETED)
            results.extend(task.result() for task in done)
    elapsed = time.perf_counter() - started
    ok_results = [r for r in results if r.get("ok")]
    total_output_tokens = sum(r.get("output_tokens") or 0 for r in ok_results)
    total_input_tokens = sum(r.get("input_tokens") or 0 for r in ok_results)
    return {
        "concurrency": concurrency,
        "requests_total": requests_total,
        "elapsed_s": elapsed,
        "ok_count": len(ok_results),
        "error_count": len(results) - len(ok_results),
        "aggregate_output_tok_s": total_output_tokens / elapsed if elapsed else None,
        "aggregate_total_tok_s": (total_input_tokens + total_output_tokens) / elapsed if elapsed else None,
        "ttft_s": summarize([r["ttft_s"] for r in ok_results if r.get("ttft_s") is not None]),
        "total_s": summarize([r["total_s"] for r in ok_results if r.get("total_s") is not None]),
        "tpot_s": summarize([r["tpot_s"] for r in ok_results if r.get("tpot_s") is not None]),
        "per_request": sorted(results, key=lambda r: r["request_id"]),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--model", default="qwen32b-awq")
    p.add_argument("--tokenizer-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--prompt-tokens", type=int, default=64)
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--concurrency-list", default="1,2,4")
    p.add_argument("--requests-per-concurrency", type=int, default=4)
    p.add_argument("--label", default="server_benchmark")
    p.add_argument("--prompt-mode", choices=["shared", "random"], default="shared")
    p.add_argument("--seed", type=int, default=20260812)
    return p.parse_args()


async def async_main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, trust_remote_code=True)
    concurrencies = [int(x) for x in args.concurrency_list.split(",") if x.strip()]
    result: dict[str, Any] = {
        "label": args.label,
        "model": args.model,
        "base_url": args.base_url,
        "tokenizer_dir": args.tokenizer_dir,
        "prompt_tokens_target": args.prompt_tokens,
        "prompt_mode": args.prompt_mode,
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "concurrency_list": concurrencies,
        "requests_per_concurrency": args.requests_per_concurrency,
        "ixsmi_before": ixsmi_snapshot(),
        "workloads": [],
    }
    try:
        for concurrency in concurrencies:
            requests_total = max(concurrency * args.requests_per_concurrency, concurrency)
            workload = await run_concurrency(
                args.base_url,
                args.model,
                tokenizer,
                args.prompt_tokens,
                args.max_tokens,
                concurrency,
                requests_total,
                args.prompt_mode,
                args.seed,
            )
            result["workloads"].append(workload)
    finally:
        result["ixsmi_after"] = ixsmi_snapshot()
        Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(async_main())
