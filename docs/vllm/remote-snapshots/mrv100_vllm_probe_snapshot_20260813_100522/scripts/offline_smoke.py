import json, time, os, traceback
MODEL=os.environ.get('MODEL_DIR','/data/models/Qwen2.5-0.5B-Instruct')
JSON_PATH=os.environ.get('OUT_JSON','/tmp/vllm_smoke.json')
out={"model": MODEL, "env": {k: os.environ.get(k) for k in ['VLLM_TARGET_DEVICE','VLLM_ATTENTION_BACKEND','VLLM_USE_V1','VLLM_WORKER_MULTIPROC_METHOD','VLLM_ENFORCE_CUDA_GRAPH']}}
t0=time.perf_counter()
try:
    from vllm import LLM, SamplingParams
    t_import=time.perf_counter()
    llm=LLM(model=MODEL, trust_remote_code=True, dtype="float16", max_model_len=1024, gpu_memory_utilization=0.45, enforce_eager=True)
    t_load=time.perf_counter()
    prompts=["用一句话解释 vLLM 的 PagedAttention。", "What is the main bottleneck in LLM decode?"]
    params=SamplingParams(temperature=0.0, max_tokens=32)
    outputs=llm.generate(prompts, params)
    t_done=time.perf_counter()
    out.update({"ok": True, "timing_s": {"import": t_import-t0, "load": t_load-t_import, "generate": t_done-t_load, "total": t_done-t0}, "outputs": []})
    for o in outputs:
        out["outputs"].append({"prompt": o.prompt, "text": o.outputs[0].text, "finish_reason": o.outputs[0].finish_reason})
except Exception as e:
    out.update({"ok": False, "error": repr(e), "traceback": traceback.format_exc()})
print(json.dumps(out, ensure_ascii=False, indent=2))
with open(JSON_PATH, 'w') as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
