import argparse, json, os, time, traceback, gc

def run_once(llm, prompts, max_tokens):
    from vllm import SamplingParams
    params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    t0 = time.perf_counter()
    outs = llm.generate(prompts, params)
    t1 = time.perf_counter()
    out_toks = sum(len(o.outputs[0].token_ids) for o in outs)
    in_toks = sum(len(getattr(o, 'prompt_token_ids', []) or []) for o in outs)
    return {
        'elapsed_s': t1 - t0,
        'input_tokens': in_toks,
        'output_tokens': out_toks,
        'output_tok_s': out_toks / max(t1 - t0, 1e-9),
        'sample_text': outs[0].outputs[0].text[:240] if outs else '',
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='/data/models/Qwen2.5-0.5B-Instruct')
    ap.add_argument('--variant', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--enforce-eager', action='store_true')
    ap.add_argument('--cuda-graph', action='store_true')
    ap.add_argument('--prefix-cache', choices=['on','off'], default='on')
    ap.add_argument('--max-model-len', type=int, default=1024)
    ap.add_argument('--gpu-memory-utilization', type=float, default=0.45)
    args = ap.parse_args()
    if args.cuda_graph:
        os.environ['VLLM_ENFORCE_CUDA_GRAPH'] = '1'
    result = {'variant': args.variant, 'args': vars(args), 'env': {k: os.environ.get(k) for k in ['VLLM_ENFORCE_CUDA_GRAPH','VLLM_TARGET_DEVICE','VLLM_ATTENTION_BACKEND','VLLM_WORKER_MULTIPROC_METHOD']}}
    t0 = time.perf_counter()
    try:
        from vllm import LLM
        t_import = time.perf_counter()
        llm = LLM(
            model=args.model,
            trust_remote_code=True,
            dtype='float16',
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enforce_eager=args.enforce_eager,
            enable_prefix_caching=(args.prefix_cache == 'on'),
        )
        t_load = time.perf_counter()
        short_prompts = [
            'Explain KV cache in one sentence.',
            'What makes LLM decode memory bandwidth bound?',
            'Give one reason PagedAttention helps serving.',
            'Name one benefit of continuous batching.',
            'Why does TTFT differ from TPOT?',
            'What does prefix caching reuse?',
            'What is chunked prefill?',
            'Why are CUDA Graphs useful for decode?',
        ]
        shared = 'System: You are an inference systems engineer. Context: vLLM uses paged KV cache, continuous batching, chunked prefill, prefix caching, and CUDA graphs. ' * 12
        prefix_prompts = [shared + f'Question {i}: summarize one optimization in exactly one sentence.' for i in range(8)]
        result['timing_s'] = {'import': t_import - t0, 'load': t_load - t_import}
        result['runs'] = {}
        result['runs']['short_decode'] = run_once(llm, short_prompts, 64)
        result['runs']['prefix_round1'] = run_once(llm, prefix_prompts, 32)
        result['runs']['prefix_round2'] = run_once(llm, prefix_prompts, 32)
        t_done = time.perf_counter()
        result['timing_s']['total'] = t_done - t0
        result['ok'] = True
        # Best-effort cleanup; vLLM may still warn if distributed group is alive.
        del llm
        gc.collect()
    except Exception as e:
        result['ok'] = False
        result['error'] = repr(e)
        result['traceback'] = traceback.format_exc()
    with open(args.out, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    main()
