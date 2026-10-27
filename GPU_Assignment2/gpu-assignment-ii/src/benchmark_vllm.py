"""HF/vLLM serving benchmark with batch-completion latency and theoretical KV accounting."""
import argparse
import gc
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import config, append_csv, gpu_mem

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "serving_metrics.csv"


def gpu_used_gb():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=used_gpu_memory", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL,
        )
        return round(sum(float(x.strip()) for x in out.splitlines() if x.strip()) / 1024, 3)
    except Exception:
        return None


def kv_spec(model):
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model, trust_remote_code=False)
    layers = getattr(cfg, "num_hidden_layers", None)
    heads = getattr(cfg, "num_key_value_heads", None) or getattr(cfg, "num_attention_heads", None)
    attn_heads = getattr(cfg, "num_attention_heads", None)
    hidden = getattr(cfg, "hidden_size", None)
    head_dim = getattr(cfg, "head_dim", None) or (hidden // attn_heads if hidden and attn_heads else None)
    if not all((layers, heads, head_dim)):
        raise ValueError("Cannot determine KV dimensions from model configuration")
    return {"layers": layers, "kv_heads": heads, "head_dim": head_dim}


def run(backend="vllm", batch=1, context=0, max_model_len=None, kv_dtype="auto"):
    if batch < 1 or context < 0:
        raise ValueError("batch must be >= 1 and context must be >= 0")

    c = config()
    model = c["serving_model"]
    repeats = int(c["serving_repeats"])
    max_t = int(c["serving_max_tokens"])
    if repeats < 1 or max_t < 1:
        raise ValueError("serving_repeats and serving_max_tokens must be positive")

    row = {
        "backend": backend, "batch": batch,
        "requested_prefix_tokens": context,
        "max_model_len": max_model_len,
        "kv_dtype": kv_dtype,
        "status": "error",
        "error_type": None,
        "error_message": None,
    }
    engine = None
    pipe = None
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model, trust_remote_code=False)
        prompt = c["serving_prompt"]
        if context:
            prefix_ids = tok.encode(" word" * max(context * 4, 100), add_special_tokens=False)[:context]
            prompt = tok.decode(prefix_ids, skip_special_tokens=True) + " " + prompt
        actual_prompt_tokens = len(tok.encode(prompt, add_special_tokens=False))
        prompts = [prompt] * batch
        row["actual_prompt_tokens"] = actual_prompt_tokens

        kv = kv_spec(model)
        # fp8 is 1 byte per element; auto/fp16/bf16 assumed 2 bytes on this GPU.
        dtype_bytes = 1 if kv_dtype == "fp8" else 2
        kv_bytes_per_token = 2 * kv["layers"] * kv["kv_heads"] * kv["head_dim"] * dtype_bytes
        row["kv_bytes_per_token_per_sequence"] = kv_bytes_per_token
        row["theoretical_kv_gb_for_batch"] = round(
            kv_bytes_per_token * (actual_prompt_tokens + max_t) * batch / 2**30, 5
        )
        row["kv_note"] = (
            "Analytical upper-bound for active prompt+max-output tokens, not measured KV usage; "
            "excludes cache block padding and model weights."
        )
        row["latency_definition"] = (
            "p50/p95 of batch completion times across repeats; NOT per-request latency or TTFT."
        )

        if backend == "vllm":
            from vllm import LLM, SamplingParams
            kwargs = {
                "model": model,
                "trust_remote_code": False,
                "gpu_memory_utilization": 0.80,
                "enforce_eager": True,
                "kv_cache_dtype": kv_dtype,
                "enable_prefix_caching": False,
            }
            if max_model_len is not None:
                kwargs["max_model_len"] = int(max_model_len)
            engine = LLM(**kwargs)
            sampling = SamplingParams(max_tokens=max_t, temperature=0)

            def infer(items):
                outputs = engine.generate(items, sampling, use_tqdm=False)
                return sum(len(o.outputs[0].token_ids) for o in outputs)

        elif backend == "hf":
            from transformers import pipeline
            pipe = pipeline("text-generation", model=model, device=0, dtype=torch.float16)

            def infer(items):
                outputs = pipe(items, max_new_tokens=max_t, do_sample=False, return_full_text=False,
                               batch_size=len(items))
                total = 0
                for item in outputs:
                    if isinstance(item, list):
                        item = item[0]
                    total += len(pipe.tokenizer.encode(item["generated_text"], add_special_tokens=False))
                return total

        else:
            raise ValueError(f"Unknown backend: {backend}")

        # Warm up outside timed region.
        infer(prompts[:1])
        torch.cuda.synchronize()
        durations_ms = []
        total_tokens = 0
        started = time.perf_counter()
        for _ in range(repeats):
            t0 = time.perf_counter()
            total_tokens += infer(prompts)
            torch.cuda.synchronize()
            durations_ms.append((time.perf_counter() - t0) * 1000)
        elapsed = time.perf_counter() - started

        row.update({
            "status": "ok",
            "requests": batch * repeats,
            "generated_tokens": total_tokens,
            "tokens_per_second": round(total_tokens / elapsed, 3),
            "p50_batch_ms": round(float(np.percentile(durations_ms, 50)), 3),
            "p95_batch_ms": round(float(np.percentile(durations_ms, 95)), 3),
            "peak_allocated_gb": gpu_mem() if backend == "hf" else None,
            "gpu_used_gb_from_nvidia_smi": gpu_used_gb(),
            "elapsed_s": round(elapsed, 3),
        })
    except Exception as exc:
        message = str(exc)
        row["error_type"] = type(exc).__name__
        row["error_message"] = message[:1000]
        row["status"] = "oom" if "out of memory" in message.lower() or "cuda oom" in message.lower() else "error"
        print(traceback.format_exc(), file=sys.stderr, flush=True)
    finally:
        RESULTS.parent.mkdir(parents=True, exist_ok=True)
        # common.append_csv uses a relative path; ensure it resolves from project root.
        import os
        previous_cwd = Path.cwd()
        try:
            os.chdir(ROOT)
            append_csv("results/serving_metrics.csv", row)
        finally:
            os.chdir(previous_cwd)
        print(json.dumps(row, default=str), flush=True)
        del engine, pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return row["status"] == "ok"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--context", type=int, default=0)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--kv-dtype", choices=["auto", "fp16", "bf16", "fp8"], default="auto")
    args = parser.parse_args()
    success = run(args.backend, args.batch, args.context, args.max_model_len, args.kv_dtype)
    sys.exit(0 if success else 1)
