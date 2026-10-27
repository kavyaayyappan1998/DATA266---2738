# Required analysis — fill using measured outputs only

## Part B.5 — technique ranking

Read `results/full_memory_comparison.csv`, `results/technique_ranking.csv`, and `results/combination_selection.json`.

- Baseline throughput / peak allocated memory / test accuracy: **TODO**
- AMP result and memory/speed trade-off: **TODO**
- Gradient checkpointing result; would you skip it on a 24 GB GPU, and why? **TODO**
- 8-bit optimizer result and overhead: **TODO**
- Gradient accumulation: effective batch changed; explain comparison caveat: **TODO**
- Chosen combination, measured result, and why: **TODO**

## Part D — KV cache

KV bytes/token/sequence = `2 × num_hidden_layers × num_key_value_heads × head_dim × dtype_bytes`. For a batch of 8 multiply by active token count and 8. This is theoretical occupied KV, **not** the vLLM GPU memory reservation shown by nvidia-smi. Record `results/kv_context_sweep.csv` and read the full engine startup logs in `results/kv_logs/` for cache block counts. If startup logs omit block counts, explicitly mark them unavailable; do not infer blocks from nvidia-smi.

- GPU blocks / block size (from logs if available): **TODO**
- Maximum successful engine `max_model_len` at batch 8: **TODO**
- Did the next length fail, or did the model-supported limit end the sweep? **TODO**
- Explanation of theoretical KV growth vs flat reserved process memory: **TODO**

## HW4.a.6 — serving latency vs throughput (3–4 sentences)

**TODO.** Current p50/p95 are **batch-completion latency** duplicated for each request in the same batch; they are not independently measured per-request response times. Explain the batching/throughput trade-off without claiming TTFT or true per-request tail latency.

## Other written requirements

- Five worst-class error diagnoses (review `results/worst_class_errors.json`): **TODO**
- Best validation epoch, overfitting evidence and GPU-cost sentence: **TODO**
- Part E configuration recommendation supported by measurements: **TODO**
- Instructor's written one-GPU / optional-Part-C clarification: paste or attach **TODO**
- Git tag `hw4-5` after verifying deliverables: **TODO**
