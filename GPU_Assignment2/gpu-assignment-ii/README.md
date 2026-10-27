# GPU Assignment II — reproducible RTX 4090 project

**Status:** Executable starter project, NOT completed assignment. Results are produced on your machine. Professor Savitha V’s later clarification permits **one GPU** and makes **Part C optional**. Use the same dataset, exact source row range, and SEED for HW4.a and HW4.5. No second GPU is required.

## Prerequisites
- NVIDIA RTX 4090, CUDA-enabled PyTorch, Linux or supported WSL2, Python 3.10–3.12 recommended.
- Supply the **actual HW4.a Twitter CSV** in `data/twitter.csv`, and set `seed`, `slice`, `text_column`, `label_column` in `configs/experiment.yaml`. The PDF does not provide the CSV, SLICE, or SEED. **Do not invent these values.**
- The serving model is a *separate causal LM* because BERT classification cannot be served by vLLM as a generative model.

## Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# vLLM separately: follow https://docs.vllm.ai/en/latest/getting_started/installation/ for compatible torch/CUDA
pip install vllm
jupyter lab
```

Open `notebooks/GPU_Assignment_II_Main.ipynb` and run in order. For vLLM/HF runs use fresh processes, since model memory is otherwise retained between experiments.

## CLI
```bash
python src/train_bert.py
python src/evaluate_bert.py
python src/batch_scaling.py
python src/memory_techniques.py
python src/benchmark_vllm.py --backend hf --batch 1
python src/benchmark_vllm.py --backend vllm --batch 1
python src/benchmark_vllm.py --backend vllm --batch 8
python src/benchmark_vllm.py --backend vllm --batch 32
```

## Assignment-critical follow-ups (NOT automated in this starter)
1. Part A: run full epochs at every successful batch size; the batch-scaling script currently records short-step extrapolated epoch time, not measured full-epoch time. Report results for your single card. The original PDF mentions a two-card comparison, but the later instructor announcement allows one GPU.
2. Part B: short-step probes do not measure final evaluation accuracy, full-training throughput, or maximum batch size for each technique. Full independent training/evaluation must be performed to complete the assignment. Gradient accumulation must be compared at equal optimizer steps; track real optimizer steps and accuracy.
3. Part C is **optional**. Default is SKIP and METRICS.md must say exactly: `Part C skipped (optional)`. If you choose to run it, use an instructor-approved pairing and measure actual distributed metrics; the included DDP script is only a smoke test.
4. Part D: context-length tests use *word* prefixes as a rough context sweep, not controlled exact token counts. Use tokenizer-verified context lengths and record actual KV-cache allocations/limits. Run on your one card.
5. Part E: calculate measured GPU-hours and produce the ranking table and conclusions after collecting real runs. Do not claim unmeasured values. Record dataset name and exact source row range in METRICS.md.
6. Save `nvidia-smi -L` output or GPU screenshot, applicable UUID captures, reservation records if applicable, GPU-hour records and tag `hw4-5`.

## Hardware limits
A single RTX 4090 is sufficient under the later instructor announcement. Part C is optional and skipped by default. No simulated values should be submitted.

## One-GPU RunPod quick start
From the repository root, supply your original HW4.a dataset at `data/twitter.csv`, and set the actual instructor-assigned `slice` and `seed` in `configs/experiment.yaml`. Do **not** change them between HW4.a and HW4.5.

```bash
nvidia-smi -L | tee GPU_INFO.txt
python -m pip install -r requirements.txt
# Install vLLM in a compatible environment using its official installation guide.
jupyter lab --ip 0.0.0.0 --no-browser
```

Open `notebooks/GPU_Assignment_II_Main.ipynb`. The notebook explicitly skips Part C and uses a single card. `results/dataset_lock.json` is created at the first dataset load and checks the dataset file hash, SLICE, SEED, and source row range on subsequent runs. To intentionally reset it, first archive your results and remove that lock.

**Scope warning:** This project is an executable starter, not a turnkey completed assignment. Full-epoch Part A, full accuracy/maximum-batch Part B, and tokenizer-verified KV-cache memory measurements in Part D require additional implementation and measured runs. Never submit probe measurements as full-run results.

## Updated full measurement workflow (lecture-aligned)
The original `batch_scaling.py` and `memory_techniques.py` remain **diagnostic probes only**. Use these measured runs for assignment tables:

```bash
python src/train_bert.py
python src/evaluate_bert.py
python src/full_experiments.py --mode scaling --max-batch 1024
python src/full_experiments.py --mode techniques
python src/full_experiments.py --mode techniques --fixed-steps 50  # separate equal-step comparison; archive previous CSV first
python src/summarize_results.py
python src/benchmark_vllm.py --backend vllm --batch 8 --context 512
python src/benchmark_vllm.py --backend vllm --batch 8 --context 1024
```

`full_experiments.py` measures actual epoch training duration, test accuracy, GPU memory, and throughput. One technique is applied per independent model initialization, with a separate combined run. `--fixed-steps` is for fair optimizer-step comparisons, not full-epoch tables. To find the *maximum batch for each technique*, run the technique-specific configurations at increasing batch sizes (extend the script's case batches as needed); the default technique table uses the configured baseline batch and **does not establish each technique's maximum batch**.

**Lecture mapping:** Transformer self-attention and causal generation (03); BERT's bidirectional encoder, classification, and MLM versus GPT causal decoding (04); weight/gradient/optimizer/activation memory, precision, and memory tradeoffs (05). BERT training is distinct from vLLM causal-LM serving. Mixed precision changes compute precision, checkpointing trades activation memory for recomputation, accumulation trades steps for effective batch, and 8-bit Adam reduces optimizer-state memory.

**Serving measurement caveat:** p50/p95 now report batch completion time attributed to each simultaneously submitted request, *not* artificially divided by batch size. This is not individual completion latency or time-to-first-token. vLLM GPU memory is captured via `nvidia-smi` where available, not PyTorch allocation counters. Context lengths are tokenizer-derived; KV-cache-only memory is not isolated from model/runtime allocations. Do not claim direct KV-cache memory measurement without a controlled before/after measurement.

**Manual grading requirements:** inspect and write one-line diagnoses for five real worst-class mistakes in `results/worst_class_errors.json`; explain the best validation epoch, throughput/latency tradeoff, KV-cache trend, and GPU-hour calculation in `METRICS.md`. Never present unmeasured GPU numbers.


## Lecture-aligned grading audit (2026-10-27)

Run `python src/submission_audit.py` after full experiments. See `SUBMISSION_AUDIT.md` for rubric mapping and `THEORY_VIVA_GUIDE.md` for course concepts. The notebook now uses `full_experiments.py` for graded batch scaling and independent technique runs rather than short-step probes. These may be lengthy and may OOM by design. No performance results are prefilled.

**Limitations:** Current synchronous serving benchmark reports batch-completion latency attributed to requests, not independently timed request arrivals. The context sweep reports process-level memory, not isolated KV-cache allocation. Do not claim those are exact per-request or KV-cache metrics without improved instrumentation. RunPod GPU execution has not been validated here.


## Pre-run fixes (October 2026)

- `src/full_experiments.py --mode techniques`: independent measurements first; data-selected combination afterward (`results/combination_selection.json`). The decision uses same-batch throughput and memory, with a stated accuracy tolerance; do not describe it as universally optimal.
- `src/kv_context_sweep.py`: separate vLLM processes for increasing context windows at batch 8. Saves startup logs, parsed GPU blocks when available, and a summary. GPU process reservation is not KV usage.
- `src/benchmark_vllm.py`: computes theoretical KV bytes/token from model architecture and explicitly labels p50/p95 as batch-completion latencies.
- `src/finalize_evidence.py`: writes actual `nvidia-smi -L` and a measured-timings `gpu_hours.csv`; enter RunPod pod start/stop times manually to calculate total billed hours without double-counting.
- `PART_B_D_ANALYSIS_TEMPLATE.md`: mandatory written analysis checklist. No numerical results are prefilled.
- `configs/experiment.yaml`: supply instructor-approved seed and slice; place actual CSV at `data/twitter.csv`.
- RunPod launch: open a terminal in the extracted `gpu-assignment-ii` directory, install requirements, run the notebook, and execute `python src/finalize_evidence.py` at the end.

### RunPod serving benchmark settings (7B)
The serving model is `Qwen/Qwen2.5-7B-Instruct` for both Hugging Face and vLLM comparisons. vLLM prefix caching is disabled so identical batch prompts do not share cached prefix blocks. The KV sweep may terminate at a GPU-capacity limit **or** the model context limit; report the observed reason, not an assumed OOM. A 24 GB GPU may fail at engine initialization or for larger batch/context combinations. Record actual failures and full logs. SEED and SLICE must be supplied before running.
