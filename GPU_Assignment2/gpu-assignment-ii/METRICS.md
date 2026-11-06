# HW4.5 summary (not an HW5 baseline)

All numbers below are copied from the files in `results/` and `figures/`. Nothing is estimated except where a line says "analytical". Items marked **TODO** still need the author's input.

- GPU: NVIDIA GeForce RTX 4090 (24 GB), one GPU at a time; see `GPU_INFO.txt`. Two pod sessions were used, so two different physical cards appear in `RUN_LOG.txt`:
  - Session 1 (2026-10-27): `GPU-49f6e7a5-0d30-4554-8244-7a00552b9b36`
  - Session 2 (2026-11-04): `GPU-5a5ec28e-51c2-7128-c85d-2e75b9b3cd39`
- Dataset name: `twitter.csv` (original HW4.a Twitter dataset)
- Dataset SHA256: `655de5e6212bb9d6725b02759da84652e102e876d527e466c867d2f9605a6774` (`results/dataset_lock.json`)
- SLICE: 0
- SEED: 2738
- Exact source row range (zero-based inclusive): 0 to 4999 (5,000 rows, none dropped; `results/row_range.json`)
- Split: 70% train / 15% validation / 15% test, same as HW4.a (3,500 / 750 / 750 rows)
- Labels: 3 classes (0, 1, 2)

**Part C skipped (optional)**

## 1. HW4.a training baseline

Source: `results/hw4_baseline.json`, `results/training_history.csv` (session 2, batch 16, 3 epochs).

| Epoch | Train loss | Validation loss | Validation accuracy |
|---:|---:|---:|---:|
| 1 | 0.837 | 0.685 | 0.691 |
| 2 | 0.540 | 0.685 | 0.688 |
| 3 | 0.297 | 0.848 | 0.675 |

- Wall time: 34.9 s. Peak allocated GPU memory: 2.48 GB.
- Best validation accuracy: 0.691 at epoch 1. Training ran 2 epochs past it.

### Training past the best validation epoch

The model overfits after epoch 1. Train loss keeps falling (0.837 to 0.297) while validation loss rises (0.685 to 0.848) and validation accuracy slips from 0.691 to 0.675. Epochs 2 and 3 therefore bought no validation gain. Their cost was two of the three training epochs: about 20.7 s of the 31.0 s of training time in this run, or roughly 0.006 GPU-hours on one RTX 4090. Stopping at epoch 1 (early stopping on validation accuracy) would have given the same or better model for about a third of the training time.

`figures/training_curves.png` plots loss only; the best epoch is defined by validation accuracy, which is in the table above.

### Test-set evaluation

Source: `results/classification_report.json`, `figures/confusion_matrix.png`.

- Evaluated on the full 750-row test split (confusion-matrix cells sum to 750).
- Test accuracy: 0.705 (529 / 750). Macro F1: 0.700.

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| 0 | 0.695 | 0.650 | 0.672 | 137 |
| 1 | 0.719 | 0.685 | 0.701 | 343 |
| 2 | 0.695 | 0.759 | 0.726 | 270 |

**TODO (author to confirm):** state which checkpoint produced this report. 0.705 matches the one-epoch batch-16 run in `full_batch_scaling.csv`, which is consistent with the epoch-1 (best validation) checkpoint, but the files do not say so.

### Five worst-class misclassifications

Worst class is 0 (lowest recall, 0.650; 48 of 137 misclassified: 37 as class 1, 11 as class 2). The five examples in `results/worst_class_errors.json` are all true 0, predicted 1. Each needs an original one-line diagnosis written by the author; the same line must replace the placeholder in `worst_class_errors.json`.

| # | Text (truncated) | Actual | Predicted | Diagnosis |
|---:|---|---:|---:|---|
| 1 | Municipal Guard in al-Kuwaifiya ... dispose of 13,000 frozen chickens unfit for consumption | 0 | 1 | **TODO (author)** |
| 2 | @user In other related news, you know there's no Parks tomorrow? | 0 | 1 | **TODO (author)** |
| 3 | With James Comey in office, how could the US investigate the Russians? | 0 | 1 | **TODO (author)** |
| 4 | @user I still think it was fake. I always thought Nicki to be the type to call you out ... | 0 | 1 | **TODO (author)** |
| 5 | @user It seems even Nobel prize is for sale. Amartya Sen then and Bob Dylan now | 0 | 1 | **TODO (author)** |

## 2. Memory techniques, full 3-epoch runs

Source: `results/full_memory_comparison.csv`, session 2 rows (the last complete block in the file). Throughput is training samples per second; peak memory is `torch` peak allocated GB; accuracy is the `eval_accuracy` column.

"Max batch" is the largest batch whose peak memory fits inside the card's 24 GB. Larger rows exist in the CSV but are excluded; see "Measurement limitations".

| Technique | Max batch | Throughput (samples/s) | Peak memory (GB) | Eval accuracy |
|---|---:|---:|---:|---:|
| Baseline | 256 | 532.2 | 20.68 | 0.677 |
| Mixed precision | 256 | 1129.0 | 13.91 | 0.671 |
| Gradient checkpointing | 1024 | 400.0 | 13.91 | 0.457 |
| Gradient accumulation | 16 (x4 steps, effective 64) | 392.0 | 2.89 | 0.691 |
| 8-bit optimizer | 256 | 526.6 | 20.08 | 0.677 |
| Best combination | 256 | 529.5 | 20.08 | 0.677 |

Gradient accumulation was only run at batch 16, so 16 is the largest measured batch, not a memory limit.

Same-batch comparison at batch 16 (this is what `figures/memory_comparison.png` shows):

| Technique | Throughput (samples/s) | Peak memory (GB) | Eval accuracy |
|---|---:|---:|---:|
| Baseline | 338.7 | 2.48 | 0.691 |
| Mixed precision | 396.2 | 2.25 | 0.672 |
| Gradient checkpointing | 241.2 | 2.07 | 0.691 |
| Gradient accumulation (effective 64) | 392.0 | 2.89 | 0.691 |
| 8-bit optimizer | 313.7 | 1.88 | 0.681 |
| Best combination | 298.4 | 1.88 | 0.681 |

### Ranking: memory saved per unit of throughput lost (Part B.5)

Each technique is compared with the baseline at the same batch, from the session 2 rows of `full_memory_comparison.csv`. This replaces the saving columns in the original `technique_ranking.csv`, which were computed against the batch-1024 baseline row.

| Rank | Technique | Compared with | Memory saved (GB) | Throughput change (samples/s) | GB saved per sample/s lost | Eval accuracy |
|---:|---|---|---:|---:|---:|---:|
| 1 | Mixed precision | baseline, batch 16 | 0.22 | +57.5 (gain) | no throughput lost | 0.672 |
| 2 | Gradient accumulation (16 x 4) | baseline, batch 64 (same effective batch) | 3.24 | -93.8 | 0.0346 | 0.691 |
| 3 | 8-bit optimizer | baseline, batch 16 | 0.60 | -25.0 | 0.0240 | 0.681 |
| 4 | Gradient checkpointing | baseline, batch 16 | 0.41 | -97.6 | 0.0042 | 0.691 |

At batch 256 the picture is the same at the top and tighter below: mixed precision saves 6.78 GB and gains 596.8 samples/s; gradient checkpointing saves 16.27 GB for 139.2 samples/s lost (0.117 GB per sample/s); the 8-bit optimizer saves 0.61 GB for 5.7 samples/s lost (0.107).

Gradient accumulation is ranked against the batch-64 baseline because it reaches an effective batch of 64. Against the batch-16 baseline it saves nothing (2.89 GB against 2.48 GB).

### Part B answers

- **Mixed precision (B.1):** max batch 256 within 24 GB, 1129.0 samples/s, 13.91 GB, accuracy 0.671, against the baseline's 532.2 samples/s, 20.68 GB and 0.677 at the same batch. It is the only technique that saves memory and is faster.
- **Gradient checkpointing (B.2):** it trades compute for memory. Throughput is 28.8% lower at batch 16 (241.2 against 338.7 samples/s) and 26.1% lower at batch 256 (393.0 against 532.2). In return peak memory at batch 256 falls 79% (4.42 against 20.68 GB), which is why it is the only technique that fits batch 512 and 1024 inside 24 GB.
- **Gradient accumulation (B.3):** physical batch 16 with 4 accumulation steps gives an effective batch of 64. For the same 50 optimizer steps (`fixed_steps_comparison.csv`) it took 9.3 s wall-clock against 5.8 s for the baseline, because it processes four times the samples (3,200 against 800); accuracy after those 50 steps was 0.623 against 0.525. Over 3 full epochs its accuracy was 0.691 at 2.89 GB.
- **8-bit optimizer (B.4):** max batch 256, 526.6 samples/s, 20.08 GB, accuracy 0.677. The saving is a near-constant 0.6 GB (optimizer state), whatever the batch.
- **Which first:** mixed precision. It costs no throughput and its saving grows with batch size.
- **Which not to bother with on a 32 GB card:** the 8-bit optimizer. Its 0.6 GB saving is 2.9% of the batch-256 footprint and would not move the largest batch that fits on either a 24 GB or a 32 GB card, and it adds a dependency (`bitsandbytes`) that failed to import in session 1.

Accuracy falls at large batch because the number of optimizer steps drops (657 steps at batch 16, 12 at batch 1024) with the learning rate unchanged.

### Batch scaling, baseline, one full epoch

Source: `results/full_batch_scaling.csv`, session 2 rows.

| Batch | Throughput (samples/s) | Time per epoch (s) | Peak memory (GB) | Eval accuracy | Note |
|---:|---:|---:|---:|---:|---|
| 16 | 333.1 | 10.5 | 2.48 | 0.705 | |
| 32 | 446.0 | 7.8 | 3.74 | 0.695 | |
| 64 | 486.3 | 7.2 | 6.14 | 0.653 | |
| 128 | 510.7 | 6.9 | 11.03 | 0.495 | |
| 256 | 234.1 | 15.0 | 20.68 | 0.457 | Throughput outlier: 532.2 samples/s (6.6 s per epoch) in the 3-epoch run at the same batch |
| 512 | 15.0 | 232.8 | 40.03 | 0.449 | Exceeds 24 GB; excluded |
| 1024 | 3.7 | 937.9 | 78.70 | 0.377 | Exceeds 24 GB; excluded |

### Part A findings

- **Largest batch that runs:** 256, at 20.68 GB peak on the 24 GB card. Batch 512 needs about 40 GB: it raised out-of-memory in session 1, and in session 2 it only "ran" by exceeding the card's memory at 15 samples/s.
- **Where throughput stops improving:** the gains shrink quickly. In the 3-epoch baseline runs, throughput goes 338.7, 443.1, 485.8, 511.7, 532.2 samples/s for batch 16, 32, 64, 128, 256, which is +31%, +10%, +5% and +4% per doubling. Beyond batch 64 each doubling buys 5% or less, and past 128 the gain is within run-to-run variation, while memory keeps doubling (6.1, 11.0, 20.7 GB).
- **Accuracy cost:** with the learning rate unchanged, one-epoch accuracy falls from 0.705 at batch 16 to 0.457 at batch 256 because the number of optimizer steps falls from 219 to 14.
- **Two-card ratios (A.2, A.3):** not measured; one card was used (instructor clarification, 2026-10-27). In general the ratio of maximum batch sizes would not equal the VRAM ratio, because a fixed part of memory (model weights, optimizer state, CUDA context) does not grow with batch, and because batch size is only tested in powers of two.

## 3. Serving benchmarks

Source: the first successful run in `results/serving_metrics.csv`. Model `Qwen/Qwen2.5-7B-Instruct`, bfloat16, 9-token prompt, 5 repeats per batch size; vLLM 0.32.0 with `max_model_len=512`.

| Backend | Batch | Tokens/s | p50 batch completion (ms) | p95 batch completion (ms) | GPU memory used, nvidia-smi (GB) |
|---|---:|---:|---:|---:|---:|
| Hugging Face `generate` | 1 | 41.2 | 1042.7 | 1047.6 | 14.81 |
| vLLM | 1 | 62.0 | 771.3 | 782.0 | 18.48 |
| vLLM | 8 | 466.1 | 905.0 | 950.8 | 18.49 |
| vLLM | 32 | 1648.5 | 1005.9 | 1049.3 | 18.52 |

Hugging Face was measured at batch 1 only, so there is no like-for-like comparison at batch 8 or 32.

The file also holds a second complete run, which agrees closely: Hugging Face batch 1 at 40.1 tokens/s (p50 1060.8 ms, p95 1093.6 ms); vLLM at 62.2, 455.9 and 1596.9 tokens/s for batch 1, 8 and 32 (p50 773.6, 942.8 and 1052.2 ms; p95 773.8, 979.5 and 1055.7 ms).

### Latency versus throughput (4.a item 6)

In these measurements vLLM's latency does not become clearly worse than the Hugging Face batch-1 baseline (p50 1042.7 ms, p95 1047.6 ms) at any tested batch size. It is lower at batch 1 (771 ms) and batch 8 (905 ms), and reaches parity only at batch 32 (p50 1005.9 ms, p95 1049.3 ms, which is 1.7 ms above the baseline p95). Total throughput keeps improving the whole way, from 62.0 to 466.1 to 1648.5 tokens/s, so the crossover where latency is worse while throughput still rises sits at or just beyond batch 32. For an interactive application the number that matters is latency, p95 in particular, because each user waits for their own reply; total throughput matters for offline or bulk jobs. These latencies are batch completion times, so the crossover is approximate (see "Measurement limitations").

### Part D status

- **D.1 (re-run on the card):** the Hugging Face batch-1 benchmark was repeated in session 2 and gave 34.8 to 43.4 tokens/s (p50 963 to 1261 ms) against 41.2 tokens/s (p50 1043 ms) in session 1, so that baseline reproduces within run-to-run variation on a second RTX 4090. vLLM could not be re-run in session 2: the engine failed to start because the pod image had no C compiler (`/usr/bin/gcc` not found). The vLLM rows in the table above are therefore the session 1 measurements only.
- **D.2 (KV cache against context length):** not measured, for the same reason. No longest context at batch 8 is reported. The only KV figures available are the analytical 57,344 bytes per token per sequence and vLLM's start-up pool of 3.16 GiB (59,168 tokens) at `max_model_len=512`.
- **D.3 (comparison with HW4):** there is no second set of vLLM numbers to compare, so no hardware or precision difference can be claimed. The gap is an environment failure in the later run, not a measurement flaw in the earlier one.
- Only one card was used (instructor clarification, 2026-10-27), so "both cards" comparisons do not apply.

## 4. Part E: analysis and summary

### GPU-hours for one full fine-tune (E.1)

GPU-hours = wall-clock seconds x number of GPUs / 3600.

- From the measured wall time: 34.9 s x 1 GPU / 3600 = **0.0097 GPU-hours** (HW4.a settings: batch 16, 3 epochs, 3,500 training rows, including per-epoch validation).
- Cross-check from training speed: 3 epochs x 3,500 rows = 10,500 samples; 10,500 / 338.7 samples/s = 31.0 s of training = 0.0086 GPU-hours. The remaining 3.9 s is evaluation and overhead.

### Configuration I would use for a full fine-tune (E.2)

**Mixed precision at batch 128**, stopping at the best validation epoch.

- Wall-clock: 10.4 s for 3 epochs including evaluation, against 33.4 s for the baseline at batch 16 (3.2x faster).
- Memory: 7.67 GB peak, about a third of the 24 GB card, so there is headroom for longer sequences.
- Accuracy: 0.708, the highest measured in the comparison (baseline batch 16: 0.691). With 750 evaluation rows a difference of this size is within noise, so the claim is "no accuracy loss", not "more accurate".
- Batch 256 with mixed precision is only 3% faster (10.1 s) and scores lower (0.671) because it takes half as many optimizer steps, so it is not worth the extra 6.2 GB.
- Checkpointing, accumulation and the 8-bit optimizer are not needed: the model already fits with a wide margin, and each of them costs throughput.
- From section 1, validation accuracy peaked at epoch 1, so early stopping would cut the cost further.

### HW4.5 summary (E.3)

- Best peak-memory config: 8-bit optimizer at batch 16, 1.88 GB (0.60 GB below baseline). In the 50-step comparison (`results/fixed_steps_comparison.csv`), checkpointing plus 8-bit optimizer reached 1.08 GB.
- Best throughput config: mixed precision at batch 256, 1129.0 samples/s (2.1x the baseline at the same batch).
- Best wall-clock config: mixed precision at batch 256, 10.1 s for 3 epochs including evaluation (baseline batch 16: 33.4 s).

## Measurement limitations

- **Latency is batch completion time.** The p50/p95 figures are the time for a whole batch to finish, taken across 5 repeats. They are not independent per-request latency and not time to first token. At batch 32, every request in the batch is assigned the same completion time.
- **KV cache is a proxy, not a measurement.** The 57,344 bytes per token per sequence figure is analytical (model shape times dtype size). It is an upper bound for active prompt plus output tokens and excludes block padding. vLLM's startup log reports a preallocated pool of 3.16 GiB (59,168 tokens); that is reserved capacity, not bytes in use. The nvidia-smi column is whole-process GPU memory (weights plus reserved cache), not KV-cache bytes. No isolated KV-cache byte count was measured.
- **KV context sweep produced no result.** All four attempts in `results/kv_context_sweep.csv` (`max_model_len=512`, batch 8) failed at engine start-up with `FileNotFoundError: /usr/bin/gcc` in session 2. This is an environment failure, not an out-of-memory limit, so no longest-successful context length is reported.
- **Rows above 24 GB are excluded.** In session 2, batch 512 and 1024 report 26 to 79 GB peak allocated on a 24 GB card, with throughput 30 to 150 times lower than at batch 256. In session 1 the same batch-512 runs ended in out-of-memory. These rows are kept in the CSVs as recorded but are not used for any "max batch" or "best" claim.
- **"Best combination" full-epoch rows are not the selected combination.** `results/combination_selection.json` selects gradient checkpointing plus 8-bit optimizer, but the full-epoch rows were run with checkpointing off and are identical in memory to the 8-bit-optimizer rows. Only the 50-step row in `fixed_steps_comparison.csv` ran both flags (1.08 GB, 258.6 samples/s).
- **`technique_ranking.csv` was recomputed.** The original file measured `memory_saved_gb` against the 78.70 GB batch-1024 baseline row, so it showed about 76 GB "saved" for every small batch. The replacement file compares each technique with the baseline at the same batch, using only measured session 2 rows; no new measurements were added.
- **CSV files hold more than one run.** `full_batch_scaling.csv`, `full_memory_comparison.csv`, `technique_ranking.csv`, `hw4_baseline.csv` and `serving_metrics.csv` were appended to across both sessions, including failed attempts. Session 1 training rows used the placeholder SEED 42 and are superseded; no training number in this report comes from them. `serving_metrics.csv` originally had rows with 13, 21 and 23 columns under a 23-column header, so values sat under the wrong names; it has been realigned to the header with no value changed and no row removed. Its 9 failed vLLM rows and 6 Hugging Face rows with a 0.0 GB nvidia-smi reading (not captured) are kept as recorded. `figures/throughput_vs_batch.png` draws both sessions as one line, which is why it doubles back.
- **Eval accuracy column.** `combination_selection.json` describes `eval_accuracy` as test accuracy. **TODO (author to confirm)** against the training script.
- **GPU-hours.** `results/gpu_hours_calculated.json` (0.0097 GPU-hours) covers the 34.9 s baseline only. The measured wall-time rows in the training CSVs across both sessions sum to about 14,888 s (4.14 GPU-hours), excluding model loading, serving runs and idle pod time. See `gpu_hours.csv` for the billed record.
