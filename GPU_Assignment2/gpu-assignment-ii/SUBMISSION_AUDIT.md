# Submission audit — HW4.a and HW4.5 (one GPU)

**Instructor clarification:** One GPU is sufficient; Part C optional. Keep the original announcement as evidence. Run `python src/submission_audit.py` after all experiments.

| Requirement | Notebook section | Evidence | What to verify |
|---|---|---|---|
| Exact Twitter data, consecutive SLICE rows, SEED, 70/15/15 | 1 | `results/dataset_lock.json`, `row_range.json` | Identical HW4.a/HW4.5; record actual row range and dataset name |
| BERT fine-tuning and per-epoch train/validation metrics | 2 | `training_history.csv`, `hw4_baseline.json` | Full epochs, best epoch and cost of further training |
| Full held-out test, confusion matrix, per-class precision/recall | 3 | evaluation script outputs | Worst class and five actual mistakes with personally written diagnoses |
| HF batch 1, vLLM batch 1/8/32 | 4, 9 | `serving_metrics.csv` | Tokens/sec, p50/p95, latency definition, trade-off discussion |
| Batch scaling until OOM | 5 | `full_batch_scaling.csv` | Full epoch samples/sec, peak memory, epoch time, plateau |
| Four independent techniques plus best combination | 6 | `full_memory_comparison.csv` | Maximum batch, throughput, memory, test accuracy, comparable optimizer steps |
| Optional Part C | 8 | `METRICS.md` | Exact phrase: Part C skipped (optional) |
| KV cache vs context and max context batch 8 | 9 | context sweep + interpretation | Existing process-memory measurement is a proxy; direct KV cache measurement needs engine instrumentation |
| GPU hours and final recommendations | 10 | `gpu_hours.csv`, `METRICS.md` | Actual measured wall-clock / 3600; technique ranking and choice |
| Final submission | 11, 12 | repository | `hw4-5` tag, two UUID captures, GPU identification, reservation record, logs, AI_USE |

## Rubric lessons from earlier LAB 1

- **No incomplete metrics:** compare against the exact metric list; no character-level-only proxies when a broader metric is required.
- **No silent overfitting:** training/validation plots and explanation of the epoch where validation stopped improving.
- **No short substitute runs:** distinguish quick smoke tests from complete graded runs.
- **No tiny evaluation subsets:** use the entire required test set.
- **No omitted metrics:** audit every row and plot; report limitations transparently.
- **No mismatched scales:** identical units, denominators, data splits, hardware labels and benchmark settings.
- **Viva:** explain every optimization and its trade-offs in your own words.

**Not a certificate of correctness:** static checks cannot replace execution on RunPod or human review of the assignment report.
