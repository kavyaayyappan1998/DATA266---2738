PROJECT OVERVIEW
This project reproduces BERT fine-tuning and evaluates GPU training and inference behavior, including batch-size scaling, memory optimization techniques, Hugging Face versus vLLM serving, and a vLLM context-length/KV-cache sweep.

This README describes the workflow in the supplied GPU_Assignment_II_Main_(1).ipynb notebook. It does not claim that any experiment has successfully completed. Use the actual generated CSV files and logs for results.

REPOSITORY AND FILES
GitHub: https://github.com/kavyaayyappan1998/DATA266---2738.git
Project subdirectory: GPU_Assignment2/gpu-assignment-ii
Main notebook: notebooks/GPU_Assignment_II_Main.ipynb
Configuration: configs/experiment.yaml
Dataset: data/twitter.csv
Training and evaluation scripts: src/
Generated measurements and logs: results/
Generated figures: figures/

REPRODUCIBILITY SETTINGS
- SEED = 2738 (last four digits of the SJSU student ID).
- SLICE = 0 in the supplied notebook; verify this is the instructor-required slice.
- Use the same dataset, source row range, SEED, and SLICE for HW4.a and HW4.5.
- The notebook is designed for a single GPU. Part C (distributed training) is optional and skipped.
- BERT fine-tuning is configured for 3 epochs; confirm the instructor's required settings.
- The serving benchmark uses a separate causal language model (Qwen/Qwen2.5-7B-Instruct), not the BERT classifier.

RUNNING IN THE SJSU GPU LAB
1. Clone or download the repository on the lab workstation, following lab rules.
2. Start the lab-approved GPU-enabled PyTorch Docker image and mount the project folder as /app.
3. Start Jupyter in the container, then connect Google Colab to the local runtime using the Jupyter URL and token (or use Jupyter directly).
4. Verify GPU access with nvidia-smi and torch.cuda.is_available().
5. Open notebooks/GPU_Assignment_II_Main.ipynb. Before running, inspect the notebook's ROOT paths. Several cells explicitly use ROOT = Path('/app'), so the project itself must be mounted at /app for those cells.
6. Check configs/experiment.yaml, SEED, SLICE, dataset path, and dataset lock. Back up earlier results before rerunning with a different seed.
7. Run notebook cells in order, checking each experiment's exit status and generated output. Some dependency setup may require an approved lab environment.
8. Save outputs to GitHub or approved cloud storage before leaving the lab.

NOTEBOOK EXPERIMENTS
1. Setup and dataset: configure environment and lock the data split.
2. HW4.a: fine-tune BERT and evaluate the test set, including a confusion matrix and class-level metrics.
3. Table HW4.1: benchmark Hugging Face and vLLM serving at specified batch sizes.
4. Part A: full-epoch batch-size scaling with src/full_experiments.py --mode scaling --max-batch 1024.
5. Part B: independent optimization-technique experiments with src/full_experiments.py --mode techniques --scan-technique-max --max-batch 1024. Includes mixed precision, gradient checkpointing, gradient accumulation, and an 8-bit optimizer, subject to successful bitsandbytes installation.
6. Part B.3: equal-optimizer-step comparison. Keep this output separate from full-epoch measurements.
7. Part C: skipped (optional).
8. Part D.2: run src/kv_context_sweep.py in the working vLLM environment, using batch size 8 and recording the context lengths actually served.
9. Part E: summarize measurements, record GPU evidence and GPU-hours, and run the submission audit.

RUNNING INDIVIDUAL EXPERIMENTS
From the project root inside the GPU environment:

  python src/train_bert.py
  python src/evaluate_bert.py
  python src/full_experiments.py --mode scaling --max-batch 1024
  python src/full_experiments.py --mode techniques --scan-technique-max --max-batch 1024
  python src/kv_context_sweep.py
  python src/finalize_evidence.py
  python src/summarize_results.py
  python src/submission_audit.py

IMPORTANT: The full techniques command reruns all configured technique cases. To run only the 8-bit optimizer, restrict the cases list in src/full_experiments.py and disable any automatic best-combination run temporarily; restore the original script afterward. Do not confuse a vLLM import error with GPU out-of-memory.

RESULTS AND REPORTING
- Use results/full_batch_scaling.csv for measured batch-scaling results.
- Use results/full_memory_comparison.csv for technique comparisons.
- Use results/serving_metrics.csv for serving measurements; check the CSV schema before reporting percentiles, since earlier runs produced incompatible column layouts.
- Use results/kv_context_sweep.csv and associated logs for the context sweep only if the sweep completed successfully.
- Update METRICS.md and PART_B_D_ANALYSIS_TEMPLATE.md using actual observed measurements.
- Write five human-reviewed one-line explanations for misclassified examples in results/worst_class_errors.json.
- Report the GPU model, memory limits, failed runs, elapsed GPU time, and any unresolved limitations.
- Do not fabricate missing 8-bit or KV-cache measurements or present process GPU memory as isolated KV-cache memory.


