# HW2.5 GPU Assignment I — Run Order

1. `00_hw2_5_setup_provenance.ipynb`
   - Edit `SID4_TEXT` in the first cell.
   - Run on the reserved RTX 4090/5090.
   - Fill the generated reservation/GPU-hours record with your real reservation information.

2. `01_hw2_5_precision_bandwidth.ipynb`
   - Parts B and C.
   - Produces precision/bandwidth CSVs and the throughput figure.

3. `02_hw2_5_attention.ipynb`
   - Part D.
   - Measures naive/fused attention, OOM brackets, quadratic memory fit, and speedups.

4. `03_hw2_5_thermal_analysis_submission.ipynb`
   - Parts E and F.
   - The thermal cell runs for 20 minutes.
   - Produces `METRICS.md`, validates deliverables, and creates the artifact backup zip.

Before submission:
- complete `AI_USE.md` in your own words;
- replace all bracketed fields in `reservation_gpu_hours.md`;
- make sure notebooks are executed top-to-bottom with outputs intact;
- commit incrementally and create the `hw2-5` Git tag only after final review.
