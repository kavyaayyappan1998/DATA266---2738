"""Non-destructive submission audit. Missing results remain missing."""
from pathlib import Path
import json, csv, sys
ROOT=Path(__file__).resolve().parents[1]
checks={
 'Exact dataset fingerprint': 'results/dataset_lock.json',
 'Exact row range': 'results/row_range.json',
 'HW4.a training baseline': 'results/hw4_baseline.json',
 'Per-epoch curves': 'results/training_history.csv',
 'Confusion matrix': 'figures/confusion_matrix.png',
 'Full-epoch batch scaling': 'results/full_batch_scaling.csv',
 'Full-epoch memory comparison': 'results/full_memory_comparison.csv',
 'Serving benchmarks': 'results/serving_metrics.csv',
 'GPU identification': 'GPU_INFO.txt',
 'Metrics report': 'METRICS.md',
 'Run log': 'RUN_LOG.txt',
 'AI disclosure': 'AI_USE.md',
 'GPU-hour record': 'gpu_hours.csv',
}
lines=[]
for name,rel in checks.items():
 p=ROOT/rel
 ok=p.exists() and p.stat().st_size>0
 if rel=='GPU_INFO.txt' and ok:
  ok='NVIDIA' in p.read_text() and 'placeholder' not in p.read_text().lower()
 lines.append(f"{'PASS' if ok else 'MISSING/VERIFY'} | {name} | {rel}")
for rel,needed in [('results/full_batch_scaling.csv',{'batch','samples_per_second','peak_gb','train_seconds','status'}),('results/full_memory_comparison.csv',{'technique','eval_accuracy','peak_gb','samples_per_second','status'}),('results/serving_metrics.csv',{'backend','batch','tokens_per_second','p50_request_ms','p95_request_ms'})]:
 p=ROOT/rel
 if p.exists() and p.stat().st_size:
  with p.open() as f: headers=set(next(csv.reader(f)))
  lines.append(('PASS' if needed<=headers else 'MISSING COLUMNS')+' | '+rel+' | '+str(sorted(needed-headers)))
metrics=(ROOT/'METRICS.md').read_text() if (ROOT/'METRICS.md').exists() else ''
lines.append(('PASS' if 'Part C skipped (optional)' in metrics else 'MISSING')+' | Instructor-approved optional Part C statement')
lines.append('MANUAL REVIEW | Five worst-class misclassifications each need an original one-line diagnosis')
lines.append('MANUAL REVIEW | Training past best validation epoch: describe overfitting and cost')
lines.append('MANUAL REVIEW | Confirm full test-set evaluation, correct units, consistent scales, no fabricated measurements')
lines.append('MANUAL REVIEW | Serving batch completion latency is not independent per-request latency; disclose measurement limitation')
lines.append('MANUAL REVIEW | KV-cache proxy is not isolated KV-cache bytes; do not label it as direct KV-cache measurement')
lines.append('MANUAL REVIEW | Two UUID captures, reservation record, tagged repository and instructor one-GPU clarification')
(ROOT/'results').mkdir(exist_ok=True)
(ROOT/'results/audit_report.txt').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
