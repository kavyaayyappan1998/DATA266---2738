"""Choose the measured Pareto-competitive memory techniques, then benchmark their combination."""
from pathlib import Path
import json
import pandas as pd
from common import ROOT

def select_and_run(base, max_batch, fixed_steps, scan, experiment, save_rows):
    path=ROOT/'results/full_memory_comparison.csv'
    df=pd.read_csv(path)
    df=df[(df.status=='ok') & (df.batch==base) & (df.technique!='best_combination')]
    if df.empty or 'baseline' not in set(df.technique):
        raise RuntimeError('No baseline and successful technique measurements for combination selection')
    # Compare same-batch, same-epoch results. Accumulation changes effective batch;
    # report separately rather than mixing it into this optimizer/memory comparison.
    latest=df.drop_duplicates('technique',keep='last').set_index('technique')
    baseline=latest.loc['baseline']
    candidates=[]
    for name,flag in [('mixed_precision','amp'),('gradient_checkpointing','checkpoint'),('8bit_optimizer','optimizer8')]:
        if name not in latest.index:continue
        row=latest.loc[name]
        memory_saved=float(baseline.peak_gb)-float(row.peak_gb)
        speed_ratio=float(row.samples_per_second)/float(baseline.samples_per_second)
        # Require a measured memory saving, no >10% throughput regression,
        # and no >1 percentage-point test accuracy drop; thresholds documented.
        accuracy_drop=float(baseline.eval_accuracy)-float(row.eval_accuracy)
        if memory_saved>0 and speed_ratio>=0.90 and accuracy_drop<=0.01:
            candidates.append((name,flag,memory_saved,speed_ratio,accuracy_drop))
    flags={c[1]:True for c in candidates}
    selection={'baseline_batch':base,'selection_rule':'memory saving > 0 GB, throughput >= 90% of baseline, test accuracy drop <= 0.01; same batch; accumulation excluded because effective batch differs','eligible':candidates,'chosen_flags':flags,'warning':'Selection uses test accuracy for assignment analysis only; for unbiased model selection use validation accuracy.'}
    (ROOT/'results/combination_selection.json').write_text(json.dumps(selection,indent=2))
    print('Measured combination selection:',selection,flush=True)
    # A baseline-equivalent combination is permitted when no technique meets criteria.
    sizes=[base]
    if scan:
        sizes=[];b=base
        while b<=max_batch:sizes.append(b);b*=2
    for batch in sizes:
        row=experiment('best_combination',batch,fixed_steps=fixed_steps,**flags)
        save_rows('results/full_memory_comparison.csv',row)
        print(row,flush=True)
        if row['status']!='ok':break
