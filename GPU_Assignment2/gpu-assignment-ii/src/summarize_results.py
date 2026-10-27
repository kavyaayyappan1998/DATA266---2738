"""Generate analysis from actual recorded runs only; missing experiments remain missing."""
from common import *

def main():
 p=ROOT/'results/full_memory_comparison.csv'
 if not p.exists():raise FileNotFoundError('Run python src/full_experiments.py --mode techniques first')
 df=pd.read_csv(p);df=df[df.status=='ok'].copy()
 if df.empty:raise ValueError('No successful measured runs')
 base=df[df.technique=='baseline']
 if len(base):
  b=base.iloc[-1];df['memory_saved_gb']=b.peak_gb-df.peak_gb;df['throughput_lost_sps']=b.samples_per_second-df.samples_per_second
  df['memory_saved_per_throughput_lost']=df.apply(lambda r: r.memory_saved_gb/r.throughput_lost_sps if r.throughput_lost_sps>0 else float('nan'),axis=1)
 df.to_csv(ROOT/'results/technique_ranking.csv',index=False)
 print(df[['technique','batch','samples_per_second','peak_gb','eval_accuracy']].to_string(index=False))
 baseline=ROOT/'results/hw4_baseline.json'
 if baseline.exists():
  b=json.loads(baseline.read_text());hours=b['wall_seconds']/3600
  print(f'Baseline full fine-tune: {b["wall_seconds"]:.2f} seconds / 3600 = {hours:.5f} GPU-hours (one GPU)')
  (ROOT/'results/gpu_hours_calculated.json').write_text(json.dumps({'baseline_wall_seconds':b['wall_seconds'],'gpu_count':1,'gpu_hours':hours},indent=2))
if __name__=='__main__':main()
