"""Record actual GPU identity and run-derived GPU-hours; never fabricate metrics."""
import sys,subprocess,csv,json,datetime
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import ROOT

def main():
 stamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
 try:gpu=subprocess.check_output(['nvidia-smi','-L'],text=True).strip()
 except Exception as e:gpu=f'UNAVAILABLE: {e}'
 (ROOT/'GPU_INFO.txt').write_text('Captured UTC: '+stamp+'\n'+gpu+'\n')
 records=[]
 p=ROOT/'results/hw4_baseline.json'
 if p.exists():
  b=json.loads(p.read_text())
  if 'wall_seconds' in b:records.append(['HW4.a baseline',1,float(b['wall_seconds']),round(float(b['wall_seconds'])/3600,6),'results/hw4_baseline.json'])
 for fname in ['full_batch_scaling.csv','full_memory_comparison.csv','serving_metrics.csv','kv_context_sweep.csv']:
  p=ROOT/'results'/fname
  if not p.exists():continue
  with p.open() as f:
   for row in csv.DictReader(f):
    seconds=row.get('wall_seconds_including_evaluation') or row.get('elapsed_s')
    if seconds and row.get('status','ok')=='ok':
     label=fname+': '+(row.get('technique') or row.get('backend') or row.get('max_model_len') or '')
     records.append([label,1,round(float(seconds),3),round(float(seconds)/3600,6),'results/'+fname])
 with (ROOT/'gpu_hours.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['experiment','gpu_count','measured_seconds','gpu_hours','source']);w.writerows(records)
 log=ROOT/'RUN_LOG.txt'
 with log.open('a') as f:f.write(f'\n{stamp} | evidence captured | {gpu} | {len(records)} measured timing rows; see gpu_hours.csv\n')
 print('Captured GPU identity and',len(records),'timed rows. Note: sum may double-count overlapping experiment timings; do not use as total billed GPU-hours.')
if __name__=='__main__':main()
