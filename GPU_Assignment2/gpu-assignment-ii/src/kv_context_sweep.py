"""Launch fresh vLLM processes with increasing context windows until failure."""
import sys,subprocess,re,csv,json,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import ROOT,config,append_csv

def main():
 c=config();batch=8
 # vLLM engine max_model_len is separate from actual prompt length.
 # Increase geometrically, up to model's supported limit / configured safety cap.
 from transformers import AutoConfig
 mc=AutoConfig.from_pretrained(c['serving_model'],trust_remote_code=False)
 limit=int(getattr(mc,'max_position_embeddings',32768))
 limit=min(limit,int(c.get('kv_sweep_cap',32768)))
 outdir=ROOT/'results/kv_logs';outdir.mkdir(parents=True,exist_ok=True)
 longest=None
 for length in [512,1024,2048,4096,8192,16384,32768,65536]:
  if length>limit:break
  # Reserve generation tokens, construct prompt close to the model limit.
  prompt_target=length-int(c['serving_max_tokens'])-32
  if prompt_target<1:continue
  cmd=[sys.executable,str(ROOT/'src/benchmark_vllm.py'),'--backend','vllm','--batch',str(batch),'--context',str(prompt_target),'--max-model-len',str(length)]
  started=time.perf_counter();p=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True)
  log=p.stdout+'\n'+p.stderr
  (outdir/f'context_{length}.log').write_text(log)
  # vLLM versions differ; preserve full startup logs and parse known block patterns.
  blocks=re.findall(r'(?:#\s*GPU\s*blocks|num_gpu_blocks|GPU blocks)\s*[:=]\s*(\d+)',log,re.I)
  block_size=re.findall(r'(?:block_size|block size)\s*[:=]\s*(\d+)',log,re.I)
  row={'max_model_len':length,'batch':batch,'requested_prefix_tokens':prompt_target,'status':'ok' if p.returncode==0 else 'FAILED','return_code':p.returncode,'gpu_blocks_reported':blocks[-1] if blocks else '', 'block_size_reported':block_size[-1] if block_size else '', 'log_file':f'results/kv_logs/context_{length}.log','elapsed_s':round(time.perf_counter()-started,2)}
  append_csv('results/kv_context_sweep.csv',row);print(row,flush=True)
  if p.returncode:break
  longest=length
 (ROOT/'results/kv_context_summary.json').write_text(json.dumps({'longest_successful_max_model_len':longest,'batch':batch,'tested_up_to_model_limit':limit,'caveat':'If no failure before model limit, report model context limit, not GPU OOM; logs preserve startup cache-block messages. GPU reserved memory is not KV-cache usage.'},indent=2))
if __name__=='__main__':main()
