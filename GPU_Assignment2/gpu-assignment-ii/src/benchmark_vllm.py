"""HF/vLLM serving benchmark with transparent batch-latency and analytical KV accounting."""
import sys,time,json,argparse,subprocess,re,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import *

def gpu_used_gb():
 try:
  out=subprocess.check_output(['nvidia-smi','--query-compute-apps=used_gpu_memory','--format=csv,noheader,nounits'],text=True)
  return round(sum(float(x.strip()) for x in out.splitlines() if x.strip())/1024,3)
 except Exception:return None

def kv_spec(model):
 from transformers import AutoConfig
 cfg=AutoConfig.from_pretrained(model,trust_remote_code=False)
 layers=getattr(cfg,'num_hidden_layers',None)
 heads=getattr(cfg,'num_key_value_heads',getattr(cfg,'num_attention_heads',None))
 hidden=getattr(cfg,'hidden_size',None)
 attn_heads=getattr(cfg,'num_attention_heads',None)
 head_dim=getattr(cfg,'head_dim',None) or (hidden//attn_heads if hidden and attn_heads else None)
 if not all([layers,heads,head_dim]):raise ValueError('Cannot determine KV dimensions from model config')
 # Default vLLM auto dtype on CUDA may be BF16/FP16 (2 bytes); override using --kv-dtype.
 return {'layers':layers,'kv_heads':heads,'head_dim':head_dim}

def run(backend='vllm',batch=1,context=0,max_model_len=None,kv_dtype='auto'):
 c=config();prompt=c['serving_prompt'];model=c['serving_model'];repeats=int(c['serving_repeats']);max_t=int(c['serving_max_tokens']);lat=[];total_tokens=0
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained(model,trust_remote_code=False)
 if context:
  prefix=' word'*max(context*4,100)
  ids=tok.encode(prefix,add_special_tokens=False)[:context]
  prompt=tok.decode(ids,skip_special_tokens=True)+' '+prompt
 actual_context_tokens=len(tok.encode(prompt,add_special_tokens=False))
 ps=[prompt]*batch
 kv=kv_spec(model);dtype_bytes=2 if kv_dtype in ('auto','fp16','bf16') else 1
 kv_bytes_per_token=2*kv['layers']*kv['kv_heads']*kv['head_dim']*dtype_bytes
 # vLLM's memory reservation stays near its configured fraction; NOT a KV usage measurement.
 if backend=='vllm':
  from vllm import LLM,SamplingParams
  kwargs={'model':model,'trust_remote_code':False,'gpu_memory_utilization':0.80,'enforce_eager':True,'kv_cache_dtype':kv_dtype,'enable_prefix_caching':False}
  if max_model_len is not None:kwargs['max_model_len']=int(max_model_len)
  llm=LLM(**kwargs)
  sp=SamplingParams(max_tokens=max_t,temperature=0)
  def infer(prompts):
   outs=llm.generate(prompts,sp,use_tqdm=False)
   return sum(len(o.outputs[0].token_ids) for o in outs)
 else:
  from transformers import pipeline
  pipe=pipeline('text-generation',model=model,device=0,torch_dtype=torch.float16)
  def infer(prompts):
   out=pipe(prompts,max_new_tokens=max_t,do_sample=False,return_full_text=False)
   return sum(len(pipe.tokenizer.encode(item[0]['generated_text'] if isinstance(item,list) else item['generated_text'],add_special_tokens=False)) for item in out)
 infer(ps[:1]);torch.cuda.synchronize()
 start=time.perf_counter()
 for _ in range(repeats):
  t=time.perf_counter();n=infer(ps);torch.cuda.synchronize()
  lat.extend([(time.perf_counter()-t)*1000]*batch);total_tokens+=n
 elapsed=time.perf_counter()-start
 row={'backend':backend,'batch':batch,'requested_prefix_tokens':context,'actual_prompt_tokens':actual_context_tokens,'max_model_len':max_model_len,'requests':batch*repeats,'generated_tokens':total_tokens,'tokens_per_second':round(total_tokens/elapsed,3),'latency_definition':'BATCH COMPLETION TIME repeated for each request in the batch; not independent per-request latency or TTFT','p50_request_ms':round(float(np.percentile(lat,50)),3),'p95_request_ms':round(float(np.percentile(lat,95)),3),'peak_allocated_gb':gpu_mem() if backend=='hf' else None,'gpu_used_gb_from_nvidia_smi':gpu_used_gb(),'kv_bytes_per_token_per_sequence':kv_bytes_per_token,'theoretical_kv_gb_for_batch':round(kv_bytes_per_token*(actual_context_tokens+max_t)*batch/2**30,5),'kv_note':'Theoretical active-token KV allocation; excludes block padding, model weights and vLLM reserved/free cache; GPU process memory is not KV usage','elapsed_s':round(elapsed,3)}
 append_csv('results/serving_metrics.csv',row);print(row,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--backend',choices=['vllm','hf'],default='vllm');p.add_argument('--batch',type=int,default=1);p.add_argument('--context',type=int,default=0);p.add_argument('--max-model-len',type=int);p.add_argument('--kv-dtype',choices=['auto','fp16','bf16','fp8'],default='auto');a=p.parse_args();run(a.backend,a.batch,a.context,a.max_model_len,a.kv_dtype)
