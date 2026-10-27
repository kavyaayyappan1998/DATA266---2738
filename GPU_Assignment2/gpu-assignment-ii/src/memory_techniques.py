import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import *
from batch_scaling import probe

def run():
 c=config();base=int(c['batch_size']);cases=[('baseline',{}),('mixed_precision',{'amp':True}),('gradient_checkpointing',{'checkpoint':True}),('gradient_accumulation',{'accum':4}),('8bit_optimizer',{'optimizer8':True})]
 for name,kw in cases:
  try:r=probe(base,steps=12,**kw)
  except Exception as e:r={'batch':base,'amp':kw.get('amp',False),'checkpoint':kw.get('checkpoint',False),'optimizer8':kw.get('optimizer8',False),'accum':kw.get('accum',1),'effective_batch':base*kw.get('accum',1),'steps':0,'samples':0,'seconds':None,'samples_per_second':None,'peak_gb':None,'seconds_per_epoch_estimate':None,'status':f'ERROR: {e}'}
  r={'technique':name,**r};append_csv('results/memory_comparison.csv',r);print(r,flush=True)
 print('NOTE: short probes are not final accuracy or maximum-batch experiments. See notebook full-run instructions.')
if __name__=='__main__':run()
