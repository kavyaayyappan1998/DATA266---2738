import sys,time,argparse,gc
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import *
def probe(batch,steps=12,amp=False,checkpoint=False,optimizer8=False,accum=1):
 c=config();seed_all(int(c['seed']));(tr,_,_),(_,_,_,labels)=loaders(c,batch)
 torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();m=make_model(c,len(labels)).cuda();
 if checkpoint:m.gradient_checkpointing_enable();m.config.use_cache=False
 if optimizer8:
  import bitsandbytes as bnb
  opt=bnb.optim.AdamW8bit(m.parameters(),lr=float(c['learning_rate']))
 else:opt=torch.optim.AdamW(m.parameters(),lr=float(c['learning_rate']))
 scaler=torch.amp.GradScaler('cuda',enabled=amp);m.train();start=time.perf_counter();seen=0;step_count=0;loss_total=0
 try:
  opt.zero_grad(set_to_none=True)
  for i,b in enumerate(tr):
   if i>=steps*accum:break
   b={k:v.cuda(non_blocking=True) for k,v in b.items()}
   with torch.autocast('cuda',dtype=torch.float16,enabled=amp):out=m(**b);loss=out.loss/accum
   scaler.scale(loss).backward();seen+=len(b['labels']);loss_total+=float(loss.detach())
   if (i+1)%accum==0:scaler.step(opt);scaler.update();opt.zero_grad(set_to_none=True);step_count+=1
  torch.cuda.synchronize();elapsed=time.perf_counter()-start
  return {'batch':batch,'amp':amp,'checkpoint':checkpoint,'optimizer8':optimizer8,'accum':accum,'effective_batch':batch*accum,'steps':step_count,'samples':seen,'seconds':round(elapsed,3),'samples_per_second':round(seen/elapsed,3),'peak_gb':gpu_mem(),'seconds_per_epoch_estimate':round(elapsed/max(seen,1)*len(tr.dataset),3),'status':'ok'}
 except torch.cuda.OutOfMemoryError:
  return {'batch':batch,'amp':amp,'checkpoint':checkpoint,'optimizer8':optimizer8,'accum':accum,'effective_batch':batch*accum,'steps':step_count,'samples':seen,'seconds':None,'samples_per_second':None,'peak_gb':None,'seconds_per_epoch_estimate':None,'status':'OOM'}
 finally:
  del m,opt;gc.collect();torch.cuda.empty_cache()
def run():
 if not torch.cuda.is_available():raise RuntimeError('CUDA required')
 batch=8
 while batch<=2048:
  r=probe(batch);append_csv('results/batch_scaling.csv',r);print(r,flush=True)
  if r['status']=='OOM':break
  batch*=2
if __name__=='__main__':run()
