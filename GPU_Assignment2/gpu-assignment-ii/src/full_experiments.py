"""Measured one-GPU, full-epoch independent comparisons. No fabricated results."""
import argparse, gc, json, time
from common import *
from sklearn.metrics import accuracy_score

def experiment(name, batch, *, amp=False, checkpoint=False, optimizer8=False, accum=1, epochs=None, fixed_steps=None):
 c=config(); seed_all(int(c['seed']))
 if not torch.cuda.is_available(): raise RuntimeError('CUDA GPU required')
 (train,val,test),(_,_,_,labels)=loaders(c,batch)
 model=make_model(c,len(labels)).cuda()
 if checkpoint:
  model.gradient_checkpointing_enable()
  model.config.use_cache=False
 if optimizer8:
  import bitsandbytes as bnb
  optimizer=bnb.optim.AdamW8bit(model.parameters(),lr=float(c['learning_rate']),weight_decay=float(c['weight_decay']))
 else: optimizer=torch.optim.AdamW(model.parameters(),lr=float(c['learning_rate']),weight_decay=float(c['weight_decay']))
 scaler=torch.amp.GradScaler('cuda',enabled=amp)
 torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
 start=time.perf_counter(); epoch_times=[]; seen=0; steps=0; history=[]
 try:
  for ep in range(epochs or int(c['epochs'])):
   model.train(); optimizer.zero_grad(set_to_none=True); epoch_seen=0; losses=[]; ep_start=time.perf_counter()
   for i,b in enumerate(train):
    if fixed_steps is not None and steps>=fixed_steps:break
    b={k:v.cuda(non_blocking=True) for k,v in b.items()}
    with torch.autocast('cuda',dtype=torch.float16,enabled=amp): out=model(**b); loss=out.loss
    scaler.scale(loss/accum).backward(); losses.append(float(loss.detach())); n=len(b['labels']); seen+=n;epoch_seen+=n
    if (i+1)%accum==0 or i+1==len(train):
     scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True);steps+=1
   torch.cuda.synchronize(); ep_sec=time.perf_counter()-ep_start; epoch_times.append(ep_sec)
   model.eval(); gold=[];pred=[]; vloss=[]
   with torch.inference_mode():
    for b in val:
     b={k:v.cuda(non_blocking=True) for k,v in b.items()}
     with torch.autocast('cuda',dtype=torch.float16,enabled=amp):o=model(**b)
     vloss.append(float(o.loss));gold.extend(b['labels'].cpu().tolist());pred.extend(o.logits.argmax(-1).cpu().tolist())
   history.append({'epoch':ep+1,'train_loss':float(np.mean(losses)),'val_loss':float(np.mean(vloss)),'val_accuracy':accuracy_score(gold,pred),'train_epoch_seconds':ep_sec,'train_samples':epoch_seen})
   if fixed_steps is not None and steps>=fixed_steps:break
  model.eval();gold=[];pred=[]
  with torch.inference_mode():
   for b in test:
    b={k:v.cuda(non_blocking=True) for k,v in b.items()}
    with torch.autocast('cuda',dtype=torch.float16,enabled=amp):o=model(**b)
    gold.extend(b['labels'].cpu().tolist());pred.extend(o.logits.argmax(-1).cpu().tolist())
  torch.cuda.synchronize()
  row={'technique':name,'batch':batch,'effective_batch':batch*accum,'amp':amp,'checkpoint':checkpoint,'optimizer8':optimizer8,'accum':accum,'epochs_completed':len(history),'optimizer_steps':steps,'train_samples':seen,'train_seconds':round(sum(epoch_times),3),'wall_seconds_including_evaluation':round(time.perf_counter()-start,3),'samples_per_second':round(seen/sum(epoch_times),3),'peak_gb':gpu_mem(),'eval_accuracy':accuracy_score(gold,pred),'status':'ok'}
  (ROOT/'results').mkdir(exist_ok=True)
  (ROOT/f'results/history_{name}_batch{batch}.json').write_text(json.dumps(history,indent=2))
 except torch.cuda.OutOfMemoryError:
  row={'technique':name,'batch':batch,'effective_batch':batch*accum,'status':'OOM'}
 finally:
  del model,optimizer;gc.collect();torch.cuda.empty_cache()
 return row

def save_rows(path,row):
 p=ROOT/path;p.parent.mkdir(parents=True,exist_ok=True)
 rows=pd.read_csv(p).to_dict('records') if p.exists() else []
 rows.append(row);pd.DataFrame(rows).to_csv(p,index=False)

def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['scaling','techniques'],required=True);p.add_argument('--max-batch',type=int,default=1024);p.add_argument('--accum',type=int,default=4);p.add_argument('--fixed-steps',type=int,default=None,help='For equal-step accumulation comparison');p.add_argument('--scan-technique-max',action='store_true',help='Run complete epoch batch scans for each independent technique');a=p.parse_args()
 c=config();base=int(c['batch_size'])
 if a.mode=='scaling':
  batch=base
  while batch<=a.max_batch:
   r=experiment('baseline_scaling',batch,epochs=1);save_rows('results/full_batch_scaling.csv',r);print(r,flush=True)
   if r['status']=='OOM':break
   batch*=2
 else:
  cases=[('baseline',{}),('mixed_precision',{'amp':True}),('gradient_checkpointing',{'checkpoint':True}),('gradient_accumulation',{'accum':a.accum}),('8bit_optimizer',{'optimizer8':True})]
  for name,kwargs in cases:
   sizes=[]
   if a.scan_technique_max and name != 'gradient_accumulation':
    b=base
    while b<=a.max_batch:
     sizes.append(b);b*=2
   else: sizes=[base]
   for b in sizes:
    try:r=experiment(name,b,fixed_steps=a.fixed_steps,**kwargs)
    except (ImportError,RuntimeError) as e:r={'technique':name,'batch':b,'status':'ERROR','details':str(e)}
    save_rows('results/full_memory_comparison.csv',r);print(r,flush=True)
    if r['status']!='ok':break
  # Select a combination only AFTER independent measurements; no hard-coded winner.
  from select_combination import select_and_run
  select_and_run(base, a.max_batch, a.fixed_steps, a.scan_technique_max, experiment, save_rows)

if __name__=='__main__':main()
