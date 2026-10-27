import sys,time,json,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import *
from sklearn.metrics import accuracy_score
from transformers import get_linear_schedule_with_warmup

def train(c=None):
 c=c or config();seed_all(int(c['seed']))
 if not torch.cuda.is_available():raise RuntimeError('CUDA GPU required for measured RTX 4090 baseline')
 (tr,va,te),(_,_,_,labels)=loaders(c)
 dev='cuda';m=make_model(c,len(labels)).to(dev)
 opt=torch.optim.AdamW(m.parameters(),lr=float(c['learning_rate']),weight_decay=float(c['weight_decay']))
 history=[];best=-1;best_epoch=0;start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
 for ep in range(int(c['epochs'])):
  m.train();losses=[]
  for b in tr:
   b={k:v.to(dev,non_blocking=True) for k,v in b.items()};opt.zero_grad(set_to_none=True)
   out=m(**b);out.loss.backward();opt.step();losses.append(out.loss.item())
  m.eval();v_losses=[];pred=[];gold=[]
  with torch.inference_mode():
   for b in va:
    b={k:v.to(dev) for k,v in b.items()};out=m(**b);v_losses.append(out.loss.item());pred.extend(out.logits.argmax(-1).cpu().tolist());gold.extend(b['labels'].cpu().tolist())
  row={'epoch':ep+1,'train_loss':float(np.mean(losses)),'val_loss':float(np.mean(v_losses)),'val_accuracy':accuracy_score(gold,pred)}
  history.append(row);print(row,flush=True)
  if row['val_accuracy']>best:best=row['val_accuracy'];best_epoch=ep+1;m.save_pretrained(ROOT/'results/bert_model')
 tok=AutoTokenizer.from_pretrained(c['model_name']);tok.save_pretrained(ROOT/'results/bert_model')
 torch.cuda.synchronize();duration=time.perf_counter()-start
 pd.DataFrame(history).to_csv(ROOT/'results/training_history.csv',index=False)
 summary={'wall_seconds':duration,'peak_gpu_gb':gpu_mem(),'best_val_epoch':best_epoch,'last_epoch':len(history),'epochs_after_best':len(history)-best_epoch,'best_val_accuracy':best,'gpu_name':torch.cuda.get_device_name(0)}
 (ROOT/'results/hw4_baseline.json').write_text(json.dumps(summary,indent=2));append_csv('results/hw4_baseline.csv',summary)
 print('Saved baseline',summary)
 return summary
if __name__=='__main__':train()
