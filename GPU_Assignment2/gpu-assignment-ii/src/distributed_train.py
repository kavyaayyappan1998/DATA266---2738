# Launch ONLY on instructor-approved two-GPU topology:
# torchrun --standalone --nproc_per_node=2 src/distributed_train.py
import os,sys,time,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import *
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def run():
 if int(os.environ.get('WORLD_SIZE','1'))!=2:raise RuntimeError('Requires instructor-approved two-GPU pairing; do not simulate with one GPU.')
 dist.init_process_group('nccl');rank=dist.get_rank();local=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(local)
 c=config();seed_all(int(c['seed']));(tr,_,_),(_,_,_,labels)=loaders(c)
 sampler=DistributedSampler(tr.dataset,num_replicas=2,rank=rank,seed=int(c['seed']))
 dl=DataLoader(tr.dataset,batch_size=int(c['batch_size']),sampler=sampler,pin_memory=True)
 model=DDP(make_model(c,len(labels)).to(local),device_ids=[local]);opt=torch.optim.AdamW(model.parameters(),lr=float(c['learning_rate']))
 dist.barrier();start=time.perf_counter();steps=0
 for i,b in enumerate(dl):
  if i>=20:break
  b={k:v.to(local) for k,v in b.items()};opt.zero_grad(set_to_none=True);loss=model(**b).loss;loss.backward();opt.step();steps+=1
 torch.cuda.synchronize();dist.barrier();elapsed=time.perf_counter()-start
 times=[None,None];dist.all_gather_object(times,elapsed)
 if rank==0:
  throughput=steps*int(c['batch_size'])*2/max(times)
  row={'world_size':2,'batch_per_gpu':c['batch_size'],'steps':steps,'max_rank_seconds':max(times),'aggregate_samples_per_sec':throughput,'pairing':'FILL_IN_APPROVED_TOPOLOGY','interconnect':'MEASURE_AND_RECORD','communication_ms_per_step':'MEASURE_WITH_PROFILER','bandwidth_gbps':'MEASURE_WITH_PROFILER'}
  append_csv('results/distributed_metrics.csv',row);print(row)
 dist.destroy_process_group()
if __name__=='__main__':run()
