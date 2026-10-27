import json,random,time,csv
from pathlib import Path
import numpy as np,pandas as pd,torch,yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset,DataLoader
from transformers import AutoTokenizer,AutoModelForSequenceClassification
ROOT=Path(__file__).resolve().parents[1]
def config():
 c=yaml.safe_load((ROOT/'configs/experiment.yaml').read_text())
 if c['seed'] is None or c['slice'] is None: raise ValueError('Set instructor-provided seed and slice in configs/experiment.yaml first.')
 return c
def seed_all(s):
 random.seed(s);np.random.seed(s);torch.manual_seed(s)
 if torch.cuda.is_available():torch.cuda.manual_seed_all(s)
def load_data(c):
 p=ROOT/c['csv_path']
 if not p.exists():raise FileNotFoundError(f'Missing {p}; supply HW4.a Twitter CSV, do not substitute another dataset silently.')
 df=pd.read_csv(p)
 for col in (c['text_column'],c['label_column']):
  if col not in df:raise ValueError(f'Missing column {col}; CSV columns: {list(df.columns)}')
 start=int(c['slice']);end=min(start+5000,len(df))
 if start<0 or start>=len(df):raise ValueError('SLICE out of range')
 sub=df.iloc[start:end].copy();sub=sub.dropna(subset=[c['text_column'],c['label_column']])
 labels=sorted(sub[c['label_column']].unique().tolist(),key=str)
 label_map={str(k):i for i,k in enumerate(labels)}
 sub['_label']=sub[c['label_column']].map(lambda x:label_map[str(x)])
 # Lock data identity, source row range and split seed across HW4.a/HW4.5.
 fingerprint={'dataset_name':p.name,'dataset_sha256':__import__('hashlib').sha256(p.read_bytes()).hexdigest(),'slice':start,'seed':int(c['seed']),'end_zero_based_inclusive':end-1,'text_column':c['text_column'],'label_column':c['label_column']}
 lock=ROOT/'results/dataset_lock.json'
 lock.parent.mkdir(exist_ok=True)
 if lock.exists():
  prior=json.loads(lock.read_text())
  if prior!=fingerprint:raise ValueError('Dataset/SLICE/SEED changed since baseline. Preserve original HW4.a data and split; see results/dataset_lock.json.')
 else:lock.write_text(json.dumps(fingerprint,indent=2))
 # Explicit exact source row range, before missing-value exclusion.
 (ROOT/'results').mkdir(exist_ok=True)
 (ROOT/'results/row_range.json').write_text(json.dumps({'start_zero_based_inclusive':start,'end_zero_based_inclusive':end-1,'rows_before_dropna':end-start,'rows_after_dropna':len(sub),'label_map':label_map},indent=2))
 try:
  train,temp=train_test_split(sub,test_size=.30,random_state=int(c['seed']),stratify=sub['_label'])
  val,test=train_test_split(temp,test_size=.50,random_state=int(c['seed']),stratify=temp['_label'])
 except ValueError as e:raise ValueError(f'Stratified split failed: {e}. Check class counts or instructor requirements.')
 return train,val,test,label_map
class Tweets(Dataset):
 def __init__(self,frame,tokenizer,c):
  self.enc=tokenizer(frame[c['text_column']].astype(str).tolist(),padding='max_length',truncation=True,max_length=int(c['max_length']),return_tensors='pt')
  self.labels=torch.tensor(frame['_label'].tolist(),dtype=torch.long)
 def __len__(self):return len(self.labels)
 def __getitem__(self,i):return {**{k:v[i] for k,v in self.enc.items()},'labels':self.labels[i]}
def loaders(c,batch=None):
 tr,va,te,lmap=load_data(c);tok=AutoTokenizer.from_pretrained(c['model_name']);b=batch or int(c['batch_size'])
 return (DataLoader(Tweets(d,tok,c),batch_size=b,shuffle=i==0,pin_memory=True) for i,d in enumerate((tr,va,te))), (tr,va,te,lmap)
def make_model(c,n):return AutoModelForSequenceClassification.from_pretrained(c['model_name'],num_labels=n,ignore_mismatched_sizes=True)
def append_csv(path,row):
 p=ROOT/path;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('a',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(row));
  if p.stat().st_size==0:w.writeheader()
  w.writerow(row)
def gpu_mem():return round(torch.cuda.max_memory_allocated()/2**30,4) if torch.cuda.is_available() else 0
