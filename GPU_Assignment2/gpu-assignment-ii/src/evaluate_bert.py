import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import *
from sklearn.metrics import classification_report,confusion_matrix,ConfusionMatrixDisplay
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt

def evaluate():
 c=config();(_,_,test,lmap)=load_data(c);tok=AutoTokenizer.from_pretrained(ROOT/'results/bert_model');ds=Tweets(test,tok,c)
 m=AutoModelForSequenceClassification.from_pretrained(ROOT/'results/bert_model').cuda().eval();pred=[];gold=[]
 with torch.inference_mode():
  for b in DataLoader(ds,batch_size=int(c['batch_size'])):
   b={k:v.cuda() for k,v in b.items()};pred+=m(**b).logits.argmax(-1).cpu().tolist();gold+=b['labels'].cpu().tolist()
 names=[str(k) for k,_ in sorted(lmap.items(),key=lambda kv:kv[1])]
 report=classification_report(gold,pred,labels=list(range(len(names))),target_names=names,output_dict=True,zero_division=0)
 (ROOT/'results/classification_report.json').write_text(json.dumps(report,indent=2))
 cm=confusion_matrix(gold,pred,labels=list(range(len(names))))
 fig,ax=plt.subplots(figsize=(7,6));ConfusionMatrixDisplay(cm,display_labels=names).plot(ax=ax);fig.tight_layout();fig.savefig(ROOT/'figures/confusion_matrix.png');plt.close(fig)
 worst=min(names,key=lambda n:report[n]['recall']);wi=names.index(worst)
 mistakes=[]
 for (_,row),g,p in zip(test.iterrows(),gold,pred):
  if g==wi and g!=p:mistakes.append({'text':str(row[c['text_column']]),'actual':names[g],'predicted':names[p],'diagnosis':'REQUIRED: add a one-line human diagnosis'})
 (ROOT/'results/worst_class_errors.json').write_text(json.dumps({'worst_class':worst,'examples':mistakes[:5],'available_mistakes':len(mistakes)},indent=2))
 if len(mistakes)<5:print('WARNING: fewer than five errors in worst class; document actual count.')
 hist=pd.read_csv(ROOT/'results/training_history.csv');fig,ax=plt.subplots();ax.plot(hist.epoch,hist.train_loss,label='Train loss');ax.plot(hist.epoch,hist.val_loss,label='Validation loss');ax.legend();ax.set_xlabel('Epoch');fig.savefig(ROOT/'figures/training_curves.png');plt.close(fig)
 print('Test report:',json.dumps(report,indent=2));return report
if __name__=='__main__':evaluate()
