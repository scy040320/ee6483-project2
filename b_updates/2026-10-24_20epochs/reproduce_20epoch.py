"""Reproduce the five B 20-epoch controls. Run from a checkout/package root."""
import argparse,json,os,subprocess,sys,csv
from pathlib import Path
MATRIX=[('cnn128_noaug','baseline_cnn',128,False),('cnn128_aug','baseline_cnn',128,True),('resnet128_noaug','resnet18',128,False),('resnet128_aug','resnet18',128,True),('resnet224_aug','resnet18',224,True)]
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--data-dir',required=True,help='Contains train/, val/ and test/')
 p.add_argument('--output-dir',required=True,help='New directory; existing results are never overwritten')
 p.add_argument('--device',default='cuda',choices=['cuda','cpu','auto'])
 p.add_argument('--num-workers',type=int,default=4)
 a=p.parse_args();base=Path(a.output_dir).resolve();base.mkdir(parents=True,exist_ok=False)
 data=Path(a.data_dir).resolve();env=os.environ.copy();env.setdefault('MPLBACKEND','Agg')
 def run(args,log):
  cmd=[sys.executable,'-m',*map(str,args)]
  with (base/log).open('x',encoding='utf-8') as f:
   f.write('COMMAND: '+repr(cmd)+'\n');f.flush()
   subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=env,check=True)
 for name,model,size,aug in MATRIX:
  out=base/name
  args=['b_module.train','--data-dir',data,'--output-dir',out,'--model',model,'--mode','layer4','--epochs',20,'--batch-size',32,'--image-size',size,'--seed',42,'--lr',.001,'--backbone-lr',.0001,'--weight-decay',.0001,'--threads',4,'--num-workers',a.num_workers,'--device',a.device]
  if aug:args.append('--augment')
  print('Training:',name,flush=True);run(args,name+'_train.log')
  run(['b_module.evaluate','--checkpoint',out/'best_model.pt','--data-dir',data,'--output-dir',out/'evaluation','--device',a.device,'--sensitivity-exclude','val/cat/cat.12408.jpg'],name+'_evaluate.log')
 candidates=[]
 for name,model,size,aug in MATRIX:
  if model=='resnet18':
   m=json.loads((base/name/'evaluation/metrics.json').read_text());candidates.append((m['accuracy'],-m['loss'],name))
 winner=max(candidates)[2]
 run(['b_module.predict','--checkpoint',base/winner/'best_model.pt','--test-dir',data/'test','--sample-submission','report_assets/sampleSubmission.csv','--output-dir',base/'selected_submission','--device',a.device],'prediction.log')
 run(['b_module.compare',*[base/name for name,*_ in MATRIX],'--output',base/'comparison.csv'],'comparison.log')
 (base/'selection.json').write_text(json.dumps({'selected':winner,'rule':'maximum validation accuracy; ties minimum validation loss','epochs_per_run':20,'test_used_for_selection':False},indent=2))
 print('Complete. Selected:',winner,'Submission:',base/'selected_submission/submission.csv')
if __name__=='__main__':main()
