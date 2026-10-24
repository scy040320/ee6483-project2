"""Read complete evidence only; produce analysis and figures, never alter a run."""
from pathlib import Path
import csv,json,hashlib,statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
R=Path(__file__).resolve().parent
NAMES=['cnn128_noaug','cnn128_aug','resnet128_noaug','resnet128_aug','resnet224_aug']
LABELS=['CNN 128 / no aug.','CNN 128 / aug.','ResNet18 128 / no aug.','ResNet18 128 / aug.','ResNet18 224 / aug.']
F=R/'figures';F.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140,'savefig.dpi':200})
runs=[]
for name,label in zip(NAMES,LABELS):
 p=R/'experiments'/name
 c=json.loads((p/'config.json').read_text());m=json.loads((p/'metrics.json').read_text());e=json.loads((p/'evaluation/metrics.json').read_text())
 h=list(csv.DictReader((p/'history.csv').open()))
 assert c['epochs']==m['completed_epochs']==len(h)==20 and c['experiment_status']=='formal'
 assert [int(x['epoch']) for x in h]==list(range(1,21))
 best=max(h,key=lambda x:(float(x['val_accuracy']),-float(x['val_loss'])))
 assert int(best['epoch'])==m['best_epoch']
 assert abs(float(best['val_accuracy'])-m['best_val_accuracy'])<1e-12
 assert e['reload_accuracy_verified'] and abs(e['accuracy']-m['best_val_accuracy'])<1e-8
 runs.append(dict(name=name,label=label,config=c,training=m,evaluation=e,history=h,total_training_seconds=sum(float(x['seconds']) for x in h)))
assert len({x['config']['manifest_sha256'] for x in runs})==1
selected=max([x for x in runs if x['config']['model']=='resnet18'],key=lambda x:(x['evaluation']['accuracy'],-x['evaluation']['loss']))
cnn0,cnn1,rn0,rn1,rn2=runs
a=dict(runs=runs,selected=selected['name'],accuracy_gains_pp={'transfer_noaug_128':100*(rn0['evaluation']['accuracy']-cnn0['evaluation']['accuracy']),'transfer_aug_128':100*(rn1['evaluation']['accuracy']-cnn1['evaluation']['accuracy']),'augmentation_cnn_128':100*(cnn1['evaluation']['accuracy']-cnn0['evaluation']['accuracy']),'augmentation_resnet_128':100*(rn1['evaluation']['accuracy']-rn0['evaluation']['accuracy']),'resolution_224_vs_128_aug':100*(rn2['evaluation']['accuracy']-rn1['evaluation']['accuracy'])})
(R/'analysis.json').write_text(json.dumps(a,indent=2))
with (R/'comparison_20epochs.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['run','model','image_size','augment','epochs','seed','best_epoch','val_accuracy','val_loss','macro_f1','sensitivity_accuracy','training_seconds'])
 for r in runs:w.writerow([r['name'],r['config']['model'],r['config']['image_size'],r['config']['augment'],20,42,r['training']['best_epoch'],r['evaluation']['accuracy'],r['evaluation']['loss'],r['evaluation']['macro_f1'],r['evaluation']['sensitivity']['accuracy'],r['total_training_seconds']])
colors=['#5e6472','#b28c43','#007c91','#1c9b60','#9c3eb3']
fig,axs=plt.subplots(2,1,figsize=(7.5,5.6),layout='constrained')
for r,color in zip(runs,colors):
 h=r['history'];ep=[int(x['epoch']) for x in h]
 axs[0].plot(ep,[100*float(x['val_accuracy']) for x in h],label=r['label'],color=color,lw=1.7)
 axs[1].plot(ep,[float(x['val_loss']) for x in h],color=color,lw=1.7)
for ax in axs:ax.set(xlabel='Epoch',xticks=[1,5,10,15,20]);ax.grid(alpha=.18)
axs[0].set(ylabel='Validation accuracy (%)');axs[1].set(ylabel='Validation cross-entropy');axs[0].legend(fontsize=9,loc='upper center',bbox_to_anchor=(0.5,1.37),ncol=2,frameon=False)
fig.savefig(F/'all_20epoch_curves.png');plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(7.5,3.1),layout='constrained');h=selected['history'];ep=np.arange(1,21)
for prefix,label,color in [('train','Training','#007c91'),('val','Validation','#d07b29')]:
 axs[0].plot(ep,[100*float(x[prefix+'_accuracy']) for x in h],label=label,color=color)
 axs[1].plot(ep,[float(x[prefix+'_loss']) for x in h],label=label,color=color)
for ax in axs:ax.axvline(selected['training']['best_epoch'],color='#777777',ls=':',lw=1);ax.set(xlabel='Epoch',xticks=[1,5,10,15,20]);ax.grid(alpha=.18);ax.legend()
axs[0].set(ylabel='Accuracy (%)');axs[1].set(ylabel='Cross-entropy')
fig.suptitle('Selected 20-epoch candidate: '+selected['label'],fontsize=11)
fig.savefig(F/'selected_20epoch_curves.png');plt.close(fig)
fig,ax=plt.subplots(figsize=(7.5,3.8),layout='constrained');v=[100*r['evaluation']['accuracy'] for r in runs];bars=ax.bar(np.arange(5),v,color=colors,width=.65)
for b,r in zip(bars,runs):ax.text(b.get_x()+b.get_width()/2,b.get_height()+.35,f"{100*r['evaluation']['accuracy']:.2f}%\nepoch {r['training']['best_epoch']}",ha='center',fontsize=10)
ax.set(ylim=(0,114),ylabel='Best validation accuracy (%)',xticks=np.arange(5),xticklabels=['CNN128\nno aug.','CNN128\naug.','ResNet128\nno aug.','ResNet128\naug.','ResNet224\naug.']);ax.axvline(3.5,color='#aaaaaa',ls='--');ax.text(3.95,53,'Resolution\nexperiment',ha='center',fontsize=9,color='white');ax.grid(axis='y',alpha=.15)
fig.savefig(F/'controlled_comparison.png');plt.close(fig)
print(json.dumps({'selected':a['selected'],'gains_pp':a['accuracy_gains_pp']},indent=2))
