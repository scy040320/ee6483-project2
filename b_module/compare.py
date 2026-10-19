"""Summarize completed runs with explicit fairness metadata; never invent metrics."""
import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs',nargs='+')
    p.add_argument('--output',required=True)
    a=p.parse_args(); rows=[]
    for folder in a.runs:
        root=Path(folder)
        c=json.loads((root/'config.json').read_text(encoding='utf-8'))
        m=json.loads((root/'metrics.json').read_text(encoding='utf-8'))
        if c.get('experiment_status')!='formal':raise ValueError(f'Smoke run excluded: {folder}')
        if m['completed_epochs']!=c['epochs']:raise ValueError(f'Incomplete run: {folder}')
        rows.append(dict(run=folder,model=c['model'],mode=c['mode'],augment=c['augment'],
            seed=c['seed'],image_size=c['image_size'],batch_size=c['batch_size'],epochs=c['epochs'],
            lr=c['lr'],backbone_lr=c['backbone_lr'],weight_decay=c['weight_decay'],
            manifest_sha256=c['manifest_sha256'],
            augmentation=json.dumps(c['preprocessing']['train']['augmentation'],sort_keys=True),
            best_val_accuracy=m['best_val_accuracy'],best_val_loss=m['best_val_loss'],best_epoch=m['best_epoch']))
    # Size, split, seed and budget differences must be visible, never silently pooled.
    for key in ['manifest_sha256','seed','image_size','batch_size','epochs']:
        if len({r[key] for r in rows})>1:print(f'WARNING: differing {key}; separate controlled groups in report')
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    groups=defaultdict(list)
    keys=['model','mode','augment','image_size','batch_size','epochs','lr','backbone_lr',
          'weight_decay','manifest_sha256','augmentation']
    for row in rows:groups[tuple(row[k] for k in keys)].append(row)
    summaries=[]
    for key,group in groups.items():
        seeds=[r['seed'] for r in group]
        if len(seeds)!=len(set(seeds)):
            raise ValueError('Repeated seed in statistical group; not independent repetitions')
        values=[r['best_val_accuracy'] for r in group]
        summaries.append(dict(settings=dict(zip(keys,key)),n=len(group),seeds=seeds,
            mean_best_val_accuracy=statistics.mean(values),
            sample_std_best_val_accuracy=statistics.stdev(values) if len(values)>1 else None))
    with out.with_suffix('.statistics.json').open('x',encoding='utf-8') as f:
        json.dump(summaries,f,indent=2)



if __name__=='__main__':main()
