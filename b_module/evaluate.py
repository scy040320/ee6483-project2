"""Independent checkpoint reload and labeled-validation evaluation."""
import argparse
import csv
from pathlib import Path
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from torch import nn
from src.common import resolve_device, set_seed, write_json
from src.data import build_loaders, manifest_hash
from src.engine import evaluate
from src.plots import plot_confusion, plot_examples
from .model import build


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    config = checkpoint['config']
    model = build(config, pretrained=False).to(device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()
    return model, checkpoint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data-dir', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--sensitivity-exclude', action='append', default=[], help='Val path excluded ONLY from supplementary accuracy')
    a = p.parse_args()
    out = Path(a.output_dir)
    if out.exists():raise FileExistsError(out)
    device = resolve_device(a.device)
    model, checkpoint = load_checkpoint(a.checkpoint, device)
    c = checkpoint['config']; set_seed(c['seed'])
    loaders, manifest = build_loaders(a.data_dir, batch_size=c['batch_size'], image_size=c['image_size'],
        seed=c['seed'], manifest_path=Path(a.checkpoint).parent / 'manifest.json')
    if manifest_hash(manifest) != c['manifest_sha256']:raise ValueError('Manifest changed')
    metrics, rows = evaluate(model, loaders['val'], nn.CrossEntropyLoss(), device, True)
    saved = checkpoint['validation']
    if abs(metrics['accuracy'] - saved['accuracy']) > 1e-8:
        raise RuntimeError('Reload accuracy differs from saved checkpoint; investigate before reporting')
    matrix = [[0, 0], [0, 0]]
    for r in rows:matrix[r['label']][r['prediction']] += 1
    classes = []
    for i, name in enumerate(('cat','dog')):
        tp = matrix[i][i]; predicted = sum(matrix[j][i] for j in (0,1)); actual = sum(matrix[i])
        precision = tp / predicted if predicted else 0.
        recall = tp / actual if actual else 0.
        classes.append(dict(name=name, precision=precision, recall=recall,
                            f1=2*precision*recall/(precision+recall) if precision+recall else 0.))
    excluded = set(a.sensitivity_exclude)
    if not excluded <= {r['path'] for r in rows}:raise ValueError('Unknown sensitivity exclusion')
    kept = [r for r in rows if r['path'] not in excluded]
    if not kept:raise ValueError('Sensitivity subset is empty')
    sensitivity = dict(excluded_paths=sorted(excluded), samples=len(kept),
        accuracy=sum(r['label']==r['prediction'] for r in kept)/len(kept))
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'metrics.json', dict(**metrics, per_class=classes, confusion_matrix=matrix,
        macro_f1=sum(r['f1'] for r in classes)/2, manifest_sha256=c['manifest_sha256'],
        checkpoint_epoch=checkpoint['epoch'], reload_accuracy_verified=True, sensitivity=sensitivity))
    with (out / 'predictions.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f); writer.writerow(['path','label','prediction','confidence','prob_cat','prob_dog'])
        for r in rows:writer.writerow([r['path'],r['label'],r['prediction'],r['confidence'],*r['probabilities']])
    plot_confusion(rows, ('cat','dog'), out / 'confusion_matrix.png')
    for correct, name in ((True,'correct'),(False,'incorrect')):
        plot_examples(rows, a.data_dir, ('cat','dog'), out / f'{name}_examples.png', correct, c['seed'])
    print(metrics)


if __name__ == '__main__':main()
