"""Train B or an isolated A-model control using the shared loaders and engine."""
import argparse
import csv
import hashlib
import math
import os
import time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from torch import nn
from src.augmentation import describe_preprocessing, load_augmentation_config
from src.common import environment_info, resolve_device, set_seed, write_json
from src.data import build_loaders, manifest_hash
from src.engine import evaluate, train_one_epoch
from src.plots import plot_history
from .model import build


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--manifest', default='data/manifests/cats_dogs_seed42.json')
    p.add_argument('--model', choices=['resnet18', 'baseline_cnn'], default='resnet18')
    p.add_argument('--mode', choices=['head', 'layer4', 'all'], default='layer4')
    p.add_argument('--augment', action='store_true')
    p.add_argument('--augmentation-config', default='configs/augmentation.json')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--image-size', type=int, default=128)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--backbone-lr', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--smoke-per-class', type=int, help='Fixed subset; NEVER a formal result')
    a = p.parse_args()
    if a.threads < 1: p.error('threads must be positive')
    torch.set_num_threads(a.threads)
    if (a.epochs < 1 or a.batch_size < 2 or a.image_size < 32 or a.num_workers < 0 or
        any(not math.isfinite(v) or v <= 0 for v in (a.lr, a.backbone_lr)) or
        not math.isfinite(a.weight_decay) or a.weight_decay < 0 or
        (a.smoke_per_class is not None and a.smoke_per_class < 1)):
        p.error('Invalid training settings; batch size must be >=2')
    out = Path(a.output_dir)
    if out.exists():
        raise FileExistsError(out)
    set_seed(a.seed)
    device = resolve_device(a.device)
    settings = load_augmentation_config(a.augmentation_config)
    # Subsets are chosen inside the shared manifest, never by scanning new splits.
    manifest_path = a.manifest
    if a.smoke_per_class:
        import json, random
        from src.data import load_manifest
        manifest = load_manifest(a.manifest, a.data_dir)
        for split, rows in manifest['splits'].items():
            selected = []
            for label in (0, 1):
                candidates = [r for r in rows if r['label'] == label]
                selected += random.Random(f'42:{split}:{label}').sample(
                    candidates, min(a.smoke_per_class, len(candidates)))
            manifest['splits'][split] = sorted(selected, key=lambda r: r['path'])
        out.mkdir(parents=True, exist_ok=False)
        write_json(out / 'manifest.json', manifest)
        manifest_path = out / 'manifest.json'
    loaders, manifest = build_loaders(
        a.data_dir, batch_size=a.batch_size, image_size=a.image_size, seed=a.seed,
        augment=a.augment, num_workers=a.num_workers, manifest_path=manifest_path,
        pin_memory=device.type == 'cuda', augmentation_settings=settings)
    # Avoid a one-sample final minibatch with trainable BatchNorm at 32x32.
    if a.image_size == 32 and len(loaders['train'].dataset) % a.batch_size == 1:
        raise ValueError('Choose another batch size to avoid singleton BatchNorm batch')
    config = vars(a).copy()
    config.update(num_classes=2, dropout=0.3, class_to_idx=manifest['class_to_idx'],
                  manifest_sha256=manifest_hash(manifest), environment=environment_info(device),
                  weights='IMAGENET1K_V1' if a.model == 'resnet18' else None,
                  source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                      for folder in ('src', 'b_module') for p in sorted(Path(folder).glob('*.py'))},
                  determinism='fixed seeds; deterministic algorithms enabled with warn_only=True',
                  experiment_status='smoke_only' if a.smoke_per_class else 'formal',
                  selection_rule='maximum val accuracy; ties minimum val loss',
                  optimizer='AdamW', scheduler='none',
                  preprocessing={s: describe_preprocessing(a.image_size, a.augment and s == 'train', settings)
                                 for s in ('train', 'val')})
    # Download failures are errors; never silently replace pretrained weights with random ones.
    model = build(config, pretrained=a.model == 'resnet18').to(device)
    if a.model == 'resnet18':
        head, backbone = [], []
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                (head if name.startswith('network.fc.') else backbone).append(parameter)
        groups = [{'params': head, 'lr': a.lr}]
        if backbone:
            groups.append({'params': backbone, 'lr': a.backbone_lr})
    else:
        groups = [{'params': model.parameters(), 'lr': a.lr}]
    optimizer = torch.optim.AdamW(groups, weight_decay=a.weight_decay)
    criterion = nn.CrossEntropyLoss()
    config.update(total_parameters=sum(p.numel() for p in model.parameters()),
                  trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
    if not out.exists():
        out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'manifest.json', manifest)
    write_json(out / 'config.json', config)
    history, best, best_epoch = [], (-1., float('-inf')), None
    for epoch in range(1, a.epochs + 1):
        start = time.perf_counter()
        tr = train_one_epoch(model, loaders['train'], criterion, optimizer, device)
        va = evaluate(model, loaders['val'], criterion, device)
        if not all(math.isfinite(x) for x in (tr['loss'], va['loss'])):
            raise RuntimeError('Nonfinite loss; experiment is incomplete')
        row = dict(epoch=epoch, train_loss=tr['loss'], train_accuracy=tr['accuracy'],
                   val_loss=va['loss'], val_accuracy=va['accuracy'], lr=a.lr,
                   backbone_lr=a.backbone_lr if a.model == 'resnet18' and a.mode != 'head' else 0.,
                   seconds=time.perf_counter()-start)
        history.append(row)
        score = (va['accuracy'], -va['loss'])
        improved = score > best
        if improved:
            best, best_epoch = score, epoch
        checkpoint = dict(model_state_dict=model.state_dict(), config=config, epoch=epoch,
                          validation=va, best_epoch=best_epoch, best_val_accuracy=best[0])
        torch.save(checkpoint, out / 'last_model.pt')
        if improved:
            torch.save(checkpoint, out / 'best_model.pt')
        with (out / 'history.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(row)); writer.writeheader(); writer.writerows(history)
        write_json(out / 'metrics.json', dict(best_val_accuracy=best[0], best_val_loss=-best[1],
                   best_epoch=best_epoch, completed_epochs=epoch, experiment_status=config['experiment_status']))
        plot_history(history, out / 'learning_curves.png')
        print(row, flush=True)


if __name__ == '__main__':
    main()
