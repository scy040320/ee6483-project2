"""Inference in sampleSubmission order; exports discrete labels, not probabilities."""
import argparse
import csv
import hashlib
from pathlib import Path
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from src.common import resolve_device, write_json
from src.data import build_transforms
from .evaluate import load_checkpoint
from .submission import read_ids, match_images, write_submission


class TestImages(Dataset):
    def __init__(self, paths, size):
        self.paths = paths; self.transform = build_transforms(size, augment=False)
    def __len__(self):return len(self.paths)
    def __getitem__(self, i):
        with Image.open(self.paths[i]) as image:return self.transform(image.convert('RGB'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--test-dir', required=True)
    p.add_argument('--sample-submission', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--device', choices=['auto','cpu','cuda'], default='auto')
    a = p.parse_args(); out = Path(a.output_dir)
    if out.exists():raise FileExistsError(out)
    ids = read_ids(a.sample_submission); paths = match_images(a.test_dir, ids)
    device = resolve_device(a.device); model, checkpoint = load_checkpoint(a.checkpoint, device)
    c = checkpoint['config']
    if c.get('experiment_status') != 'formal':raise ValueError('Refuse final submission from smoke run')
    if c['class_to_idx'] != {'cat':0,'dog':1}:raise ValueError('Wrong label map')
    loader = DataLoader(TestImages(paths,c['image_size']), batch_size=c['batch_size'], shuffle=False)
    labels, probabilities = [], []
    with torch.inference_mode():
        for inputs in loader:
            logits = model(inputs.to(device))
            if not torch.isfinite(logits).all():raise ValueError('Nonfinite prediction')
            labels.extend(logits.argmax(1).cpu().tolist())
            probabilities.extend(logits.softmax(1)[:,1].cpu().tolist())
    out.mkdir(parents=True, exist_ok=False)
    write_submission(out/'submission.csv', ids, labels)
    with (out/'test_predictions.csv').open('x',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(['id','filename','label','prob_dog'])
        writer.writerows((i,p.name,l,prob) for i,p,l,prob in zip(ids,paths,labels,probabilities))
    digest=hashlib.sha256()
    with Path(a.checkpoint).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):digest.update(block)
    write_json(out/'provenance.json',dict(checkpoint=str(Path(a.checkpoint).resolve()),
        checkpoint_sha256=digest.hexdigest(), checkpoint_epoch=checkpoint['epoch'],
        manifest_sha256=c['manifest_sha256'],class_to_idx=c['class_to_idx'],count=len(labels),
        test_accuracy=None,note='Test set has no ground truth; predictions only'))
    print(out/'submission.csv')


if __name__=='__main__':main()
