"""Check environment, manifest files and test IDs before spending training time."""
import argparse
from collections import Counter
from pathlib import Path
from .submission import read_ids, match_images


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',required=True)
    p.add_argument('--sample-submission',required=True)
    p.add_argument('--verify-images',action='store_true')
    a=p.parse_args()
    import torch, torchvision
    from PIL import Image
    from src.data import load_manifest
    manifest=load_manifest('data/manifests/cats_dogs_seed42.json',a.data_dir)
    paths=match_images(Path(a.data_dir)/'test',read_ids(a.sample_submission))
    for split,rows in manifest['splits'].items():
        print(split,Counter(r['label'] for r in rows))
        paths += [Path(a.data_dir)/r['path'] for r in rows]
    if a.verify_images:
        for path in paths:
            with Image.open(path) as image:image.verify()
            with Image.open(path) as image:image.convert('RGB').load()
    print(dict(torch=str(torch.__version__),torchvision=str(torchvision.__version__),
        cuda=torch.cuda.is_available(),gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        checked_images=len(paths) if a.verify_images else 0))


if __name__=='__main__':main()
