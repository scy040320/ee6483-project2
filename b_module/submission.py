"""Standard-library-only submission ID and label validation."""
import csv
from pathlib import Path


def read_ids(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ['id', 'label']:
            raise ValueError('Expected exactly id,label columns')
        rows = list(reader)
    ids = [r['id'] for r in rows]
    if len(ids) != 500 or len(set(ids)) != 500 or set(ids) != {str(i) for i in range(1,501)}:
        raise ValueError('Expected exactly 500 unique IDs, 1..500')
    return ids


def match_images(folder, ids):
    folder = Path(folder)
    if not folder.is_dir():raise FileNotFoundError(folder)
    by_id = {}
    for path in folder.rglob('*'):
        if not path.is_file() or path.suffix.lower() not in {'.jpg','.jpeg','.png','.bmp','.webp'}:
            continue
        if path.stem not in ids:raise ValueError(f'Unexpected image ID: {path.name}')
        if path.stem in by_id:raise ValueError(f'Duplicate image ID: {path.stem}')
        by_id[path.stem] = path
    if set(by_id) != set(ids):raise ValueError(f'Missing IDs: {sorted(set(ids)-set(by_id))[:10]}')
    return [by_id[i] for i in ids]


def write_submission(path, ids, labels):
    if len(ids) != len(labels) or any(type(x) is not int or x not in (0,1) for x in labels):
        raise ValueError('One integer binary prediction per ID required')
    with Path(path).open('x', newline='', encoding='utf-8') as f:
        writer = csv.writer(f); writer.writerow(['id','label']); writer.writerows(zip(ids,labels))
    read_ids(path)
