"""Reusable training/evaluation loops; independent of the model architecture."""

import torch
from tqdm import tqdm


def _unpack_batch(batch):
    # CIFAR-10 normally gives (images, labels); our folder loader adds paths.
    return batch[0], batch[1]


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_sum, correct, count = 0.0, 0, 0
    progress = tqdm(loader, desc="train", leave=False)
    for batch in progress:
        inputs, labels = _unpack_batch(batch)
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        size = labels.size(0)
        loss_sum += loss.item() * size
        correct += (logits.argmax(1) == labels).sum().item()
        count += size
        progress.set_postfix(loss=f"{loss_sum / count:.4f}", accuracy=f"{correct / count:.4f}")
    if count == 0:
        raise ValueError("Empty training loader")
    return {"loss": loss_sum / count, "accuracy": correct / count, "samples": count}


@torch.no_grad()
def evaluate(model, loader, criterion, device, return_predictions=False):
    model.eval()
    loss_sum, correct, count = 0.0, 0, 0
    predictions = []
    for batch in tqdm(loader, desc="validation", leave=False):
        inputs, labels = _unpack_batch(batch)
        inputs, labels = inputs.to(device), labels.to(device)
        logits = model(inputs)
        loss = criterion(logits, labels)
        predicted = logits.argmax(1)
        size = labels.size(0)
        loss_sum += loss.item() * size
        correct += (predicted == labels).sum().item()
        count += size
        if return_predictions:
            probabilities = logits.softmax(1).cpu().tolist()
            targets, outputs = labels.cpu().tolist(), predicted.cpu().tolist()
            paths = batch[2] if len(batch) > 2 else [""] * size
            predictions.extend({"path": path, "label": label, "prediction": pred,
                                "confidence": prob[pred], "probabilities": prob}
                               for path, label, pred, prob in zip(paths, targets, outputs, probabilities))
    if count == 0:
        raise ValueError("Empty validation loader")
    metrics = {"loss": loss_sum / count, "accuracy": correct / count, "samples": count}
    return (metrics, predictions) if return_predictions else metrics
