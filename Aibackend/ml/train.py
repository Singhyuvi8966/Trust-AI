#!/usr/bin/env python3
"""
Train the document tamper classifier on the synthetic dataset from make_dataset.py.

The model is a ResNet-18 (pretrained on ImageNet, then fine-tuned) with 5 outputs:
    genuine, name, dob, expiry, id_number
"Tampered" means the probability of "genuine" is below 0.5, and the most likely
of the other four classes tells WHICH field was edited.

Usage:
    python train.py                              # 8 epochs, needs data/ from make_dataset.py
    python train.py --epochs 12 --batch-size 16  # smaller batches if you run out of memory
    python train.py --no-pretrained              # if you have no internet for the ImageNet weights

Output: document_tamper.pt  (copy it into Aibackend/models/)  and  metrics.json
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(n_classes, pretrained):
    weights = None
    if pretrained:
        try:
            weights = models.ResNet18_Weights.DEFAULT
        except AttributeError:
            weights = None
    try:
        net = models.resnet18(weights=weights)
    except Exception as e:  # no internet, blocked download...
        print(f"Could not load the pretrained weights ({e}).\n"
              f"Training from scratch instead - it works, but needs more epochs and data.")
        net = models.resnet18(weights=None)
    net.fc = nn.Linear(net.fc.in_features, n_classes)
    return net


@torch.no_grad()
def evaluate(model, loader, device, classes):
    n = len(classes)
    g = classes.index("genuine")
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        probs.append(model(x.to(device)).softmax(1).cpu())
        labels.append(y)
    probs, labels = torch.cat(probs), torch.cat(labels)

    preds = probs.argmax(1)
    acc = (preds == labels).float().mean().item()

    cm = [[0] * n for _ in range(n)]
    for l, p in zip(labels.tolist(), preds.tolist()):
        cm[l][p] += 1

    actual_t = labels != g                     # really tampered
    pred_t = probs[:, g] < 0.5                 # the rule used by the backend
    tp = int((actual_t & pred_t).sum())
    fp = int((~actual_t & pred_t).sum())
    fn = int((actual_t & ~pred_t).sum())
    tn = int((~actual_t & ~pred_t).sum())

    # for correctly flagged forgeries: did we also name the right field?
    caught = actual_t & pred_t
    field_acc = None
    if caught.any():
        other = probs.clone()
        other[:, g] = -1
        field_acc = (other.argmax(1)[caught] == labels[caught]).float().mean().item()

    return {
        "accuracy": acc,
        "tamper_recall": tp / max(1, tp + fn),         # forgeries caught
        "tamper_precision": tp / max(1, tp + fp),
        "false_alarm_rate": fp / max(1, fp + tn),      # genuine cards wrongly flagged
        "field_accuracy_when_caught": field_acc,
        "confusion_matrix": cm,
    }


def print_report(name, r, classes):
    print(f"\n--- {name} ---")
    print(f"accuracy (5 classes):          {r['accuracy']:.3f}")
    print(f"forgeries caught (recall):     {r['tamper_recall']:.3f}")
    print(f"precision when flagging:       {r['tamper_precision']:.3f}")
    print(f"genuine wrongly flagged:       {r['false_alarm_rate']:.3f}")
    if r["field_accuracy_when_caught"] is not None:
        print(f"right field named when caught: {r['field_accuracy_when_caught']:.3f}")
    print("confusion matrix (rows = true, columns = predicted):")
    print("            " + " ".join(f"{c[:9]:>9}" for c in classes))
    for c, row in zip(classes, r["confusion_matrix"]):
        print(f"{c[:11]:>11} " + " ".join(f"{v:>9}" for v in row))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="document_tamper.pt")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3, help="peak learning rate (one-cycle schedule)")
    ap.add_argument("--img-h", type=int, default=256)
    ap.add_argument("--img-w", type=int, default=384)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    root = Path(args.data)
    if not (root / "train").exists():
        raise SystemExit(f"No {root}/train folder. Run:  python make_dataset.py")

    device = get_device()
    print("Using device:", device)

    size = (args.img_h, args.img_w)
    # no flips or crops: they would hide or destroy the tiny text artefacts we want the model to see
    train_tf = transforms.Compose([
        transforms.Resize(size),
        transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.1)], p=0.5),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])
    eval_tf = transforms.Compose([transforms.Resize(size), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    train_ds = datasets.ImageFolder(root / "train", train_tf)
    val_ds = datasets.ImageFolder(root / "val", eval_tf)
    classes = train_ds.classes
    if "genuine" not in classes:
        raise SystemExit("The data must contain a 'genuine' class folder.")
    print("Classes:", classes, "| train images:", len(train_ds), "| val images:", len(val_ds))

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.workers, pin_memory=pin)

    model = build_model(len(classes), pretrained=not args.no_pretrained).to(device)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(train_loader))

    best_acc, best_state = -1.0, None
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
            sched.step()
            loss_sum += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            total += len(y)
        val = evaluate(model, val_loader, device, classes)
        print(f"epoch {epoch:>2}/{args.epochs}  loss {loss_sum / total:.3f}  train acc {correct / total:.3f}  "
              f"val acc {val['accuracy']:.3f}  caught {val['tamper_recall']:.3f}  "
              f"false alarms {val['false_alarm_rate']:.3f}  ({time.time() - t0:.0f}s)")
        if val["accuracy"] > best_acc:
            best_acc = val["accuracy"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    torch.save({
        "state_dict": best_state, "classes": classes, "img_size": [args.img_h, args.img_w],
        "mean": MEAN, "std": STD, "arch": "resnet18",
    }, args.out)
    print(f"\nSaved the best model (val acc {best_acc:.3f}) to {args.out}")

    metrics = {"validation": evaluate(model, val_loader, device, classes), "classes": classes}
    print_report("validation", metrics["validation"], classes)
    if (root / "test").exists():
        test_ds = datasets.ImageFolder(root / "test", eval_tf)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, num_workers=args.workers)
        metrics["test"] = evaluate(model, test_loader, device, classes)
        print_report("test (images the model never saw)", metrics["test"], classes)
    Path("metrics.json").write_text(json.dumps(metrics, indent=2))
    print("\nMetrics saved to metrics.json")
    print("Next: copy", args.out, "into Aibackend/models/ and restart the backend.")


if __name__ == "__main__":
    main()
