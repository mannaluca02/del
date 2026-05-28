"""
Finales Modell — W&B Random Sweep.

Self-contained Script: Datenaufbereitung, Modell, Training, Sweep-Agent.
Die fixierte Architektur und die Hyperparameter-Distributionen sind aus den
Fazits H1-H10 im Notebook abgeleitet und dort ausführlich begründet.

Ausführung (einmal im Terminal, Laptop bleibt 48 h wach):
    cd src && caffeinate -i python sweep.py

Test-Modus (einzelner Run, 3 Epochen, fixe Hyperparameter):
    cd src && python sweep.py --test
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import wandb
from datasets import load_dataset, load_from_disk
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Globale Konstanten — identisch zum Notebook
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
DATA_DIR = Path("../data/intel-image-classification")
IMG_SIZE = 64
NUM_CLASSES = 6
PHASE2_EPOCHS = 50
WANDB_PROJECT = "del-mini-challenge"
SWEEP_COUNT = 600

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

CRITERION = nn.CrossEntropyLoss()


# ---------------------------------------------------------------------------
# Daten laden (einmalig, wird von allen Sweep-Runs geteilt)
# ---------------------------------------------------------------------------
def load_intel_dataset():
    if DATA_DIR.exists():
        return load_from_disk(str(DATA_DIR))
    ds = load_dataset("sfarrukhm/intel-image-classification")
    DATA_DIR.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(DATA_DIR))
    return ds


def compute_channel_stats(hf_train, train_indices):
    """Kanal-Statistiken nur über den Train-Split (kein Leakage)."""
    channel_sum = np.zeros(3)
    channel_sq_sum = np.zeros(3)
    n_pixels = 0
    for i in train_indices:
        img = np.array(hf_train[int(i)]["image"], dtype=np.float32) / 255.0
        channel_sum += img.sum(axis=(0, 1))
        channel_sq_sum += (img**2).sum(axis=(0, 1))
        n_pixels += img.shape[0] * img.shape[1]
    mean = channel_sum / n_pixels
    std = np.sqrt(channel_sq_sum / n_pixels - mean**2)
    return mean.tolist(), std.tolist()


class IntelImageDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        sample = self.hf_dataset[idx]
        return self.transform(sample["image"]), sample["label"]


def build_train_transform(mean, std, color_jitter, rotation_deg):
    """H6: RandomResizedCrop + HFlip + ColorJitter + Rotation."""
    ops = [
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.4, 1.0)),
        transforms.RandomHorizontalFlip(),
    ]
    if color_jitter > 0:
        ops.append(
            transforms.ColorJitter(
                brightness=color_jitter,
                contrast=color_jitter,
                saturation=color_jitter,
            )
        )
    if rotation_deg > 0:
        ops.append(transforms.RandomRotation(rotation_deg))
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
    return transforms.Compose(ops)


def build_eval_transform(mean, std):
    return transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


# ---------------------------------------------------------------------------
# Fixierte Architektur (aus H1-H10 begründet): DeepBNCNN
# ---------------------------------------------------------------------------
class DeepBNCNN(nn.Module):
    """4 Conv-Blöcke mit BatchNorm, MaxPool(2), 3x3 Kernel. FC mit Dropout.

    Parametrisiert auf base_filters, fc_hidden, dropout_rate. Alles andere fix
    (aus H1/H3/H4/H5/H8/H9).
    """

    def __init__(self, num_classes, base_filters, fc_hidden, dropout_rate):
        super().__init__()
        c1 = base_filters
        c2 = base_filters * 2
        c3 = base_filters * 4
        c4 = base_filters * 8

        self.features = nn.Sequential(
            nn.Conv2d(3, c1, kernel_size=3, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(c2, c3, kernel_size=3, padding=1),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(c3, c4, kernel_size=3, padding=1),
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        feat_size = IMG_SIZE // 16  # 4x MaxPool(2)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c4 * feat_size * feat_size, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(fc_hidden, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ---------------------------------------------------------------------------
# Training + Evaluation (mit Best-Val-Acc-Checkpointing)
# ---------------------------------------------------------------------------
def train_epoch(model, loader, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X, Y in loader:
        X, Y = X.to(DEVICE), Y.to(DEVICE)
        pred = model(X)
        loss = CRITERION(pred, Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (pred.argmax(dim=1) == Y).sum().item()
        total += Y.size(0)
    return total_loss / len(loader), correct / total


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for X, Y in loader:
        X, Y = X.to(DEVICE), Y.to(DEVICE)
        pred = model(X)
        total_loss += CRITERION(pred, Y).item()
        correct += (pred.argmax(dim=1) == Y).sum().item()
        total += Y.size(0)
    return total_loss / len(loader), correct / total


# ---------------------------------------------------------------------------
# Sweep-Run Funktion (wird vom wandb.agent aufgerufen)
# ---------------------------------------------------------------------------
# Cache für geteilte Ressourcen (einmal pro Prozess)
_DATA_CACHE = {}


def _prepare_data_once():
    if "hf_train" in _DATA_CACHE:
        return _DATA_CACHE
    ds = load_intel_dataset()
    labels = ds["train"]["label"]
    train_idx, val_idx = train_test_split(
        range(len(ds["train"])),
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=labels,
    )
    mean, std = compute_channel_stats(ds["train"], train_idx)
    _DATA_CACHE.update(
        hf_train=ds["train"].select(train_idx),
        hf_val=ds["train"].select(val_idx),
        mean=mean,
        std=std,
    )
    return _DATA_CACHE


def _run_training(cfg, epochs, run):
    """Gemeinsamer Trainings-Core für Sweep- und Test-Runs.

    cfg darf jedes Dict-artige Objekt sein (wandb.config oder ein Plain-Dict).
    """
    try:
        data = _prepare_data_once()
        train_tf = build_train_transform(
            data["mean"],
            data["std"],
            color_jitter=cfg["aug_color_jitter"],
            rotation_deg=cfg["aug_rotation_deg"],
        )
        eval_tf = build_eval_transform(data["mean"], data["std"])

        train_ds = IntelImageDataset(data["hf_train"], transform=train_tf)
        val_ds = IntelImageDataset(data["hf_val"], transform=eval_tf)

        train_loader = DataLoader(
            train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=0
        )

        torch.manual_seed(RANDOM_SEED)
        model = DeepBNCNN(
            num_classes=NUM_CLASSES,
            base_filters=cfg["base_filters"],
            fc_hidden=cfg["fc_hidden"],
            dropout_rate=cfg["dropout_rate"],
        ).to(DEVICE)

        optimizer = optim.Adam(
            model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
        )

        n_params = sum(p.numel() for p in model.parameters())
        wandb.log({"n_params": n_params}, step=0)

        best_val_acc = 0.0
        best_epoch = 0

        for epoch in range(epochs):
            train_loss, train_acc = train_epoch(model, train_loader, optimizer)
            val_loss, val_acc = evaluate(model, val_loader)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch

            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_acc_best": best_val_acc,
                    "best_epoch": best_epoch,
                }
            )

            if epoch % 10 == 0 or epoch == epochs - 1:
                print(
                    f"[{run.name}] Epoch {epoch:3d} | "
                    f"TrL {train_loss:.3f} TrA {train_acc:.3f} | "
                    f"VaL {val_loss:.3f} VaA {val_acc:.3f} | "
                    f"Best {best_val_acc:.4f}@{best_epoch}"
                )

        wandb.summary["val_acc_best"] = best_val_acc
        wandb.summary["best_epoch"] = best_epoch

    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        # OOM oder andere Laufzeitfehler sauber loggen, Agent weiterlaufen lassen
        print(f"[{run.name}] FAILED: {type(e).__name__}: {e}")
        wandb.summary["status"] = "failed"
        wandb.summary["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    finally:
        if DEVICE.type == "mps":
            torch.mps.empty_cache()
        elif DEVICE.type == "cuda":
            torch.cuda.empty_cache()


def train_sweep_run():
    """Vom wandb.agent aufgerufen. Config kommt aus dem Sweep."""
    run = wandb.init()
    try:
        _run_training(dict(wandb.config), epochs=PHASE2_EPOCHS, run=run)
    finally:
        wandb.finish()


# Fixe Hyperparameter für den --test Modus.
# Bewusst mittig im Search Space, damit der Test-Run repräsentativ ist.
TEST_CONFIG = {
    "lr": 5e-4,
    "batch_size": 32,
    "weight_decay": 5e-4,
    "dropout_rate": 0.5,
    "base_filters": 16,
    "fc_hidden": 128,
    "aug_color_jitter": 0.2,
    "aug_rotation_deg": 15,
}


def run_test():
    """Einzelner Sanity-Run: 3 Epochen, fixe Hyperparameter, W&B-Logging aktiv."""
    print("=" * 60)
    print("TEST-MODUS: einzelner Run, 3 Epochen, fixe Hyperparameter")
    print("=" * 60)
    for k, v in TEST_CONFIG.items():
        print(f"  {k}: {v}")
    print("=" * 60)

    run = wandb.init(
        project=WANDB_PROJECT,
        name="sweep-smoke-test",
        config={**TEST_CONFIG, "epochs": 3, "mode": "smoke-test"},
    )
    try:
        _run_training(TEST_CONFIG, epochs=3, run=run)
    finally:
        wandb.finish()
    print("\nTEST DONE — prüfe den Run auf W&B, dann den vollen Sweep starten.")


# ---------------------------------------------------------------------------
# Sweep-Konfiguration (Random Search, Hyperband Early-Termination)
# ---------------------------------------------------------------------------
sweep_config = {
    "method": "random",
    # Hyperband nutzt genau diese Metrik an den Bracket-Epochen. Deshalb das
    # instantane val_acc (nicht val_acc_best), damit overfittende Runs, deren
    # Val-Acc wieder abfaellt, korrekt abgebrochen werden. Das finale Ranking
    # im Sweep-UI bleibt der Peak, weil W&B den Maximalwert pro Run betrachtet.
    # val_acc_best wird weiterhin in wandb.summary geschrieben und ist die
    # Grundlage fuer die Post-Sweep-Auswertung.
    "metric": {"name": "val_acc", "goal": "maximize"},
    "parameters": {
        "lr": {
            "distribution": "log_uniform_values",
            "min": 1e-4,
            "max": 3e-3,
        },
        "batch_size": {"values": [8, 16, 32, 64, 128, 256]},
        "weight_decay": {
            "distribution": "log_uniform_values",
            "min": 1e-6,
            "max": 1e-2,
        },
        "dropout_rate": {
            "distribution": "uniform",
            "min": 0.1,
            "max": 0.9,
        },
        "base_filters": {"values": [8, 16, 24, 32, 48, 64]},
        "fc_hidden": {"values": [32, 64, 128, 256, 512]},
        "aug_color_jitter": {
            "distribution": "uniform",
            "min": 0.0,
            "max": 0.3,
        },
        "aug_rotation_deg": {"values": [0, 10, 15, 20]},
    },
    "early_terminate": {
        "type": "hyperband",
        "min_iter": 10,
        "eta": 3,
    },
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test",
        action="store_true",
        help="Einzelner Sanity-Run (3 Epochen, fixe Hyperparameter) statt vollem Sweep.",
    )
    args = parser.parse_args()

    print(f"Device: {DEVICE}")

    if args.test:
        run_test()
    else:
        print(f"Starting sweep on project '{WANDB_PROJECT}' with count={SWEEP_COUNT}")
        sweep_id = wandb.sweep(sweep_config, project=WANDB_PROJECT)
        print(f"Sweep ID: {sweep_id}")
        wandb.agent(sweep_id, function=train_sweep_run, count=SWEEP_COUNT)
