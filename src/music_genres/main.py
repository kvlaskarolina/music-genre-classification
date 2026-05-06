import copy
import json
import torch
import torchaudio.transforms as T
import sys
from torch.utils.data import DataLoader, random_split

from music_genres.dataset import (
    SAMPLE_RATE,
    GTZANDataset,
    download_dataset,
    load_files,
)
from music_genres.model import GenreCNN
from music_genres.train import evaluate, train_epoch

BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-3
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1
PATIENCE = 10


def build_transform() -> torch.nn.Module:
    return torch.nn.Sequential(
        T.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=2048, hop_length=512, n_mels=128),
        T.AmplitudeToDB(),
    )


def main_main(
    device: torch.device,
    /,
    should_train: bool = False,
    should_test: bool = False,
) -> None:
    print("Downloading dataset...")
    root = download_dataset()

    files = load_files(root)
    print(f"Found {len(files)} tracks across 10 genres")

    dataset = GTZANDataset(files, transform=build_transform())
    n = len(dataset)
    n_val = int(n * VAL_SPLIT)
    n_test = int(n * TEST_SPLIT)
    n_train = n - n_val - n_test
    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42),
    )

    # num_workers=0: macOS spawn-based multiprocessing can't import packages in worker processes
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=0)

    if should_train:
        model = GenreCNN(n_classes=10).to(device)
    else:
        print("Loading model.pt")
        model_state_dict = torch.load("model.pt", map_location=device)

        model = GenreCNN(n_classes=10)
        model.load_state_dict(model_state_dict)
        model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    if should_train:
        print("Starting train loop")

        best_val_loss = float("inf")
        best_state_dict = copy.deepcopy(model.state_dict())
        patience_counter = 0

        with open("training_log.jsonl", "w") as log:
            for epoch in range(1, EPOCHS + 1):
                train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
                val_loss, val_acc = evaluate(model, val_loader, criterion, device)
                scheduler.step()
                print(
                    f"Epoch {epoch:3d}/{EPOCHS}  "
                    f"train loss={train_loss:.4f} acc={train_acc:.3f}  "
                    f"val loss={val_loss:.4f} acc={val_acc:.3f}"
                )
                log.write(json.dumps({
                    "epoch": epoch,
                    "train_loss": train_loss, "train_acc": train_acc,
                    "val_loss": val_loss, "val_acc": val_acc,
                }) + "\n")
                log.flush()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state_dict = copy.deepcopy(model.state_dict())
                else:
                    patience_counter += 1
                    if patience_counter >= PATIENCE:
                        print(f"Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
                        break

        print("Training loop done. Saving model to model.pt")

        torch.save(best_state_dict, "model.pt")
        print("Saved model.pt")
    
    if should_test:
        print("Evaluating on test set...")
        _, test_acc = evaluate(model, test_loader, criterion, device)
        print(f"Test accuracy: {test_acc:.3f}")


def main() -> None:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")
    
    action = sys.argv[1] if len(sys.argv) >= 2 else None
    match action:
        case "train":
            main_main(device, should_train=True, should_test=True)
        case "evaluate":
            main_main(device, should_train=False, should_test=True)
        case _:
            raise Exception("available actions: train, evaluate")

