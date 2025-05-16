import os
import torch
from tqdm import tqdm
from torch import nn
from torchmetrics import Accuracy
from spikingjelly.activation_based import functional
from torch.utils.data import random_split, DataLoader
from early_stopping_pytorch import EarlyStopping
import argparse
from typing import Dict, Callable, Any

from datasets import MNISTRepeated
from models import SewResnet18

MODEL_MAP: Dict[str, Callable[[Any], nn.Module]] = {"sew_resnet": SewResnet18}


def train_epoch(
    model, dataloader, criterion, optimizer, accuracy_metric, epoch
):
    model.train()
    epoch_loss = 0
    epoch_preds = []
    epoch_targets = []
    dataloader_progbar = tqdm(dataloader, desc=f"Train Epoch {epoch+1}")
    for n, (img, target) in enumerate(dataloader_progbar):
        optimizer.zero_grad()
        out = model(img.transpose(0, 1)).mean(0)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_preds.append(out)
        epoch_targets.append(target)
        functional.reset_net(model)
        dataloader_progbar.set_postfix(loss=loss.item())
        if (n + 1) % 200 == 0:
            break
    epoch_preds = torch.cat(epoch_preds)
    epoch_targets = torch.cat(epoch_targets)
    epoch_loss /= len(dataloader)
    epoch_acc = accuracy_metric(epoch_preds, epoch_targets)
    print(
        f"Train Epoch {epoch+1}: Loss: {epoch_loss:.4f}",
        f"Accuracy: {epoch_acc:.4f}",
    )
    return epoch_loss, epoch_acc


def validate_epoch(model, dataloader, criterion, accuracy_metric, epoch):
    model.eval()
    epoch_loss = 0
    epoch_preds = []
    epoch_targets = []
    with torch.no_grad():
        dataloader_progbar = tqdm(
            dataloader, desc=f"Validation Epoch {epoch+1}"
        )
        for img, target in dataloader_progbar:
            out = model(img.transpose(0, 1)).mean(0)
            loss = criterion(out, target)
            epoch_loss += loss.item()
            epoch_preds.append(out)
            epoch_targets.append(target)
            functional.reset_net(model)
            dataloader_progbar.set_postfix(loss=loss.item())

    epoch_preds = torch.cat(epoch_preds)
    epoch_targets = torch.cat(epoch_targets)
    epoch_loss /= len(dataloader)
    epoch_acc = accuracy_metric(epoch_preds, epoch_targets)
    print(
        f"Validation Epoch {epoch+1}: Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}"
    )
    return epoch_loss, epoch_acc


def test_model(model, dataloader, accuracy_metric):
    model.eval()
    epoch_preds = []
    epoch_targets = []
    with torch.no_grad():
        for img, target in dataloader:
            out = model(img.transpose(0, 1)).mean(0)
            epoch_preds.append(out)
            epoch_targets.append(target)
            functional.reset_net(model)

    epoch_preds: torch.Tensor = torch.cat(epoch_preds)
    epoch_targets = torch.cat(epoch_targets)
    epoch_acc = accuracy_metric(epoch_preds, epoch_targets)
    print(f"Test Accuracy: {epoch_acc:.4f}")


def main(args):
    checkpoint_path = os.path.join(
        args.checkpoint_dir, f"{args.experiment_name}_best_{args.model}.pth"
    )
    model = MODEL_MAP[args.model](n_channels=1)
    functional.set_step_mode(model, step_mode="m")

    mnist_dataset_repeat_train_full = MNISTRepeated(
        root="./data", train=True, repeat=args.repeats, download=True
    )
    mnist_dataset_repeat_test = MNISTRepeated(
        root="./data", train=False, repeat=args.repeats, download=True
    )

    train_size = int(
        (1 - args.val_split) * len(mnist_dataset_repeat_train_full)
    )
    val_size = len(mnist_dataset_repeat_train_full) - train_size
    train_dataset, val_dataset = random_split(
        mnist_dataset_repeat_train_full, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        mnist_dataset_repeat_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    criterion = nn.CrossEntropyLoss()
    accuracy_metric = Accuracy(task="multiclass", num_classes=10)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    early_stopping = EarlyStopping(
        patience=args.patience, path=checkpoint_path, verbose=True
    )

    epoch_progbar = tqdm(range(args.epochs), desc="Epoch")
    for epoch in epoch_progbar:
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, accuracy_metric, epoch
        )
        val_loss, val_acc = validate_epoch(
            model, val_loader, criterion, accuracy_metric, epoch
        )

        epoch_progbar.set_postfix(
            train_loss=train_loss,
            val_loss=val_loss,
            train_acc=train_acc,
            val_acc=val_acc,
        )

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    model.load_state_dict(torch.load(checkpoint_path))
    test_model(model, test_loader, accuracy_metric)

    print("Training finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SNN Training Script")
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-2, help="Learning rate"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0, help="Weight decay"
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.1,
        help="Fraction of training data to use for validation",
    )
    parser.add_argument(
        "--repeats", type=int, default=10, help="Number of repeats for MNIST"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for data loading",
    )
    parser.add_argument(
        "--patience", type=int, default=5, help="Patience for early stopping"
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="default_experiment",
        help="Name of the experiment, used for checkpointing",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sew_resnet",
        choices=MODEL_MAP.keys(),
        help=f"Model architecture to use. Available options: {', '.join(MODEL_MAP.keys())}",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )

    args: argparse.Namespace = parser.parse_args()

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    # Instantiate the chosen model
    if args.model not in MODEL_MAP:
        raise ValueError(f"Model '{args.model}' not found in MODEL_MAP.")
    main(args)
