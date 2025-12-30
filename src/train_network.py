import argparse
import os

import torch
from early_stopping_pytorch import EarlyStopping
from spikingjelly.activation_based import functional
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from tqdm import tqdm

from datasets import DatasetFactory
from models import MODEL_MAP
from utils import determine_input_size
import hydra
from omegaconf import DictConfig, OmegaConf

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_epoch(model, dataloader, criterion, optimizer, accuracy_metric, epoch):
    model.train()
    epoch_loss = 0
    epoch_preds = []
    epoch_targets = []
    dataloader_progbar = tqdm(dataloader, desc=f"Train Epoch {epoch+1}")
    for n, (img, target) in enumerate(dataloader_progbar):
        optimizer.zero_grad()
        out = model(img.transpose(0, 1).to(DEVICE)).mean(0)
        loss = criterion(out, target.to(DEVICE))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.cpu().item()
        epoch_preds.append(out)
        epoch_targets.append(target)
        functional.reset_net(model)
        dataloader_progbar.set_postfix(loss=loss.cpu().item())

    epoch_preds = torch.cat(epoch_preds).to(DEVICE)
    epoch_targets = torch.cat(epoch_targets).to(DEVICE)
    epoch_loss /= len(dataloader)
    epoch_acc = accuracy_metric(epoch_preds, epoch_targets).cpu().item()
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
        dataloader_progbar = tqdm(dataloader, desc=f"Validation Epoch {epoch+1}")
        for img, target in dataloader_progbar:
            out = model(img.transpose(0, 1).to(DEVICE)).mean(0)
            loss = criterion(out, target.to(DEVICE))
            epoch_loss += loss.cpu().item()
            epoch_preds.append(out)
            epoch_targets.append(target)
            functional.reset_net(model)
            dataloader_progbar.set_postfix(loss=loss.cpu().item())

    epoch_preds = torch.cat(epoch_preds).to(DEVICE)
    epoch_targets = torch.cat(epoch_targets).to(DEVICE)
    epoch_loss /= len(dataloader)
    epoch_acc = accuracy_metric(epoch_preds, epoch_targets).cpu().item()
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
            out = model(img.transpose(0, 1).to(DEVICE)).mean(0)
            epoch_preds.append(out)
            epoch_targets.append(target)
            functional.reset_net(model)

    epoch_preds: torch.Tensor = torch.cat(epoch_preds).to(DEVICE)
    epoch_targets = torch.cat(epoch_targets).to(DEVICE)
    epoch_acc = accuracy_metric(epoch_preds, epoch_targets).cpu().item()
    print(f"Test Accuracy: {epoch_acc:.4f}")


def main(cfg: DictConfig):
    output_size = DatasetFactory.num_classes(cfg.dataset.name)
    n_channels = determine_input_size(cfg.dataset.name, cfg.model.name)
    checkpoint_path = os.path.join(
        cfg.checkpoint_dir, f"{cfg.experiment_name}_best.pth"
    )
    model = MODEL_MAP[cfg.model.name](
        n_channels=n_channels,
        output_size=output_size,
        native_dvs_input=DatasetFactory.is_native_dvs(cfg.dataset.name),
    ).to(DEVICE)
    functional.set_step_mode(model, step_mode="m")

    dataset_repeat_train_full = DatasetFactory.create_dataset(
        name=cfg.dataset.name,
        normalize=cfg.dataset.normalize,
        root=cfg.data_dir,
        train=True,
        repeat=cfg.dataset.repeats,
        download=True,
    )
    dataset_repeat_test = DatasetFactory.create_dataset(
        name=cfg.dataset.name,
        normalize=cfg.dataset.normalize,
        root=cfg.data_dir,
        train=False,
        repeat=cfg.dataset.repeats,
        download=True,
    )

    train_size = int((1 - cfg.val_split) * len(dataset_repeat_train_full))
    val_size = len(dataset_repeat_train_full) - train_size
    train_dataset, val_dataset = random_split(
        dataset_repeat_train_full, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    test_loader = DataLoader(
        dataset_repeat_test,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    criterion = nn.CrossEntropyLoss().to(DEVICE)
    accuracy_metric = Accuracy(task="multiclass", num_classes=output_size).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    early_stopping = EarlyStopping(
        patience=cfg.patience, path=checkpoint_path, verbose=True
    )
    print("Sample shape:", next(iter(train_loader))[0].shape)
    epoch_progbar = tqdm(range(cfg.epochs), desc="Epoch")
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

    print(f"Training finished for {cfg.experiment_name}.")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def run_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    

    if not os.path.isabs(cfg.data_dir):
        cfg.data_dir = hydra.utils.to_absolute_path(cfg.data_dir)
        
    if not os.path.isabs(cfg.checkpoint_dir):
        cfg.checkpoint_dir = hydra.utils.to_absolute_path(cfg.checkpoint_dir)

    if not os.path.exists(cfg.checkpoint_dir):
        os.makedirs(cfg.checkpoint_dir)

    if cfg.model.name not in MODEL_MAP:
        raise ValueError(f"Model '{cfg.model.name}' not found in MODEL_MAP.")
    print(f"Using model: {cfg.model.name}, Dataset: {cfg.dataset.name}")

    main(cfg)

if __name__ == "__main__":
    run_app()
