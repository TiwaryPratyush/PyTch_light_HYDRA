import torch
import lightning as L
import argparse
from pathlib import Path
from torchvision import models
from datamodules.catdogimage_datamodule import CatDogDataModule  # Update this import
from rich.console import Console
import torch.nn as nn
from torchmetrics import Accuracy
from models.catdog_classifier import CatDogClassifier  # Add this import

console = Console()

# Configuration
class CFG:
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_CLASSES = 2
    BATCH_SIZE = 32
    NUM_WORKERS = 4

def evaluate_model(ckpt_path, batch_size=CFG.BATCH_SIZE, num_classes=CFG.NUM_CLASSES, num_workers=CFG.NUM_WORKERS):
    console.print(f"[bold green]Loading model from checkpoint: {ckpt_path}[/bold green]")

    # Load the model using CatDogClassifier
    lightning_model = CatDogClassifier.load_from_checkpoint(ckpt_path, num_classes=num_classes)
    lightning_model.to(CFG.DEVICE)

    # Set model to evaluation mode
    lightning_model.eval()

    # Set up the data module
    data_module = CatDogDataModule(batch_size=batch_size, num_workers=num_workers)
    data_module.setup("test")

    # Create a validation dataloader
    val_loader = data_module.val_dataloader()

    # Create a Lightning Trainer for evaluation
    trainer = L.Trainer(accelerator="auto", logger=False)

    # Run validation
    console.print("[bold green]Running validation...[/bold green]")
    validation_results = trainer.validate(model=lightning_model, dataloaders=val_loader, verbose=True)

    # Print validation metrics
    console.print("[bold yellow]Validation metrics:[/bold yellow]")
    for key, value in validation_results[0].items():
        console.print(f"{key}: {value:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model on validation dataset")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Path to the model checkpoint (.ckpt file)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for validation"
    )
    parser.add_argument(
        "--num_classes", type=int, default=2, help="Number of classes for the model"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers for data loading"
    )
    args = parser.parse_args()

    evaluate_model(args.ckpt_path, args.batch_size, args.num_classes, args.num_workers)
