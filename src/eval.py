import torch
import lightning as L
import argparse
from pathlib import Path
from torchvision import models
from datamodules.catdog import CatDogDataModule  # Ensure correct import for your data module
from rich.console import Console
import torch.nn as nn
from torchmetrics import Accuracy

console = Console()

# Configuration
class CFG:
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_CLASSES = 2  # Changed to 2 for cat and dog classes
    BATCH_SIZE = 32  # Increased batch size to match train.py
    NUM_WORKERS = 4

# Define the LightningModule for evaluation
#class MobileNetV2LightningModule(L.LightningModule):
class CatDogClassifier(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task="multiclass", num_classes=CFG.NUM_CLASSES)

    def forward(self, x):
        return self.model(x)

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        acc = self.accuracy(outputs.softmax(dim=-1), labels)
        self.log('val/loss', loss, prog_bar=True)
        self.log('val/acc', acc, prog_bar=True)
        return loss

def evaluate_model(ckpt_path, batch_size=CFG.BATCH_SIZE, num_classes=CFG.NUM_CLASSES, num_workers=CFG.NUM_WORKERS):
    console.print(f"[bold green]Loading model from checkpoint: {ckpt_path}[/bold green]")

    # Load ResNet50 model and modify the classifier
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(num_ftrs, num_classes)
    )

    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location=CFG.DEVICE)
    model.load_state_dict(checkpoint['state_dict'])

    # Wrap the model in a LightningModule
    #lightning_model = MobileNetV2LightningModule(model)
    lightning_model = CatDogClassifier(model)
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
