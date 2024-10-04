# src/train.py

import os
import torch
import random
import lightning as L
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models  # Import torchvision models
from torchvision.datasets import ImageFolder
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

# Set configurations
class CFG:
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_CLASSES = 10  # Ensure this matches your dataset's number of classes
    EPOCHS = 2  # Number of epochs to train
    BATCH_SIZE = 16  # Reduce batch size to reduce memory footprint
    LR = 1e-4  # Learning rate
    NUM_WORKERS = 4
    SEED = 2024
    HEIGHT = 224
    WIDTH = 224
    VAL_SPLIT = 0.2  # Use 20% of the data for validation

# Set the seed for reproducibility
random.seed(CFG.SEED)
torch.manual_seed(CFG.SEED)

# Define data transformations
transform = transforms.Compose([
    transforms.Resize((CFG.HEIGHT, CFG.WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define dataset paths and load datasets using ImageFolder
data_dir = Path("data/dataset")
dataset = ImageFolder(root=data_dir, transform=transform)

# Split the dataset into training and validation sets
val_size = int(CFG.VAL_SPLIT * len(dataset))
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=CFG.BATCH_SIZE, shuffle=True, num_workers=CFG.NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=CFG.BATCH_SIZE, num_workers=CFG.NUM_WORKERS)

# Load MobileNetV2 model from torchvision
model = models.mobilenet_v2(pretrained=True)
# Modify the last layer to match the number of classes
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, CFG.NUM_CLASSES)

# Set up PyTorch Lightning module for MobileNetV2
class MobileNetV2Classifier(L.LightningModule):
    def __init__(self, model, num_classes=CFG.NUM_CLASSES, lr=CFG.LR):
        super().__init__()
        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss()
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        acc = (outputs.argmax(dim=1) == labels).float().mean()
        self.log('train/loss', loss)
        self.log('train/acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        acc = (outputs.argmax(dim=1) == labels).float().mean()
        self.log('val/loss', loss, prog_bar=True)
        self.log('val/acc', acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

# Initialize the LightningModule with MobileNetV2
lightning_model = MobileNetV2Classifier(model)

# Trainer configuration using PyTorch Lightning
trainer = L.Trainer(
    max_epochs=CFG.EPOCHS,
    accelerator="auto",
    precision=32,
    log_every_n_steps=1,
    callbacks=[
        ModelCheckpoint(
            dirpath="checkpoints/",
            filename="mobilenetv2_checkpoint",
            save_top_k=1,
            monitor="val/loss",
            mode="min"
        ),
        EarlyStopping(
            monitor="val/loss",
            patience=3,
            mode="min"
        )
    ],
)

# Start training the model with validation data
trainer.fit(lightning_model, train_loader, val_loader)
