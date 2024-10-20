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
    NUM_CLASSES = 2  # Changed to 2 for cat and dog classes
    EPOCHS = 5  # Changed to 5 epochs
    BATCH_SIZE = 32  # Increased batch size since we have fewer classes
    LR = 3e-4  # Slightly increased learning rate
    NUM_WORKERS = 4
    SEED = 2024
    HEIGHT = 224
    WIDTH = 224
    VAL_SPLIT = 0.2  # Kept the same 80/20 split

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
data_dir = Path("src/data/dataset")  # Updated path
dataset = ImageFolder(root=data_dir, transform=transform)

# Split the dataset into training and validation sets
val_size = int(CFG.VAL_SPLIT * len(dataset))
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Modify the DataLoader configurations
train_loader = DataLoader(train_dataset, batch_size=CFG.BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=CFG.BATCH_SIZE, num_workers=0)

# Load MobileNetV2 model from torchvision
model = models.mobilenet_v2(pretrained=True)
# Modify the last layer to match the new number of classes (2)
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

# Modify the Trainer configuration
trainer = L.Trainer(
    max_epochs=CFG.EPOCHS,
    accelerator="auto",
    precision=32,
    log_every_n_steps=5,  # Changed to match the command line argument
    callbacks=[
        ModelCheckpoint(
            dirpath="checkpoints/",
            filename="mobilenetv2_cat_dog_checkpoint",
            save_top_k=1,
            monitor="val/loss",
            mode="min"
        ),
        EarlyStopping(
            monitor="val/loss",
            patience=2,
            mode="min"
        )
    ],
    num_sanity_val_steps=0,  # Add this line to skip sanity validation check
)

# Start training the model with validation data
trainer.fit(lightning_model, train_loader, val_loader)
