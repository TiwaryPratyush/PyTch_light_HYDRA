import torch
import torch.nn as nn
import lightning as L
from torchvision.models import resnet50, ResNet50_Weights
import torchmetrics

class DogClassifier(L.LightningModule):
    def __init__(self, num_classes=10, lr=1e-3, weight_decay=1e-5):
        super().__init__()
        self.save_hyperparameters()

        # Load the pre-trained ResNet50 model using the correct weights
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)

        # **Modify convolutional layers to match the checkpoint**
        # Replace all 1x1 convolutions in bottleneck layers with 3x3 convolutions
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d) and module.kernel_size == (1, 1) and module.stride == (1, 1):
                module.kernel_size = (3, 3)
                module.padding = (1, 1)
                # Adjust weight dimensions if necessary
                module.weight = nn.Parameter(torch.randn(
                    module.out_channels, module.in_channels, 3, 3))
                # Remove bias if not present
                if module.bias is not None:
                    module.bias = nn.Parameter(torch.zeros(module.out_channels))

        # Modify the final fully connected layer
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(num_ftrs, num_classes)
        )

        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        acc = self.accuracy(outputs.softmax(dim=-1), labels)
        self.log('train/loss', loss)
        self.log('train/acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        acc = self.accuracy(outputs.softmax(dim=-1), labels)
        self.log('val/loss', loss, prog_bar=True)
        self.log('val/acc', acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        acc = self.accuracy(outputs.softmax(dim=-1), labels)
        self.log('test/loss', loss)
        self.log('test/acc', acc)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=2
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val/loss'
        }
