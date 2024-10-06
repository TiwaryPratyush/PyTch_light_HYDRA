import torch
import torch.nn as nn
import lightning as L
from torchvision import models
import torchmetrics

class DogClassifier(L.LightningModule):
    def __init__(self, base_model='resnet50', num_classes=10, pretrained=True, lr=1e-3, weight_decay=1e-5, 
                 optimizer_type='Adam', scheduler_type='ReduceLROnPlateau', min_lr=1e-6, 
                 scheduler_args=None):
        super().__init__()
        
        # Exposing hyperparameters to args
        self.save_hyperparameters()

        # Load the base model (can be ResNet, VGG, etc.) based on the base_model parameter
        self.model = self._get_model(base_model, pretrained, num_classes)
        
        # Loss function and accuracy metric
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

    def _get_model(self, base_model, pretrained, num_classes):
        """
        Load the specified base model and modify the final fully connected layer for classification.
        """
        if base_model == 'resnet50':
            model = models.resnet50(pretrained=pretrained)
            num_ftrs = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(num_ftrs, num_classes)
            )
        elif base_model == 'vgg16':
            model = models.vgg16(pretrained=pretrained)
            num_ftrs = model.classifier[-1].in_features
            model.classifier[-1] = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(num_ftrs, num_classes)
            )
        else:
            raise ValueError(f"Base model {base_model} not supported.")
        
        return model

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
        # Select the optimizer
        if self.hparams.optimizer_type == 'Adam':
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay
            )
        elif self.hparams.optimizer_type == 'SGD':
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.hparams.lr,
                momentum=0.9,
                weight_decay=self.hparams.weight_decay
            )
        else:
            raise ValueError(f"Optimizer {self.hparams.optimizer_type} not supported.")

        # Select the scheduler
        if self.hparams.scheduler_type == 'ReduceLROnPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=self.hparams.scheduler_args.get('factor', 0.5),
                patience=self.hparams.scheduler_args.get('patience', 2),
                min_lr=self.hparams.min_lr  # Set the minimum learning rate
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val/loss'
                }
            }
        elif self.hparams.scheduler_type == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.hparams.scheduler_args.get('step_size', 10),
                gamma=self.hparams.scheduler_args.get('gamma', 0.1)
            )
            return [optimizer], [scheduler]
        else:
            raise ValueError(f"Scheduler {self.hparams.scheduler_type} not supported.")
