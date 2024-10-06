import lightning as L
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
import os
import multiprocessing

class DogImageDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "data/dataset", batch_size: int = 8, num_workers: int = None, 
                 train_val_test_split: tuple = (0.7, 0.2, 0.1)):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers if num_workers is not None else multiprocessing.cpu_count()  # Dynamically set num_workers
        self.train_val_test_split = train_val_test_split  # Train, validation, test split percentages
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        """Downloads and prepares the Dog Breed Image dataset."""
        # Ensure the data directory exists
        os.makedirs(self.data_dir, exist_ok=True)

        # Authenticate and download the dataset from Kaggle
        api = KaggleApi()
        api.authenticate()

        # Dataset ID and download path
        dataset = 'khushikhushikhushi/dog-breed-image-dataset'

        # Download and unzip the dataset
        api.dataset_download_files(dataset, path=self.data_dir.parent, unzip=True)

    def setup(self, stage: str = None):
        """Splits the dataset into training, validation, and test sets."""
        # Point to the dataset directory containing all breed subdirectories
        data_path = self.data_dir

        # Define the full dataset (no predefined train/val/test split)
        full_dataset = ImageFolder(root=data_path, transform=self.train_transform)

        # Split dataset based on train, validation, test split percentages
        train_size = int(self.train_val_test_split[0] * len(full_dataset))
        val_size = int(self.train_val_test_split[1] * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size

        # Randomly split the dataset
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size]
        )

    def train_dataloader(self):
        """Returns the DataLoader for the training set."""
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        """Returns the DataLoader for the validation set."""
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        """Returns the DataLoader for the test set."""
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    @property
    def train_transform(self):
        """Transformations for the training set."""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),  # Augmentation: Horizontal flip
            transforms.RandomRotation(15),     # Augmentation: Random rotations
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Augmentation: Color jitter
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @property
    def val_transform(self):
        """Transformations for the validation set."""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
