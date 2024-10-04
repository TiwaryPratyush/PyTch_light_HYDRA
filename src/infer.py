import argparse
import torch
import lightning as L
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
from rich.console import Console

# Console for logging messages
console = Console()

# Configuration for the inference
class CFG:
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_CLASSES = 10  # Make sure this matches your dataset's number of classes
    BATCH_SIZE = 16
    HEIGHT = 224
    WIDTH = 224

# Define the LightningModule for inference
class MobileNetV2LightningModule(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

def load_class_names(data_dir):
    """Load class names from the subdirectory names in the dataset directory."""
    class_names = sorted([d.name for d in Path(data_dir).iterdir() if d.is_dir()])
    return class_names

def load_model(ckpt_path, num_classes=CFG.NUM_CLASSES):
    """Load the MobileNetV2 model from the checkpoint."""
    console.print(f"[bold green]Loading model from checkpoint: {ckpt_path}[/bold green]")

    # Load MobileNetV2 model and modify the classifier
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)

    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location=CFG.DEVICE)
    new_state_dict = {}
    for key, value in checkpoint['state_dict'].items():
        new_key = key.replace("model.", "")  # Remove the 'model.' prefix if present
        new_state_dict[new_key] = value
    model.load_state_dict(new_state_dict, strict=False)

    # Wrap the model in a LightningModule
    lightning_model = MobileNetV2LightningModule(model)
    lightning_model.to(CFG.DEVICE)
    lightning_model.eval()
    console.print("[bold green]Model loaded successfully.[/bold green]")
    return lightning_model

def predict_single_image(model, image_path, class_names):
    """Run inference on a single image and return the predicted class name."""
    console.print(f"[bold green]Running inference on image: {image_path}[/bold green]")

    # Define transformations for the input image
    transform = transforms.Compose([
        transforms.Resize((CFG.HEIGHT, CFG.WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load the image and apply transformations
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(CFG.DEVICE)  # Add batch dimension

    # Run the image through the model
    with torch.no_grad():
        outputs = model(image_tensor)
        predicted_class_idx = outputs.argmax(dim=1).item()

    # Map predicted index to class name
    predicted_class_name = class_names[predicted_class_idx]
    return predicted_class_name

def main(ckpt_path, image_folder, num_classes=CFG.NUM_CLASSES, data_dir="data/dataset"):
    # Load class names based on folder structure
    class_names = load_class_names(data_dir)
    console.print(f"[bold yellow]Class names: {class_names}[/bold yellow]")

    # Load the model from the checkpoint
    model = load_model(ckpt_path, num_classes)

    # List of images in the provided folder
    image_folder_path = Path(image_folder)
    images = list(image_folder_path.glob("*.jpg")) + list(image_folder_path.glob("*.png"))  # Support jpg and png images

    if len(images) == 0:
        console.print(f"[bold red]No images found in {image_folder}.[/bold red]")
        return

    # Run inference on each image and print the result
    for image_path in images:
        predicted_class_name = predict_single_image(model, image_path, class_names)
        console.print(f"[bold yellow]Predicted class for {image_path.name}: {predicted_class_name}[/bold yellow]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on images using the trained model.")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Path to the model checkpoint (.ckpt file)",
    )
    parser.add_argument(
        "--image_folder",
        type=str,
        required=True,
        help="Path to the folder containing images for inference",
    )
    parser.add_argument(
        "--num_classes", type=int, default=CFG.NUM_CLASSES, help="Number of classes for the model"
    )
    parser.add_argument(
        "--data_dir", type=str, default="data/dataset", help="Path to the dataset folder containing class subdirectories"
    )
    args = parser.parse_args()

    main(args.ckpt_path, args.image_folder, num_classes=args.num_classes, data_dir=args.data_dir)


# import argparse
# import torch
# import lightning as L
# from torchvision import models, transforms
# from PIL import Image
# from pathlib import Path
# from rich.console import Console

# # Console for logging messages
# console = Console()

# # Configuration for the inference
# class CFG:
#     DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
#     NUM_CLASSES = 10  # Make sure this matches your dataset's number of classes
#     BATCH_SIZE = 16
#     HEIGHT = 224
#     WIDTH = 224

# # Define the LightningModule for inference
# class MobileNetV2LightningModule(L.LightningModule):
#     def __init__(self, model):
#         super().__init__()
#         self.model = model

#     def forward(self, x):
#         return self.model(x)

# def load_model(ckpt_path, num_classes=CFG.NUM_CLASSES):
#     """Load the MobileNetV2 model from the checkpoint."""
#     console.print(f"[bold green]Loading model from checkpoint: {ckpt_path}[/bold green]")

#     # Load MobileNetV2 model and modify the classifier
#     model = models.mobilenet_v2(pretrained=False)
#     model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)

#     # Load checkpoint
#     checkpoint = torch.load(ckpt_path, map_location=CFG.DEVICE)
#     new_state_dict = {}
#     for key, value in checkpoint['state_dict'].items():
#         new_key = key.replace("model.", "")  # Remove the 'model.' prefix if present
#         new_state_dict[new_key] = value
#     model.load_state_dict(new_state_dict, strict=False)

#     # Wrap the model in a LightningModule
#     lightning_model = MobileNetV2LightningModule(model)
#     lightning_model.to(CFG.DEVICE)
#     lightning_model.eval()
#     console.print("[bold green]Model loaded successfully.[/bold green]")
#     return lightning_model

# def predict_single_image(model, image_path):
#     """Run inference on a single image."""
#     console.print(f"[bold green]Running inference on image: {image_path}[/bold green]")

#     # Define transformations for the input image
#     transform = transforms.Compose([
#         transforms.Resize((CFG.HEIGHT, CFG.WIDTH)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])

#     # Load the image and apply transformations
#     image = Image.open(image_path).convert("RGB")
#     image_tensor = transform(image).unsqueeze(0).to(CFG.DEVICE)  # Add batch dimension

#     # Run the image through the model
#     with torch.no_grad():
#         outputs = model(image_tensor)
#         predicted_class = outputs.argmax(dim=1).item()

#     return predicted_class

# def main(ckpt_path, image_folder, num_classes=CFG.NUM_CLASSES):
#     # Load the model from the checkpoint
#     model = load_model(ckpt_path, num_classes)

#     # List of images in the provided folder
#     image_folder_path = Path(image_folder)
#     images = list(image_folder_path.glob("*.jpg")) + list(image_folder_path.glob("*.png"))  # Support jpg and png images

#     if len(images) == 0:
#         console.print(f"[bold red]No images found in {image_folder}.[/bold red]")
#         return

#     # Run inference on each image and print the result
#     for image_path in images:
#         predicted_class = predict_single_image(model, image_path)
#         console.print(f"[bold yellow]Predicted class for {image_path.name}: {predicted_class}[/bold yellow]")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Run inference on images using the trained model.")
#     parser.add_argument(
#         "--ckpt_path",
#         type=str,
#         required=True,
#         help="Path to the model checkpoint (.ckpt file)",
#     )
#     parser.add_argument(
#         "--image_folder",
#         type=str,
#         required=True,
#         help="Path to the folder containing images for inference",
#     )
#     parser.add_argument(
#         "--num_classes", type=int, default=CFG.NUM_CLASSES, help="Number of classes for the model"
#     )
#     args = parser.parse_args()

#     main(args.ckpt_path, args.image_folder, num_classes=args.num_classes)
