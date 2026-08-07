"""
Inference and visualization script for PixelSentinel Pix2Pix.
Loads the checkpoint, processes a sample, and plots the 
Input, Generated, and Ground-Truth Target images side-by-side.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from colorization.dataset import PixelSentinelDataset
from colorization.models.generator import Generator


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """
    Safely rescales any PyTorch image tensor to [0, 1] for Matplotlib display.
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    image = tensor.cpu().detach().numpy().transpose(1, 2, 0)
    
    # Robust Min-Max normalization for display
    img_min = image.min()
    img_max = image.max()
    
    if img_max - img_min > 1e-5:
        image = (image - img_min) / (img_max - img_min)
    else:
        image = np.clip(image, 0.0, 1.0)
        
    return image

def main() -> None:
    # Configuration paths & settings
    checkpoint_path = Path("checkpoints/latest_checkpoint.pth")# - 40epoch #checkpoints/latest_checkpoint.pth - 65 epoch
    data_root = Path("datasets")
    
    # Locate validation or test split
    val_dir = data_root / "val"
    if not val_dir.exists():
        val_dir = data_root / "validation"
    if not val_dir.exists():
        raise FileNotFoundError(
            f"Could not find validation/test directory under {data_root}. "
            "Please ensure your dataset split folder exists."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Initialize Generator architecture
    generator = Generator().to(device)

    # 2. Load the checkpoint safely
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if "generator_state_dict" in checkpoint:
        generator.load_state_dict(checkpoint["generator_state_dict"], strict=True)
        epoch = checkpoint.get("epoch", "Unknown")
        print(f"Successfully loaded checkpoint from Epoch {epoch}!")
    else:
        raise KeyError("Checkpoint does not contain 'generator_state_dict'.")

    generator.eval()

    # 3. Load dataset and grab a sample
    dataset = PixelSentinelDataset(str(val_dir), is_train=False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Fetch the first sample batch
    inputs, targets = next(iter(dataloader))
    inputs_dev = inputs.to(device)

    # 4. Run inference
    print("Running inference on sample...")
    with torch.no_grad():
        generated = generator(inputs_dev)

    # 5. Denormalize tensors to [0, 1] NumPy arrays (H, W, C) for Matplotlib
    gen_img = denormalize(generated[0])
    target_img = denormalize(targets[0])
    
    # Process input image channel (NIR/multispectral band)
    input_tensor = inputs[0]  # Shape: (C, H, W)
    if input_tensor.shape[0] >= 1:
        # Denormalize input channel from [-1, 1] -> [0, 1]
        input_img = ((input_tensor[0].cpu().detach().numpy() + 1.0) / 2.0)
        input_img = np.clip(input_img, 0.0, 1.0)
    else:
        input_img = input_tensor[0].cpu().detach().numpy()

    # 6. Plotting results side-by-side
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot Input IR/NIR channel
    axes[0].imshow(input_img, cmap="gray")
    axes[0].set_title("Input (Multispectral/IR Band)")
    axes[0].axis("off")

    # Plot Generated RGB Color Image
    axes[1].imshow(gen_img)
    axes[1].set_title("Generated RGB Color")
    axes[1].axis("off")

    # Plot Original Target Image
    axes[2].imshow(target_img)
    axes[2].set_title("Original Target RGB (Ground Truth)")
    axes[2].axis("off")

    plt.tight_layout()
    print("Displaying visualization window...")
    plt.show()


if __name__ == "__main__":
    main()