"""
Utility script to check the epoch and metadata of the latest checkpoint.
"""

from pathlib import Path
import torch

def main() -> None:
    checkpoint_path = Path("checkpoints/latest_checkpoint.pth")
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    print(f"Reading checkpoint from: {checkpoint_path}")
    
    # Load checkpoint on CPU to avoid allocating GPU memory just to check metadata
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Extract metadata safely
    epoch = checkpoint.get("epoch", "Unknown")
    best_metric = checkpoint.get("best_metric", "None recorded")
    
    print("\n" + "=" * 40)
    print(f"📦 Checkpoint Metadata Summary:")
    print("=" * 40)
    print(f"🔹 Last Completed Epoch : {epoch}")
    print(f"🔹 Best Validation Metric: {best_metric}")
    
    # Check what state dictionaries are stored inside
    keys = list(checkpoint.keys())
    print(f"🔹 Stored Components    : {keys}")
    print("=" * 40)

if __name__ == "__main__":
    main()