from colorization.dataset import PixelSentinelDataset
ds = PixelSentinelDataset("datasets/val", is_train=False)
inp, tgt = ds[0]
print("Target Min:", tgt.min().item(), "Target Max:", tgt.max().item())