import torch
from train import train_ddpm
from models.unet_v2 import UNetDDPM

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    T = 1000
    model = UNetDDPM(in_channels=1, out_channels=1, features=(64, 128), T=T).to(device)
    train_ddpm(T=T, epochs=2)
    print("Done Training for 2 epochs ✅")
