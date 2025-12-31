from pathlib import Path
import torch
import torch.nn.functional as F
from tqdm import tqdm

from dataloader import get_dataloader
from ..diffusion.schedule import make_schedule
from ..diffusion.forward import q_sample
from models.unet_v2 import UNetDDPM


def train_ddpm(
    T=1000,
    beta_start=1e-4,
    beta_end=1e-2,
    batch_size=128,
    lr=2e-4,
    epochs=10,
    device=None,
    checkpoints_root="checkpoints/mnist",
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Dataloader ----
    train_loader = get_dataloader(batch_size)

    # ---- schedule ----
    schedule = make_schedule(T, beta_start, beta_end)

    # ---- model ----
    model = UNetDDPM(in_channels=1, out_channels=1, features=(64, 128), T=T).to(device)
    model.train()

    # ---- optimizer ----
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # =========================================================================

    global_step = 0

    for epoch in range(epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for x0, _ in pbar:
            x0 = x0.to(device)  # [B,1,28,28] in [-1,1]
            B = x0.size(0)

            # ---- sample random timesteps (IMPORTANT) ----
            t = torch.randint(0, T, (B,), device=device, dtype=torch.long)  # [B]

            # ---- forward diffusion ----
            xt, eps = q_sample(x0, t, schedule)  # xt [B,1,28,28], eps [B,1,28,28]

            # ---- predict noise ----
            eps_pred = model(xt, t)  # [B,1,28,28]

            # ---- loss ----
            loss = F.mse_loss(eps_pred, eps)

            # ---- optimize ----
            optim.zero_grad(set_to_none=True)
            loss.backward()

            # optional but stabilizes
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optim.step()

            global_step += 1
            pbar.set_postfix(loss=float(loss.item()))

        # ---- checkpoint each epoch ----
        ckpt_path = Path(checkpoints_root) / f"ddpm_epoch_{epoch+1}.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "opt": optim.state_dict(),
                "epoch": epoch + 1,
                "global_step": global_step,
                "T": T,
                "beta_start": beta_start,
                "beta_end": beta_end,
            },
            ckpt_path,
        )

    return model
