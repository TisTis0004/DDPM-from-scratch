import torch
from schedule import make_schedule


def extract(schedule_tensor, t, x_shape):
    """
    Reshapes the a given schedule tensor to match the image.
    Given a tensor of [B] timesteps return it shapes [B, 1, 1, 1] so it can multiply images [B, C, H, W]

    For example:

    sqrt_alphas_bar[t] → shape [B]
    x0 → [B, C, H, W]

    We can't multiply them later since they have different shapes, and PyTorch won't broadcast them on its own.

    :param schedule_tensor: one of the schedule's items (betas, alphas, alphas_bar ...etc). [T]
    :param t: batch of integer timesteps. [B]
    :param x_shape: shape of image in the dataset [B, C, H, W]
    """

    B = t.shape[0]
    out = schedule_tensor.gather(0, t)  # [B]
    return out.view(
        B, *([1] * (len(x_shape) - 1))
    )  # [B, 1, 1, 1] based on the image shape passed


if __name__ == "__main__":
    pass
    # schedule = make_schedule()  # default values

    # t = torch.randint(0, 999, (64,), device="cuda")
    # print(schedule["betas"][t].shape)  # [B]
    # print(extract(schedule["betas"], t, (64, 1, 28, 28)).shape)  # [B, 1, 1, 1]
