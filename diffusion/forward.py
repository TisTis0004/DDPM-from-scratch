import torch
from utils import extract
from schedule import make_schedule


def q_sample(x0, t, schedule, eps=None, device=None):
    """
    Docstring for q_sample

    :param x0: Description
    :param t: Description
    :param schedule: Description
    :param eps: Description
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    sqrt_alphas_bar = extract(schedule["sqrt_alphas_bar"], t, x0.shape)
    sqrt_one_minus_alphas_bar = extract(
        schedule["sqrt_one_minus_alphas_bar"], t, x0.shape
    )

    if eps is None:
        eps = torch.randn_like(x0)

    xt = sqrt_alphas_bar * x0 + sqrt_one_minus_alphas_bar * eps

    return xt, eps


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # image-like (random tensor) [64, 1, 28, 28]
    x0 = torch.randn([64, 1, 28, 28], device=device)

    # Three conditions of t
    t = torch.randint(0, 1000, (64,), device=device)  # timesteps tensor [64]
    # t = torch.zeros((64,), dtype=torch.int32, device=device)
    # t = torch.full((64,), 999, dtype=torch.int32, device=device)

    schedule = make_schedule()
    xt, eps = q_sample(x0, t, schedule, eps=None, device=device)

    print(x0[0] - xt[0])
