import torch


def make_schedule(T: int = 1000, beta_start=1e-4, beta_end=2e-4):
    """
    Create beta schedule aka "Physics Engine" for the diffusion

    :param T: Number of steps for schedule
    :param beta_start: min value for noise injection
    :param beta_end: max value for noise injection
    :param device: where the schedule will be stored
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # The amount of noise added on step t (from 0 to T)
    betas = torch.linspace(beta_start, beta_end, T, device=device)

    # How much signal do we keep from one step to the next
    alphas = 1 - betas

    # Total signal preserved after t noising steps (x0 to xt)
    # Key quantity for training later on
    alphas_bar = torch.cumprod(alphas, dim=0)

    # coefficient multiplying x0 in the closed-form
    # used in the forward q_sample (closed-form)
    sqrt_alphas_bar = torch.sqrt(alphas_bar)

    # coefficient multiplying ε in the closed-form
    # same as before, used in the forward q_sample (closed-form). Also in the reverse mean calculation
    sqrt_one_minus_alphas_bar = torch.sqrt(1 - alphas_bar)

    # rescaling factor used in reverse mean calculation
    sqrt_recip_alphas = 1 / torch.sqrt(alphas)

    # used in reverse sampling after computing the reverse mean to get xt-1
    sqrt_betas = torch.sqrt(betas)

    #

    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_bar": alphas_bar,
        "sqrt_alphas_bar": sqrt_alphas_bar,
        "sqrt_one_minus_alphas_bar": sqrt_one_minus_alphas_bar,
        "sqrt_recip_alphas": sqrt_recip_alphas,
        "sqrt_betas": sqrt_betas,
    }


if __name__ == "__main__":

    # Classic baseline parameters as in the paper
    T = 1000
    beta_start = 1e-4
    beta_end = 2e-4

    schedule = make_schedule(T, beta_start, beta_end)

    for k, v in schedule.items():
        print(f"{k}: {v.shape}")
        assert v.shape == torch.Size([T])
    print(f"All shapes are the same as the number of steps T: {T} ✅")

    # print(schedule["betas"][torch.randint(0, 999, (64,))])
    # print(schedule["betas"])
