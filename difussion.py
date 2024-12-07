

# Beta schedule (linear or cosine)
beta_start = 0.0001
beta_end = 0.02
betas = torch.linspace(beta_start, beta_end, timesteps).to(device)

# Calculate alpha and cumulative alpha
alphas = 1.0 - betas
alpha_cumprod = torch.cumprod(alphas, dim=0)
alpha_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), alpha_cumprod[:-1]])