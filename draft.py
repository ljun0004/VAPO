# sequential [batch_size, num_particles]
if method_name == 'homotopy':
  
  # Get dimensions
  data_dim = sde.config.data.image_size * sde.config.data.image_size * sde.config.data.channels
  batch_size = sde.config.training.batch_size
  num_particles = sde.config.training.small_batch_size

  # Get the mini-batch
  samples_batch = batch[:batch_size]
  samples_batch = samples_batch.reshape(len(samples_batch), -1)

  # Construct time steps and step sizes
  time_range = (0,1)
  nfe_training = 100
  eps_time = 1
  time_steps = 1 - (np.exp(np.linspace(np.log(time_range[0] + eps_time), np.log(time_range[1] + eps_time), nfe_training + 1)) - eps_time)[::-1]
  step_sizes = np.append(time_steps[1:] - time_steps[:-1], 0)
  # plt.plot(time_steps, np.zeros_like(time_steps), 'x')

  # Sample from prior
  # torch.manual_seed(0)
  gaussian = torch.randn(batch_size, num_particles, data_dim)
  sigma_prior = 1
  samples_x = sigma_prior * gaussian.to(samples_batch.device)
  samples_x.requires_grad = True
  sigma_dec = 1

  # Prepare loss containers
  Cov = 0; Corr = 0; Norm = 0
  # Cov = []; Corr = []; Norm = []

  for _, step_size in zip(time_steps, step_sizes):

    with torch.enable_grad():
      # Predict scalar potential (FC)
      psi = model.module.psi_pooling(samples_x)
      # print(psi.mean())

      # Normalize potential by its mean
      # psi -= psi.detach().mean(dim=0, keepdim=True)

      # Compute (backpropagate) N-dimensional Poisson field (gradient)
      x_drift = torch.autograd.grad(psi, samples_x, torch.ones_like(psi), create_graph=True)[0]

    # Compute drift norm
    norm = x_drift.pow(2).sum(dim=-1).mean(dim=1)

    # Predict decoder std
    # sigma_dec = model.module.dec_sigma(perturbed_samples_x)

    with torch.no_grad():
      # Compute Normalized Innovation Squared (Gamma)
      # self_deviation = samples_x - samples_x.mean(dim=1, keepdim=True)
      # cross_deviation_norm = (samples_batch.unsqueeze(dim=1) * self_deviation).sum(dim=-1)
      # self_norm = samples_x.pow(2).sum(dim=-1)
      # self_norm_deviation = self_norm - self_norm.mean(dim=1, keepdim=True)
      # Gamma = torch.div(self_norm_deviation - 2*cross_deviation_norm, sigma_dec**2)

      # Compute Normalized Innovation Squared (Gamma)
      distance = samples_batch.unsqueeze(dim=1) - samples_x
      innovation = torch.div(distance, sigma_dec).pow(2).sum(dim=-1)
      Gamma = innovation - innovation.mean(dim=1, keepdim=True)

    # Compute sample correlation between potential and NIS
    cov = torch.mean(Gamma * psi.squeeze(dim=-1), dim=1)
    vars = torch.sum(Gamma.pow(2), dim=1) * torch.sum(psi.detach().squeeze(dim=-1).pow(2), dim=1)
    corr = cov / vars.sqrt()
    
    Cov += cov.mean(dim=0) / nfe_training
    Corr += corr.mean(dim=0) / nfe_training
    Norm += norm.mean(dim=0) / nfe_training

    # Posterior correction
    with torch.no_grad():
    # with torch.enable_grad():
      samples_x = samples_x + x_drift * step_size
      samples_x.requires_grad = True

  # Compute decoder log-likelihood
  # constant = math.log(2*math.pi)
  # nll = torch.log(sigma_dec.unsqueeze(dim=1).pow(2)) + constant
  # nll /= nll.detach().norm(p=2, dim=-1, keepdim=True)
  # nll = nll.sum(dim=-1).mean(dim=0)

  # Stack and reduce along time steps
  # Cov = torch.stack(Cov).mean(dim=0)
  # Corr = torch.stack(Corr).mean(dim=0)
  # Norm = torch.stack(Norm).mean(dim=0)
  Nll = torch.zeros(1)

  Loss = 0.5 * (Cov + Norm)

  return Loss, Cov, Corr, Norm, Nll


# perturb (sigmas along particles) [batch_size, num_particles]
if method_name == 'homotopy':
  
  data_dim = sde.config.data.image_size * sde.config.data.image_size * sde.config.data.channels
  # print(data_dim)
  batch_size = sde.config.training.batch_size
  num_particles = sde.config.training.small_batch_size

  # Get the mini-batch
  samples_batch = batch[:batch_size]
  samples_batch = samples_batch.reshape(len(samples_batch), -1)

  # Geometric sequence of sigmas
  # torch.manual_seed(0)
  sigma_dec = 1
  sigmas_range = (0.01, 1)
  eps_sigma = 0.5
  sigma_priors = torch.tensor(np.exp(np.linspace(np.log(sigmas_range[0] + eps_sigma), np.log(sigmas_range[1] + eps_sigma), num_particles))- eps_sigma)[:,None].float().to(samples_batch.device)
  # plt.plot(sigma_priors.cpu(), np.zeros_like(sigma_priors.cpu()), 'x')
  # plt.show()

  # Perturb data samples with gaussians
  gaussian = torch.randn(batch_size, num_particles, data_dim).to(samples_batch.device)
  samples_x = samples_batch.unsqueeze(dim=1) + gaussian * sigma_priors.to(samples_batch.device)
  samples_x.requires_grad = True

  # Prepare loss containers
  Cov = 0; Corr = 0; Norm = 0
  # Cov = []; Corr = []; Norm = []

  with torch.enable_grad():
    # Predict scalar potential (FC)
    psi = model.module.psi_pooling(samples_x)
    # print(psi.mean())

    # Normalize potential by its mean
    # psi -= psi.detach().mean(dim=0, keepdim=True)

    # Compute (backpropagate) N-dimensional Poisson field (gradient)
    x_drift = torch.autograd.grad(psi, samples_x, torch.ones_like(psi), create_graph=True)[0]

  # Compute drift norm
  norm = x_drift.pow(2).sum(dim=-1).mean(dim=1)

  # Predict decoder std
  # sigma_dec = model.module.dec_sigma(perturbed_samples_x)

  with torch.no_grad():
    # Compute Normalized Innovation Squared (Gamma)
    distance = samples_batch.unsqueeze(dim=1) - samples_x
    innovation = torch.div(distance, sigma_dec).pow(2).sum(dim=-1)
    Gamma = innovation - innovation.mean(dim=1, keepdim=True)

  # Compute sample correlation between potential and NIS
  cov = torch.mean(Gamma * psi.squeeze(dim=-1), dim=1)
  vars = torch.sum(Gamma.pow(2), dim=1) * torch.sum(psi.detach().squeeze(dim=-1).pow(2), dim=1)
  corr = cov / vars.sqrt()

  Cov += cov.mean(dim=0)
  Corr += corr.mean(dim=0)
  Norm += norm.mean(dim=0)

  # Compute decoder log-likelihood
  # constant = math.log(2*math.pi)
  # nll = torch.log(sigma_dec.unsqueeze(dim=1).pow(2)) + constant
  # nll /= nll.detach().norm(p=2, dim=-1, keepdim=True)
  # nll = nll.sum(dim=-1).mean(dim=0)

  Loss = 0.5 * (Cov + Norm)
  Nll = torch.zeros(1)

  return Loss, Cov, Corr, Norm, Nll


# perturb (sigmas along particles, rand_perm) [batch_size, num_particles]
if method_name == 'homotopy':
  
  # Get dimensions
  data_dim = sde.config.data.image_size * sde.config.data.image_size * sde.config.data.channels
  batch_size = sde.config.training.batch_size
  num_particles = sde.config.training.small_batch_size

  # Get the mini-batch
  samples_batch = batch[:batch_size]
  samples_batch = samples_batch.reshape(len(samples_batch), -1)

  # Geometric sequence of sigmas
  # torch.manual_seed(0)
  sigma_dec = 1
  sigmas_range = (0.01, 1)
  eps_sigma = 0.5
  sigma_priors = torch.tensor(np.exp(np.linspace(np.log(sigmas_range[0] + eps_sigma), np.log(sigmas_range[1] + eps_sigma), num_particles))- eps_sigma)[:,None].float().to(samples_batch.device)
  # plt.plot(sigma_priors.cpu(), np.zeros_like(sigma_priors.cpu()), 'x')
  # plt.show()

  # Perturb data samples with gaussians
  gaussian = torch.randn(batch_size, num_particles, data_dim).to(samples_batch.device)
  samples_x = samples_batch.unsqueeze(dim=1) + gaussian * sigma_priors.to(samples_batch.device)
  # Shuffle each batch of samples along particles
  rand_perms = torch.rand(batch_size, num_particles).argsort(dim=1)
  samples_x = samples_x.reshape(-1,data_dim)[rand_perms.view(-1)].reshape(samples_x.shape)

  num_train_particles = 1
  samples_train, samples_val = torch.split(samples_x, [num_train_particles, num_particles - num_train_particles], dim=1)

  # Prepare loss containers
  Cov = 0; Corr = 0; Norm = 0

  # Get model
  net_fn = mutils.get_predict_fn(sde, model, train=train, continuous=continuous)

  with torch.enable_grad():
    # Predict scalar potential (FC)
    samples_train.requires_grad = True
    psi_train = net_fn(samples_train.reshape(-1, sde.config.data.channels, sde.config.data.image_size, sde.config.data.image_size))
    psi_train = psi_train.view(batch_size, samples_train.shape[1], -1).squeeze(dim=-1)
    # Normalize potential by its mean
    # psi_train -= psi_train.detach().mean(dim=0, keepdim=True)
    # Compute (backpropagate) N-dimensional Poisson field (gradient)
    x_drift = torch.autograd.grad(psi_train, samples_train, torch.ones_like(psi_train), create_graph=True)[0]
    # Compute drift norm
    norm = x_drift.pow(2).sum(dim=-1).mean(dim=1)

  with torch.no_grad():
    psi_val = net_fn(samples_val.reshape(-1, sde.config.data.channels, sde.config.data.image_size, sde.config.data.image_size))
    psi_val = psi_val.view(batch_size, samples_val.shape[1], -1).squeeze(dim=-1)
    # Normalize potential by its mean
    # psi_val -= psi_val.detach().mean(dim=0, keepdim=True)

  # Predict decoder std
  # sigma_dec = model.module.dec_sigma(perturbed_samples_x)

  with torch.no_grad():
    # Compute Normalized Innovation Squared (Gamma)
    distance = samples_batch.unsqueeze(dim=1) - samples_x
    innovation = torch.div(distance, sigma_dec).pow(2).sum(dim=-1)
    Gamma = innovation - innovation.mean(dim=1, keepdim=True)

  # Compute sample correlation between potential and NIS
  psi = torch.cat([psi_train, psi_val], dim=1)
  cov = torch.mean(Gamma * psi, dim=1)
  vars = torch.sum(Gamma.pow(2), dim=1) * torch.sum(psi.detach().pow(2), dim=1)
  corr = cov / vars.sqrt()

  Cov += cov.mean(dim=0)
  Corr += corr.mean(dim=0)
  Norm += norm.mean(dim=0)

  # Compute decoder log-likelihood
  # constant = math.log(2*math.pi)
  # nll = torch.log(sigma_dec.unsqueeze(dim=1).pow(2)) + constant
  # nll /= nll.detach().norm(p=2, dim=-1, keepdim=True)
  # nll = nll.sum(dim=-1).mean(dim=0)

  Loss = 0.5 * (Cov + Norm)
  Nll = torch.zeros(1)

  return Loss, Cov, Corr, Norm, Nll


# perturb (sigmas along particles, select) [batch_size, 1]
if method_name == 'homotopy':
  
  # Get dimensions
  data_dim = sde.config.data.channels * sde.config.data.image_size * sde.config.data.image_size
  batch_size = sde.config.training.batch_size
  num_particles = sde.config.training.small_batch_size

  # Get the mini-batch
  samples_batch = batch[:batch_size]
  samples_batch = samples_batch.reshape(len(samples_batch), -1)

  # Geometric sequence of sigmas
  # torch.manual_seed(0)
  sigma_dec = 1
  sigmas_range = (0.01, 1)
  eps_sigma = 1
  sigma_priors = torch.tensor(np.exp(np.linspace(np.log(sigmas_range[0] + eps_sigma), np.log(sigmas_range[1] + eps_sigma), num_particles))- eps_sigma)[:,None].float().to(samples_batch.device)
  # plt.plot(sigma_priors.cpu(), np.zeros_like(sigma_priors.cpu()), 'x')
  # plt.show()

  # Perturb data samples with gaussians
  gaussian = torch.randn(batch_size, num_particles, data_dim).to(samples_batch.device)
  samples_x = samples_batch.unsqueeze(dim=1) + gaussian * sigma_priors.to(samples_batch.device)

  # Random select one particle
  rand = torch.randint(0, num_particles, (batch_size,))
  samples_train = samples_x[torch.arange(batch_size), rand]

  samples_train.requires_grad = True

  # Prepare losses
  Cov = 0; Corr = 0; Norm = 0

  with torch.enable_grad():
    # Predict scalar potential (FC)
    net_fn = mutils.get_predict_fn(sde, model, train=train, continuous=continuous)
    # samples_t = torch.zeros(batch_size).to(samples_batch.device)
    psi = net_fn(samples_train.view(-1, sde.config.data.channels, sde.config.data.image_size, sde.config.data.image_size))
    psi = psi.view(samples_train.shape).mean(dim=-1)
    # print(psi.mean())

    # Normalize potential by its mean
    # psi -= psi.detach().mean(dim=0, keepdim=True)

    # Compute (backpropagate) N-dimensional Poisson field (gradient)
    x_drift = torch.autograd.grad(psi, samples_train, torch.ones_like(psi), retain_graph=True)[0]

  # Compute drift norm
  Norm = torch.mean(x_drift.pow(2).sum(dim=-1), dim=0)

  # Predict decoder std
  # sigma_dec = model.module.dec_sigma(perturbed_samples_x)

  with torch.no_grad():
    # Compute Normalized Innovation Squared (Gamma)
    distance = samples_batch.unsqueeze(dim=1) - samples_x
    innovation = torch.div(distance, sigma_dec).pow(2).sum(dim=-1)
    Gamma = innovation - innovation.mean(dim=1, keepdim=True)

  # Compute sample covariance
  Gamma = Gamma[torch.arange(batch_size), rand]
  cov = Gamma * psi / num_particles
  Cov = torch.mean(cov, dim=0)

  # Compute sample covariance
  vars = Gamma.pow(2) * psi.detach().pow(2)
  corr = cov / vars.sqrt()
  Corr = torch.mean(corr, dim=0)

  # Compute decoder log-likelihood
  # constant = math.log(2*math.pi)
  # Nll = torch.log(sigma_dec.unsqueeze(dim=1).pow(2)) + constant
  # Nll /= Nll.detach().norm(p=2, dim=-1, keepdim=True)
  # Nll = Nll.sum(dim=-1).mean(dim=0)

  # Norm = torch.zeros(1)
  # Corr = torch.zeros(1)
  Nll = torch.zeros(1)

  Loss = 0.5 * (Cov + Norm)

  return Loss, Cov, Corr, Norm, Nll

# perturb (sigmas along batch) [batch_size, num_particles]
if method_name == 'homotopy':
  
  data_dim = sde.config.data.image_size * sde.config.data.image_size * sde.config.data.channels
  # print(data_dim)
  batch_size = sde.config.training.batch_size
  num_particles = sde.config.training.small_batch_size

  # Get the mini-batch
  samples_batch = batch[:batch_size]
  samples_batch = samples_batch.reshape(len(samples_batch), -1)

  # Geometric sequence of sigmas
  # torch.manual_seed(0)
  sigma_dec = 1
  sigmas_range = (0.01, 1)
  eps_sigma = 0.5
  sigma_priors = torch.tensor(np.exp(np.linspace(np.log(sigmas_range[0] + eps_sigma), np.log(sigmas_range[1] + eps_sigma), batch_size))- eps_sigma)[:,None].float().to(samples_batch.device)
  # plt.plot(sigma_priors.cpu(), np.zeros_like(sigma_priors.cpu()), 'x')
  # plt.show()

  # Perturb data samples with gaussians
  gaussian = torch.randn(batch_size, num_particles, data_dim).to(samples_batch.device)
  samples_x = samples_batch.unsqueeze(dim=1) + gaussian * sigma_priors.unsqueeze(dim=1).to(samples_batch.device)
  samples_x.requires_grad = True

  # Prepare loss containers
  Cov = 0; Corr = 0; Norm = 0
  # Cov = []; Corr = []; Norm = []

  with torch.enable_grad():
    # Predict scalar potential (FC)
    psi = model.module.psi_pooling(samples_x)
    # print(psi.mean())

    # Normalize potential by its mean
    # psi -= psi.detach().mean(dim=0, keepdim=True)

    # Compute (backpropagate) N-dimensional Poisson field (gradient)
    x_drift = torch.autograd.grad(psi, samples_x, torch.ones_like(psi), create_graph=True)[0]

  # Compute drift norm
  norm = x_drift.pow(2).sum(dim=-1).mean(dim=1)

  # Predict decoder std
  # sigma_dec = model.module.dec_sigma(perturbed_samples_x)

  with torch.no_grad():
    # Compute Normalized Innovation Squared (Gamma)
    distance = samples_batch.unsqueeze(dim=1) - samples_x
    innovation = torch.div(distance, sigma_dec).pow(2).sum(dim=-1)
    Gamma = innovation - innovation.mean(dim=1, keepdim=True)

  # Compute sample correlation between potential and NIS
  cov = torch.mean(Gamma * psi.squeeze(dim=-1), dim=1)
  vars = torch.sum(Gamma.pow(2), dim=1) * torch.sum(psi.detach().squeeze(dim=-1).pow(2), dim=1)
  corr = cov / vars.sqrt()

  Cov += cov.mean(dim=0)
  Corr += corr.mean(dim=0)
  Norm += norm.mean(dim=0)

  # Compute decoder log-likelihood
  # constant = math.log(2*math.pi)
  # nll = torch.log(sigma_dec.unsqueeze(dim=1).pow(2)) + constant
  # nll /= nll.detach().norm(p=2, dim=-1, keepdim=True)
  # nll = nll.sum(dim=-1).mean(dim=0)

  Loss = 0.5 * (Cov + Norm)
  Nll = torch.zeros(1)

  return Loss, Cov, Corr, Norm, Nll

# perturb (sigmas along batch) [batch_size, batch_size]
if method_name == 'homotopy':

  data_dim = sde.config.data.image_size * sde.config.data.image_size * sde.config.data.channels
  # print(data_dim)
  batch_size = sde.config.training.batch_size
  num_particles = sde.config.training.small_batch_size

  # Get the mini-batch
  samples_batch = batch[:batch_size]
  samples_batch = samples_batch.reshape(len(samples_batch), -1)

  # Sample gaussian priors
  # torch.manual_seed(0)
  sigma_dec = 1
  sigmas_range = (0.01, 1)
  eps_sigma = 0.5
  sigma_priors = torch.tensor(np.exp(np.linspace(np.log(sigmas_range[0] + eps_sigma), np.log(sigmas_range[1] + eps_sigma), batch_size))- eps_sigma)[:,None].float().to(samples_batch.device)
  # plt.plot(sigma_priors.cpu(), np.zeros_like(sigma_priors.cpu()), 'x')
  # plt.show()

  # Perturb data samples
  gaussian = torch.randn(batch_size, data_dim).to(samples_batch.device)
  samples_x = samples_batch + gaussian * sigma_priors.to(samples_batch.device)
  samples_x.requires_grad = True

  # Prepare loss containers
  Cov = 0; Corr = 0; Norm = 0
  # Cov = []; Corr = []; Norm = []

  with torch.enable_grad():
    # Predict scalar potential (FC)
    psi = model.module.psi_pooling(samples_x)
    # print(psi.mean())

    # Normalize potential by its mean
    # psi -= psi.detach().mean(dim=0, keepdim=True)

    # Compute (backpropagate) N-dimensional Poisson field (gradient)
    x_drift = torch.autograd.grad(psi, samples_x, torch.ones_like(psi), create_graph=True)[0]

  # Compute drift norm
  norm = x_drift.pow(2).sum(dim=-1).mean(dim=-1)

  # Predict decoder std
  # sigma_dec = model.module.dec_sigma(perturbed_samples_x)

  with torch.no_grad():
    # Compute Normalized Innovation Squared (Gamma)
    distance = samples_batch.unsqueeze(dim=1) - samples_x
    innovation = torch.div(distance, sigma_dec).pow(2).sum(dim=-1)
    Gamma = innovation - innovation.mean(dim=1, keepdim=True)

  # Compute sample correlation between potential and NIS
  cov = torch.mean(Gamma * psi.squeeze(dim=-1), dim=1)
  vars = torch.sum(Gamma.pow(2), dim=1) * torch.sum(psi.detach().squeeze(dim=-1).pow(2), dim=-1)
  corr = cov / vars.sqrt()

  Cov += cov.mean(dim=0)
  Corr += corr.mean(dim=0)
  Norm += norm.mean(dim=0)

  # Compute decoder log-likelihood
  # constant = math.log(2*math.pi)
  # nll = torch.log(sigma_dec.unsqueeze(dim=1).pow(2)) + constant
  # nll /= nll.detach().norm(p=2, dim=-1, keepdim=True)
  # nll = nll.sum(dim=-1).mean(dim=0)

  Loss = 0.5 * (Cov + Norm)
  Nll = torch.zeros(1)

  return Loss, Cov, Corr, Norm, Nll

