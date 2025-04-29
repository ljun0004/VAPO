import ml_collections
import torch
import math


def get_default_configs():
  config = ml_collections.ConfigDict()

  # training
  config.training = training = ml_collections.ConfigDict()
  config.training.batch_size = 128
  training.n_iters = 2000001
  training.snapshot_freq = 50000
  training.log_freq = 50
  training.eval_freq = math.inf
  ## store additional checkpoints for preemption in cloud computing environments
  training.snapshot_freq_for_preemption = 10000
  ## produce samples at each snapshot.
  training.snapshot_sampling = True
  training.continuous = True
  training.reduce_mean = False
  training.M = 291

  # sampling
  config.sampling = sampling = ml_collections.ConfigDict()
  sampling.n_steps_each = 1
  sampling.noise_removal = True
  sampling.probability_flow = False
  sampling.snr = 0.16
  sampling.vs = False
  sampling.N = 1000
  sampling.z_exp = 5
  ## new args
  sampling.t_end = 1.5
  sampling.rtol = 1e-5
  sampling.atol = 1e-5
  sampling.eps_z = 1e-10

  # evaluation
  config.eval = evaluate = ml_collections.ConfigDict()
  evaluate.begin_ckpt = 1
  evaluate.end_ckpt = 8
  evaluate.batch_size = 1000
  evaluate.enable_sampling = False
  evaluate.num_samples = 10000
  evaluate.enable_loss = False
  evaluate.enable_bpd = False
  evaluate.bpd_dataset = 'test'
  evaluate.save_images = False
  evaluate.enable_interpolate = False
  evaluate.enable_rescale = False
  ## new args
  evaluate.calc_is = False # ! WARNING: don't calculate Inception Score
  evaluate.eval_increment = 1
  evaluate.save_samples = False
  evaluate.mask = False
  evaluate.get_model_outs = False
  evaluate.classify_data_distribution = False

  # data
  config.data = data = ml_collections.ConfigDict()
  data.dataset = 'CELEBA'
  data.channels = 3
  data.image_size_ori = 64
  data.image_size = 64
  data.random_flip = True
  data.centered = False
  data.uniform_dequantization = False

  # model
  config.model = model = ml_collections.ConfigDict()
  model.sigma_min = 0.1
  model.sigma_max = 50
  model.num_scales = 1000
  model.beta_min = 0.1
  model.beta_max = 20.
  model.dropout = 0.1
  model.embedding_type = 'fourier'

  # optimization
  config.optim = optim = ml_collections.ConfigDict()
  optim.optimizer = 'Adam'
  optim.lr = 1e-4
  optim.weight_decay = 0.0
  optim.alpha = 0.0
  optim.gamma = 1.0
  optim.beta1 = 0.9
  optim.eps = 1e-8
  optim.warmup = 1000
  optim.grad_clip = 1
  optim.grad_clip_mode = 'norm'
  optim.anneal_rate = 1
  optim.anneal_iters = [0,0]

  config.seed = 49
  config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

  return config