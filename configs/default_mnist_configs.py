import ml_collections
import torch


def get_default_configs():
  config = ml_collections.ConfigDict()
  # training
  config.training = training = ml_collections.ConfigDict()
  config.training.batch_size = 128
  training.n_iters = 800001
  training.snapshot_freq = 5000
  training.log_freq = 50
  training.eval_freq = 5000
  ## store additional checkpoints for preemption in cloud computing environments
  training.snapshot_freq_for_preemption = 1000
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

  # evaluation
  config.eval = evaluate = ml_collections.ConfigDict()
  evaluate.begin_ckpt = 40
  evaluate.end_ckpt = 41
  evaluate.batch_size = 1000
  evaluate.enable_sampling = False
  evaluate.num_samples = 10000
  evaluate.enable_loss = False
  evaluate.enable_bpd = False
  evaluate.bpd_dataset = 'test'
  evaluate.save_images = False
  evaluate.enable_interpolate = False
  evaluate.enable_rescale = False

  # data
  config.data = data = ml_collections.ConfigDict()
  data.dataset = 'MNIST'
  data.image_size = 28
  data.random_flip = False
  data.centered = False
  data.uniform_dequantization = False
  data.num_channels = 1

  # model
  config.model = model = ml_collections.ConfigDict()
  model.sigma_min = 0.01
  model.sigma_max = 50
  model.num_scales = 1000
  model.beta_min = 0.1
  model.beta_max = 20.
  model.dropout = 0.1
  model.embedding_type = 'fourier'

  # optimization
  config.optim = optim = ml_collections.ConfigDict()
  optim.weight_decay = 0
  optim.optimizer = 'Adam'
  optim.lr = 1e-3
  optim.alpha = 1e-4
  # optim.sample_lr = 1e-3
  optim.beta1 = 0.9
  optim.eps = 1e-8
  optim.warmup = 0
  optim.grad_clip = 1.

  config.seed = 49
  config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

  return config