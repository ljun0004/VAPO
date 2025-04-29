# Variational Potential (VAPO) Flow Bayes

Energy-based models (EBMs) are a powerful class of probabilistic generative models due to their flexibility and interpretability. However, relationships between potential flows and explicit EBMs remain underexplored, while contrastive divergence training via implicit Markov chain Monte Carlo (MCMC) sampling is often unstable and expensive in high-dimensional settings. In this paper, we propose Variational Potential (VAPO) Flow Bayes, a new energy-based generative framework that eliminates the need for implicit MCMC sampling and does not rely on auxiliary networks or cooperative training. VAPO learns an energy-parameterized potential flow by constructing a flow-driven density homotopy that is matched to the data distribution through a variational loss minimizing the Kullback-Leibler divergence between the flow-driven and marginal homotopies. This principled formulation enables robust and efficient generative modeling while preserving the interpretability of EBMs. Experimental results on image generation, interpolation, out-of-distribution detection, and compositional generation confirm the effectiveness of VAPO, showing that our method performs competitively with existing approaches in terms of sample quality and versatility across diverse generative modeling tasks.

---

*Acknowledgement:* Our implementation relies on the repo https://github.com/Newbeeer/Poisson_flow. 

## Dependencies

The necessary python (Python 3.9.12, CUDA Version 11.6) dependency for our code can be installed as follows:

```sh
pip install -r requirements.txt
```

## Usage

Train and evaluate our models through `main.py`.

```sh
python3 main.py:
  --config: Training configuration.
  --eval_folder: The folder name for storing evaluation results (default: 'eval')
  --mode: <train|eval>: Running mode: train or eval
  --workdir: Working directory
```

For example, to train a new PFGM w/ DDPM++ model on CIFAR-10 dataset, one could execute 

```sh
python3 main.py --config ./configs/homotopy/cifar10.py --mode train --workdir homotopy_cifar10
```

* `config` is the path to the config file. The prescribed config files are provided in `configs/`.

**Naming conventions of config files**: the path of a config file is a combination of the following dimensions:

- method: **VAPO**: `homotopy`
* dataset: One of `cifar10`, `celeba`

**Important Note** : We use the batch size (`training.batch_size=128` for CIFAR-10; `training.batch_size=64` for CelebA 64x64) for training. To adjust GPU memory cost, please modify the `training.batch_size` parameter in the config files. 


*  `workdir` is the path that stores all artifacts of one experiment, like checkpoints, samples, and evaluation results.

* `eval_folder` is the name of a subfolder in `workdir` that stores all artifacts of the evaluation process, like meta checkpoints for pre-emption prevention, image samples, and numpy dumps of quantitative results.

* `mode` is either "train" or "eval". When set to "train", it starts the training of a new model, or resumes the training of an old model if its meta-checkpoints (for resuming running after pre-emption in a cloud environment) exist in `workdir/checkpoints-meta` .

* Below are the list of evalutation command-line flags:

`--config.eval.enable_sampling`: Generate samples and evaluate sample quality, measured by FID and Inception score. 

`--config.eval.dataset=train/test` : Indicate whether to compute the likelihoods on the training or test dataset.

`--config.eval.enable_interpolate` : Image Interpolation
  

## Checkpoints

Please place the pretrained checkpoints under the directory `workdir/checkpoints`, e.g., `homotopy_cifar10/checkpoints`.

To generate and evaluate the FID/IS of the VAPO model on 10k samples, you could execute:

```shell
python3 main.py --config ./configs/homotopy/cifar10.py --mode eval --workdir homotopy_cifar10 --config.eval.enable_sampling --config.eval.num_samples 10000
```

To only generate and visualize 100 samples of the VAPO model, you could execute:

```shell
python3 main.py --config ./configs/homotopy/cifar10.py --mode eval --workdir homotopy_cifar10 --config.eval.enable_sampling --config.eval.save_images --config.eval.batch_size 100
```

The samples will be saved to `homotopy_cifar10/eval/`.

Pre-trainend model checkpoints (`homotopy_cifar10` for CIFAR-10 and `homotopy_celeba` for CelebA 64x64) are provided in this [Google drive folder](https://drive.google.com/drive/folders/1K86fbiS7iLvu-hu_rKzmK1oKp7CROfn0?usp=sharing).

| Dataset              | Checkpoint path                                              | Parameter Count | NFE (RK45 Sampling) | GPU Usage (Training) | Batch Size (Training) |
| -------------------- | :----------------------------------------------------------- | :---: | :---: | :---: | :---: |
| CIFAR-10             | [`homotopy_cifar10/checkpoints`] | 55.7M | 74 | < 80GB | 128 |
| CelebA 64x64        | [`homotopy_celeba/checkpoints`] | 55.7M | 74 | < 80GB | 64 |

### FID statistics

Please find the pre-computed statistics files for FID scores in the following links and place them under the directory `assets/stats`:

[CIFAR-10](https://drive.google.com/file/d/1YyympxZ95l6_ane0TxYt94yqeiGcOBNG/view?usp=sharing),  [CelebA 64x64](https://drive.google.com/file/d/1dzSsmBvJOjDy12VzdypWDVYBF8b9yRkm/view?usp=sharing)
