from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
from models import ddpm, ncsnv2, ncsnpp
import sampling
import sde_lib
import torchvision
import numpy as np
import time
import torch
import copy
import os

from models.utils import from_flattened_numpy, to_flattened_numpy, get_score_fn
from scipy import integrate
import sde_lib
from models import utils as mutils
from sampling import ReverseDiffusionPredictor
import datasets


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
  "config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("ckpt", None, "Checkpoint path to load.")
flags.mark_flags_as_required(["config", "ckpt"])


def main(argv):
    config = FLAGS.config
    world_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    score_model = mutils.create_model(config)
    ckpt = torch.load(FLAGS.ckpt)
    score_model.load_state_dict(ckpt['model'], strict=False)

    param = 0
    for p in score_model.parameters():
        v = 1
        for v_ in p.shape:
            v *= v_
        param += v
    print(f"Number of parameters: {param/1e6:.2f} M")

    dataset = datasets.get_dataset(config, world_size, mode="train", return_ds=True)

    num_data = min(len(dataset), 50000)
    idx_start = world_rank * num_data // world_size
    idx_end = idx_start + num_data // world_size

    if isinstance(dataset, torchvision.datasets.CIFAR10):
      dataset.data = dataset.data[idx_start : idx_end]
      dataset.targets = dataset.targets[idx_start : idx_end]
    elif isinstance(dataset, torchvision.datasets.ImageFolder):
      dataset.samples = dataset.samples[idx_start : idx_end]
      dataset.imgs = dataset.imgs[idx_start : idx_end]
      dataset.targets = dataset.targets[idx_start : idx_end]

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.training.batch_size, shuffle=False)

    sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    eps = 1e-5

    losses = []
    times = [[], []]

    for i, (x1, _) in enumerate(dataloader):
        x1 = x1.to("cuda")

        score_fn = mutils.get_score_fn(sde, score_model, train=True, continuous=True)
        t = torch.rand(x1.shape[0], device=x1.device) * (sde.T - eps) + eps
        z = torch.randn_like(x1)
        mean, std = sde.marginal_prob(x1, t)
        perturbed_data = mean + std[:, None, None, None] * z
        perturbed_data.requires_grad_(True)
        score = score_fn(perturbed_data, t)

        t0 = time.time()

        v = torch.sign(torch.randn_like(score))
        score_vjp = torch.autograd.grad(
          outputs=score, inputs=perturbed_data, grad_outputs=v, 
          create_graph=True,
        )[0]
        dim = perturbed_data.shape[1] * perturbed_data.shape[2] * perturbed_data.shape[3]
        score_jvp_trace = (score_vjp * v).mean((1,2,3)) * std**2

        times[0].append(time.time() - t0)

        for h in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
            t0 = time.time()
            score_jvp = (score_fn(perturbed_data + h * v, t) - score.detach()) / h
            score_jvp_trace2 = (score_jvp * v).mean((1,2,3)) * std**2
            times[1].append(time.time() - t0)

            losses.append([
              torch.mean((score_jvp_trace - score_jvp_trace2) ** 2).item(),
              torch.std((score_jvp_trace - score_jvp_trace2) ** 2).item(),
            ])

        break

    print(losses)
    print(times)


if __name__ == "__main__":
    if "OMPI_COMM_WORLD_RANK" in os.environ:
        os.environ["RANK"] = os.environ["OMPI_COMM_WORLD_RANK"]
        os.environ["WORLD_SIZE"] = os.environ["OMPI_COMM_WORLD_SIZE"]
        os.environ["LOCAL_RANK"] = os.environ["OMPI_COMM_WORLD_LOCAL_RANK"]

    app.run(main)
