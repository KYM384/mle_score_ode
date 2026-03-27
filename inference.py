from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
from models import ddpm, ncsnv2, ncsnpp
import sampling
from models import utils as mutils
import sde_lib
import torchvision
import torch
import copy
import os


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
  "config", None, "Training configuration.", lock_config=True)
flags.mark_flags_as_required(["config"])


def main(argv):
    config = FLAGS.config

    score_model = mutils.create_model(config)
    ckpt = torch.load("results/ve_celeba_ncsnpp_continous/checkpoints/checkpoint_26.pth")
    score_model.load_state_dict(ckpt['model'], strict=False)
    print(ckpt["step"])

    sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    sampling_eps = 1e-5

    config.training.batch_size = 10
    world_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    start = world_rank * 50000 // world_size
    end = start + 50000 // world_size

    sampling_shape = (config.training.batch_size, config.data.num_channels,
                        config.data.image_size, config.data.image_size)
    sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, lambda x:x, sampling_eps)

    with torch.no_grad():
        for i in range(start, end, config.training.batch_size):
            print(i)
            sample, n = sampling_fn(score_model)

            for j in range(sample.shape[0]):
                torchvision.utils.save_image(sample[j], f'out/{i+j:05}.png', normalize=True, value_range=(0, 1))

if __name__ == "__main__":
    if "OMPI_COMM_WORLD_RANK" in os.environ:
        os.environ["RANK"] = os.environ["OMPI_COMM_WORLD_RANK"]
        os.environ["WORLD_SIZE"] = os.environ["OMPI_COMM_WORLD_SIZE"]
        os.environ["LOCAL_RANK"] = os.environ["OMPI_COMM_WORLD_LOCAL_RANK"]

    app.run(main)