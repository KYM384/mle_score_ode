# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
"""Return training and evaluation/test datasets from config files."""
import torch
import torchvision
from PIL import Image
import os


def get_data_scaler(config):
  """Data normalizer. Assume data are always in [0, 1]."""
  if config.data.centered:
    # Rescale to [-1, 1]
    return lambda x: x * 2. - 1.
  else:
    return lambda x: x


def get_data_inverse_scaler(config):
  """Inverse data normalizer."""
  if config.data.centered:
    # Rescale [-1, 1] to [0, 1]
    return lambda x: (x + 1.) / 2.
  else:
    return lambda x: x


def crop_resize(image, resolution):
  """Crop and resize an image to the given resolution."""
  crop = tf.minimum(tf.shape(image)[0], tf.shape(image)[1])
  h, w = tf.shape(image)[0], tf.shape(image)[1]
  image = image[(h - crop) // 2:(h + crop) // 2,
          (w - crop) // 2:(w + crop) // 2]
  image = tf.image.resize(
    image,
    size=(resolution, resolution),
    antialias=True,
    method=tf.image.ResizeMethod.BICUBIC)
  return tf.cast(image, tf.uint8)


def resize_small(image, resolution):
  """Shrink an image to the given resolution."""
  h, w = image.shape[0], image.shape[1]
  ratio = resolution / min(h, w)
  h = tf.round(h * ratio, tf.int32)
  w = tf.round(w * ratio, tf.int32)
  return tf.image.resize(image, [h, w], antialias=True)


def central_crop(image, size):
  """Crop the center of an image to the given size."""
  top = (image.shape[0] - size) // 2
  left = (image.shape[1] - size) // 2
  return tf.image.crop_to_bounding_box(image, top, left, size, size)


def sample_iter(loader):
  while True:
    for batch in loader:
      yield batch


def get_dataset(config, world_size=1, mode="train", return_ds=False):
  """Create data loaders for training and evaluation.

  Args:
    config: A ml_collection.ConfigDict parsed from config files.

  Returns:
    train_loader
  """
  # Compute batch size for this worker.
  batch_size = config.training.batch_size

  if config.data.dataset == 'CIFAR10':
    train_ds = torchvision.datasets.CIFAR10(
      root="./data", train=(mode=="train"), download=True,
      transform=torchvision.transforms.ToTensor(),
    )

  elif config.data.dataset == "ImageNet32":
    train_ds = torchvision.datasets.ImageFolder(
      root="/data/",
      transform=torchvision.transforms.ToTensor(),
    )

  elif config.data.dataset == "CELEBA":
    train_ds = torchvision.datasets.ImageFolder(
      root="/data/",
      transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((64, 64)),
      ]),
    )

  elif config.data.dataset == "FFHQ":
    train_ds = torchvision.datasets.ImageFolder(
      root="/data/",
      transform=torchvision.transforms.ToTensor(),
    )

  if return_ds:
    return train_ds

  train_loader = torch.utils.data.DataLoader(
    train_ds, batch_size=config.training.batch_size//world_size, shuffle=True,
    num_workers=8, pin_memory=True, drop_last=True,
  )

  return sample_iter(train_loader)
