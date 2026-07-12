# coding=utf-8
"""Seed the TFDS download cache with manually downloaded ImageNet32/64 tars.

The original URLs (http://image-net.org/small/*.tar) are dead, so
`tfds.builder('downsampled_imagenet/32x32').download_and_prepare()` cannot
download them anymore. This script places tars obtained elsewhere (e.g.
Academic Torrents) into the TFDS download cache under the exact name TFDS
expects, so `download_and_prepare()` reuses them instead of downloading.

Must run inside the project container (needs tensorflow_datasets):

    python3.8 seed_tfds_imagenet32.py /path/to/train_32x32.tar /path/to/valid_32x32.tar
"""

import argparse
import hashlib
import json
import os
import shutil
import sys

URLS = {
    'train_32x32.tar': 'http://image-net.org/small/train_32x32.tar',
    'valid_32x32.tar': 'http://image-net.org/small/valid_32x32.tar',
    'train_64x64.tar': 'http://image-net.org/small/train_64x64.tar',
    'valid_64x64.tar': 'http://image-net.org/small/valid_64x64.tar',
}


def sha256_of(path):
  h = hashlib.sha256()
  with open(path, 'rb') as f:
    for chunk in iter(lambda: f.read(1 << 22), b''):
      h.update(chunk)
  return h.hexdigest()


def registered_url_infos():
  """Return {url: (size, sha256)} registered in the installed TFDS, best-effort."""
  try:
    from tensorflow_datasets.core.download import checksums as checksums_lib
    infos = checksums_lib.get_all_url_infos()
    return {u: (i.size, i.checksum) for u, i in infos.items()}
  except Exception:
    return {}


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('tars', nargs='+',
                      help='Paths to downloaded tars. Basenames must be one of: '
                           + ', '.join(URLS))
  parser.add_argument('--data_dir', default=None,
                      help='TFDS data dir (default: $TFDS_DATA_DIR or ~/tensorflow_datasets)')
  parser.add_argument('--move', action='store_true',
                      help='Move files instead of hardlink/copy')
  args = parser.parse_args()

  try:
    from tensorflow_datasets.core.download import resource as resource_lib
  except ImportError:
    sys.exit('ERROR: tensorflow_datasets is not importable. '
             'Run this script inside the project container (singularity/docker).')

  data_dir = args.data_dir or os.environ.get('TFDS_DATA_DIR') \
      or os.path.expanduser('~/tensorflow_datasets')
  downloads_dir = os.path.join(data_dir, 'downloads')
  os.makedirs(downloads_dir, exist_ok=True)
  registered = registered_url_infos()

  for src in args.tars:
    base = os.path.basename(src)
    if base not in URLS:
      sys.exit(f'ERROR: unexpected file name {base!r}. Expected one of: {sorted(URLS)}')
    if not os.path.isfile(src):
      sys.exit(f'ERROR: {src} does not exist.')
    url = URLS[base]

    print(f'[{base}] computing sha256 ...')
    sha = sha256_of(src)
    size = os.path.getsize(src)

    if url in registered:
      reg_size, reg_sha = registered[url]
      if (reg_sha and reg_sha != sha) or (reg_size and reg_size != size):
        sys.exit(
            f'ERROR: {base} does not match the checksum registered in TFDS.\n'
            f'  registered: size={reg_size} sha256={reg_sha}\n'
            f'  your file : size={size} sha256={sha}\n'
            'This is not the genuine van den Oord et al. file; '
            'download_and_prepare() would reject it. Get the original tar.')
      print(f'[{base}] checksum matches the TFDS registry.')
    else:
      print(f'[{base}] WARNING: could not read the TFDS checksum registry; '
            'placing the file unverified.')

    fname = resource_lib.get_dl_fname(url, sha)
    dst = os.path.join(downloads_dir, fname)
    if os.path.exists(dst) and os.path.getsize(dst) == size:
      print(f'[{base}] already in cache: {dst}')
    elif args.move:
      shutil.move(src, dst)
    else:
      try:
        os.link(src, dst)  # instant if same filesystem
      except OSError:
        print(f'[{base}] copying {size / 2**30:.2f} GiB ...')
        shutil.copy2(src, dst)

    info = {'dataset_names': ['downsampled_imagenet'],
            'urls': [url],
            'original_fname': base}
    with open(dst + '.INFO', 'w') as f:
      json.dump(info, f)
    print(f'[{base}] -> {dst}')

  print('\nDone. Now build the dataset (also inside the container):\n'
        "  python3.8 -c \"import tensorflow_datasets as tfds; "
        "b = tfds.builder('downsampled_imagenet/32x32'); "
        "b.download_and_prepare(); print(b.info.splits)\"")


if __name__ == '__main__':
  main()
