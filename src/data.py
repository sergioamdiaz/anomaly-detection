#*******************************************************************************
# IMPORTS:
#*******************************************************************************

from pathlib import Path
import random
import os

import numpy as np
import tensorflow as tf

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

#-------------------------------------------------------------------------------
# Constants
#-------------------------------------------------------------------------------

SEED = 42
IMG_SIZE = (512, 512)
BATCH_SIZE = 8
AUTOTUNE = tf.data.AUTOTUNE # -1

VALID_EXTS = {".jpg", ".jpeg"}

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# NOTE: These constants can be added in the future to the config file. For now they are hardcoded here for simplicity.

#*******************************************************************************
# DATA LOADING AND PREPROCESSING:
#*******************************************************************************

#-------------------------------------------------------------------------------
# List Paths:

def list_files(directory: Path, skip_dirs: list[str] = []) -> list[Path]:
    """ Returns the list of paths of all the images in the given directory and subdirectories.
    If skip_dirs is not given, no folders will be skipped.
    """
    files = []
    n = 0
    logging.debug(f"Scanned directories:")
    for root, dirs, filenames in os.walk(directory):
        # Modify 'dirs' in place so certain folders are exluded from the search:
        logging.debug(f'{root}')
        dirs[ : ] = [d for d in dirs if d not in skip_dirs]
        for fl in filenames:
            if Path(fl).suffix.lower() not in VALID_EXTS: # skip non-image files
                n+=1
                continue
            files.append(Path(root) / fl)
    logging.info(f'\n{n} invalid-format image files were skipped') if n > 0 else print('\nAll images have valid formats')
    files.sort()
    return files

#-------------------------------------------------------------------------------
# Load Data:

def _load_image_tf(tf_path: tf.Tensor) -> tf.Tensor:
    ''' Loads an image from the given path and preprocesses it for training. Only supports JPG images.
    Returns a tensor of shape (H, W, 3) and dtype float32 with pixel values in [0,1]. '''
    image_bytes = tf.io.read_file(tf_path) # read the image file as a bytes string
    image = tf.image.decode_jpeg(image_bytes, channels=3) # decode the bytes string into a tensor of shape (H, W, 3) and dtype uint8
    image = tf.image.convert_image_dtype(image, tf.float32)  # Rescale to [0,1] (better for training)
    image = tf.image.resize(image, IMG_SIZE) # Since the image is already square, the resizing will not distort the image
    return image

#-------------------------------------------------------------------------------
# Add Noise:

def _add_mild_corruption(image: tf.Tensor, corruption_factor: float) -> tf.Tensor:
    ''' Adds mild corruption to the given image tensor. Returns a corrupted image tensor of the same shape and dtype as the input, with pixel values in [0,1]. '''
    x = image
    # Mild Gaussian noise
    noise = tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=corruption_factor, dtype=tf.float32)
    x = x + noise # Add noise. Can push pixels outside [0,1].
    x = tf.clip_by_value(x, 0.0, 1.0) # Ensure pixel values are still in [0,1]. If p > 1 → 1, if p < 0 → 0.

    # Mild brightness variation
    x = tf.image.random_brightness(x, max_delta=0.05)
    x = tf.clip_by_value(x, 0.0, 1.0)

    # Mild contrast variation
    x = tf.image.random_contrast(x, lower=0.95, upper=1.05)
    x = tf.clip_by_value(x, 0.0, 1.0)

    return x

def corrupt_and_clean_from_path(tf_path: tf.Tensor, corruption_factor: float,
                                ) -> tuple[tf.Tensor, tf.Tensor]:
    ''' Loads the image at the given path, creates a mildly corrupted version of it, and returns both the corrupted and clean images as tensors. '''
    clean = _load_image_tf(tf_path)
    corrupted = _add_mild_corruption(clean, corruption_factor)
    return corrupted, clean

#-------------------------------------------------------------------------------
# Create Dataset:

def create_dataset(paths: list[Path], 
                   shuffle: bool = True, 
                   corruption_factor: float = 0.5) -> tf.data.Dataset:
  # Create a dataset with the list of paths (elements will be tf.Tensors):
  lazy_ds = tf.data.Dataset.from_tensor_slices([str(p) for p in paths])
  # Shuffles (only the paths, not the images yet) to ensure random order at each epoch during training:
  lazy_ds = (lazy_ds.shuffle(buffer_size=len(paths), seed=SEED)) if shuffle else lazy_ds
  # Apply corruption and loading in parallel to speed up the data pipeline:
  lazy_ds = lazy_ds.map(lambda x: corrupt_and_clean_from_path(x, corruption_factor), 
                        num_parallel_calls=AUTOTUNE) # x is a tf.Tensor
  # Each element of the dataset is now a tuple (corrupted_image, clean_image)
  
  # Batch the dataset:
  lazy_ds = lazy_ds.batch(BATCH_SIZE)
  # Prefetch to improve performance:
  lazy_ds = lazy_ds.prefetch(AUTOTUNE)

  return lazy_ds