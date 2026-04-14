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

VALID_EXTS = {".jpg", ".jpeg", ".png"}

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# NOTE: These constants can be added in the future to the config file. For now they are hardcoded here for simplicity.

#*******************************************************************************
# DATA LOADING AND PREPROCESSING:
#*******************************************************************************

#-------------------------------------------------------------------------------
# List Paths:

def list_files(directory: Path, skip_dirs: set[str] | None = None) -> list[Path]:
    """ Returns the list of paths of all the images in the given directory and subdirectories.
    If skip_dirs is not given, no folders will be skipped.
    """
    skip_dirs = skip_dirs or set() # None, set(), [], {}, "" → set()
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
    print(('-'*80 + f'\n    {n} invalid-format files were skipped\n' + '-'*80)
          if n > 0 else print('-'*80 + '\n    All images have valid formats\n' + '-'*80))
    
    files.sort()
    return files

#-------------------------------------------------------------------------------
# Split Data:

def split_train_val(img_paths : list[Path], mask_paths : list[Path], val_split: float
                    ) -> tuple[list[Path], list[Path], list[Path], list[Path]]:
    ''' 
    Splits the list of image and mask paths into train and validation sets.
    '''
    # Create and shuffle indices:
    indices = list(range(len(img_paths)))
    random.shuffle(indices)

    # Sort according to indices:
    img_paths = [img_paths[i] for i in indices]
    mask_paths = [mask_paths[i] for i in indices]

    n_val = int(len(img_paths) * val_split)

    # Split into train and validation:
    train_img_paths = img_paths[n_val:]
    train_mask_paths = mask_paths[n_val:]

    val_img_paths = img_paths[:n_val]
    val_mask_paths = mask_paths[:n_val]

    # Sanity check:
    same_order = (img.stem == mask.stem for img, mask in zip(img_paths, mask_paths))
    assert len(img_paths) == len(mask_paths), "Images and masks have different lengths"
    assert same_order, "Images and masks are not in the same order"
    
    return train_img_paths, train_mask_paths, val_img_paths, val_mask_paths

#-------------------------------------------------------------------------------
# Load Image and Mask:

def _load_image_tf(tf_image_path: tf.Tensor) -> tf.Tensor:
    ''' 
    Loads an image from the given path and preprocesses it for training. 
    Only supports JPG images.
    Returns a tensor of shape (H, W, 3) and dtype float32 with pixel values in [0,1]. 
    '''
    image_bytes = tf.io.read_file(tf_image_path) # read the image file as a bytes string
    image = tf.image.decode_jpeg(image_bytes, channels=3) # decode the bytes string into a tensor of shape (H, W, 3) and dtype uint8
    image = tf.image.convert_image_dtype(image, tf.float32)  # Rescale to [0,1] (better for training)
    tf.debugging.assert_equal(tf.shape(image)[:2], IMG_SIZE) # Debugging: Double check the image size
    return image

def _load_mask_tf(tf_mask_path: tf.Tensor) -> tf.Tensor:
    ''' 
    Loads a mask from the given path and preprocesses it for training (loss function). 
    Only supports PNG images. 
    Returns a tensor of shape (H, W, 1) and dtype float32 with pixel values in [0,1]. 
    '''
    mask_bytes = tf.io.read_file(tf_mask_path) # read the image file as a bytes string
    mask = tf.image.decode_png(mask_bytes, channels=1) # decode the bytes string into a tensor of shape (H, W, 1) and dtype uint8
    mask = tf.image.convert_image_dtype(mask, tf.float32)  # Rescale to [0,1] (consistency with the rest of the pipeline) 
    tf.debugging.assert_equal(tf.shape(mask)[:2], IMG_SIZE) # Debugging: Double check the image size
    return mask

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

def corrupt_clean_mask_from_path(tf_image_path: tf.Tensor, 
                                 tf_mask_path: tf.Tensor,
                                 corruption_factor: float,
                                 ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    ''' 
    Loads the image at the given path, creates a mildly corrupted version of it.
    Loads the mask at the given path.
    Returns the image, currupted image, and mask as tensors.
    '''
    clean = _load_image_tf(tf_image_path)
    corrupted = _add_mild_corruption(clean, corruption_factor)
    mask = _load_mask_tf(tf_mask_path)
    return corrupted, clean, mask

#-------------------------------------------------------------------------------
# Create Dataset:

def create_dataset(image_paths: list[Path],
                   mask_paths: list[Path],
                   shuffle: bool = True,
                   corruption_factor: float = 0.8) -> tf.data.Dataset:
    '''
    Creates a dataset containing triples of the form (corrupted_image, clean_image, mask). 
    '''
    if len(image_paths) != len(mask_paths):
        raise ValueError(f"image_paths and mask_paths must have the same length, "
                         f"got {len(image_paths)} and {len(mask_paths)}.")
        
    # Create a dataset with the list of tuples of paths (elements will be tf.Tensors tuples):
    lazy_ds = tf.data.Dataset.from_tensor_slices(
        ([str(p) for p in image_paths], [str(p) for p in mask_paths]) # tuples
        )
    # Shuffles (the paths tuples) to ensure random order at each epoch during training:
    lazy_ds = (lazy_ds.shuffle(buffer_size=len(image_paths), seed=SEED)) if shuffle else lazy_ds
    # Apply corruption and loading in parallel to speed up the data pipeline:
    lazy_ds = lazy_ds.map(lambda tf_image_path, tf_mask_path: 
        corrupt_clean_mask_from_path(tf_image_path, tf_mask_path, corruption_factor),
        num_parallel_calls=AUTOTUNE) 
    # Each element of the dataset is now a triple (corrupted_image, clean_image, mask)

    # Batch the dataset:
    lazy_ds = lazy_ds.batch(BATCH_SIZE)
    # Prefetch to improve performance:
    lazy_ds = lazy_ds.prefetch(AUTOTUNE)

    return lazy_ds