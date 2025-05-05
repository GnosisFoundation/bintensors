import os
import hashlib
from _hashlib import HASH
from typing import Dict, Optional, Union, Tuple, Callable

import numpy as np

import mlx.core as mx
from bintensors import numpy, safe_open

__all__ = ["save", "save_file", "load", "load_file", "save_with_checksum"]

def save(tensors: Dict[str, mx.array], metadata: Optional[Dict[str, str]] = None) -> bytes:
    """
    Saves a dictionary of tensors into raw bytes in bintensors format.

    Args:
        tensors (`Dict[str, mx.array]`):
            The incoming tensors. Tensors need to be contiguous and dense.
        metadata (`Dict[str, str]`, *optional*, defaults to `None`):
            Optional text only metadata you might want to save in your header.
            For instance it can be useful to specify more about the underlying
            tensors. This is purely informative and does not affect tensor loading.

    Returns:
        `bytes`: The raw bytes representing the format

    Example:

    ```python
    from bintensors.mlx import save
    import mlx.core as mx

    tensors = {"embedding": mx.zeros((512, 1024)), "attention": mx.zeros((256, 256))}
    byte_data = save(tensors)
    ```
    """
    np_tensors = _mx2np(tensors)
    return numpy.save(np_tensors, metadata=metadata)


def save_file(
    tensors: Dict[str, mx.array],
    filename: Union[str, os.PathLike],
    metadata: Optional[Dict[str, str]] = None,
) -> None:
    """
    Saves a dictionary of tensors into raw bytes in bintensors format.

    Args:
        tensors (`Dict[str, mx.array]`):
            The incoming tensors. Tensors need to be contiguous and dense.
        filename (`str`, or `os.PathLike`)):
            The filename we're saving into.
        metadata (`Dict[str, str]`, *optional*, defaults to `None`):
            Optional text only metadata you might want to save in your header.
            For instance it can be useful to specify more about the underlying
            tensors. This is purely informative and does not affect tensor loading.

    Returns:
        `None`

    Example:

    ```python
    from bintensors.mlx import save_file
    import mlx.core as mx

    tensors = {"embedding": mx.zeros((512, 1024)), "attention": mx.zeros((256, 256))}
    save_file(tensors, "model.bt")
    ```
    """
    np_tensors = _mx2np(tensors)
    return numpy.save_file(np_tensors, filename, metadata=metadata)


def save_with_checksum(
    tensor_dict: Dict[str, np.ndarray],
    metadata: Optional[Dict[str, str]] = None,
    hasher: Callable[[bytes], HASH] = hashlib.sha1,
) -> Tuple[bytes, bytes]:
    """
    Saves a dictionary of tensors into raw bytes in bintensors format.

    Args:
        tensors (`Dict[str, np.ndarray]`):
            The incoming tensors. Tensors need to be contiguous and dense.
        metadata (`Dict[str, str]`, *optional*, defaults to `None`):
            Optional text only metadata you might want to save in your header.
            For instance it can be useful to specify more about the underlying
            tensors. This is purely informative and does not affect tensor loading.
        hasher (`Callable[[bytes], HASH]`):
            A hash is an object used to calculate a checksum of a string of information.


    Returns:
        `bytes`: The raw bytes representing the format

    Example:

    ```python
    from bintensors.mlx import save_with_checksum
    import mlx.core as mx

    tensors = {"embedding": mx.zeros((512, 1024)), "attention": mx.zeros((256, 256))}
    checksum, byte_data = save_with_checksum(tensors)
    ```
    """
    np_tensors = _mx2np(tensor_dict)
    return numpy.save_with_checksum(np_tensors, metadata, hasher)


def load(data: bytes) -> Dict[str, mx.array]:
    """
    Loads a bintensors file into MLX format from pure bytes.

    Args:
        data (`bytes`):
            The content of a bintensors file

    Returns:
        `Dict[str, mx.array]`: dictionary that contains name as key, value as `mx.array`

    Example:

    ```python
    from bintensors.mlx import load

    file_path = "./my_folder/bert.bt"
    with open(file_path, "rb") as f:
        data = f.read()

    loaded = load(data)
    ```
    """
    flat = numpy.load(data)
    return _np2mx(flat)


def load_file(filename: Union[str, os.PathLike]) -> Dict[str, mx.array]:
    """
    Loads a bintensors file into MLX format.

    Args:
        filename (`str`, or `os.PathLike`)):
            The name of the file which contains the tensors

    Returns:
        `Dict[str, mx.array]`: dictionary that contains name as key, value as `mx.array`

    Example:

    ```python
    from bintensors.flax import load_file

    file_path = "./my_folder/bert.bt"
    loaded = load_file(file_path)
    ```
    """
    result = {}
    with safe_open(filename, framework="mlx") as f:
        for k in f.offset_keys():
            result[k] = f.get_tensor(k)
    return result


def _np2mx(numpy_dict: Dict[str, np.ndarray]) -> Dict[str, mx.array]:
    """
    Preform conversion from numpy storage backend to mlx storage backend.

    Args:
        tensors (`Dict[str,  np.ndarray]`):
            The incoming tensors. Tensors need to be contiguous and dense.

    Returns:
          `Dict[str, mx.array]`: dictionary that contains name as key, value as `mx.array`
    """
    for k, v in numpy_dict.items():
        numpy_dict[k] = mx.array(v)
    return numpy_dict


def _mx2np(mx_dict: Dict[str, mx.array]) -> Dict[str, np.ndarray]:
    """
    Preform conversion from mlx storage backend to numpy storage backend.

    Args:
        tensors (`Dict[str, mx.array]`):
            The incoming tensors. Tensors need to be contiguous and dense.
    Returns:
          `Dict[str, np.ndarray]`: dictionary that contains name as key, value as `np.ndarray`
    """
    new_dict = {}
    for k, v in mx_dict.items():
        new_dict[k] = np.asarray(v)
    return new_dict
