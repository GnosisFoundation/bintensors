import os
import hashlib
from _hashlib import HASH
from typing import Dict, Optional, Union, Callable, Tuple

import numpy as np
import tensorflow as tf

from bintensors import numpy, safe_open


def save(tensors: Dict[str, tf.Tensor], metadata: Optional[Dict[str, str]] = None) -> bytes:
    """
    Saves a dictionary of tensors into raw bytes in bintensors format.

    Args:
        tensors (`Dict[str, tf.Tensor]`):
            The incoming tensors. Tensors need to be contiguous and dense.
        metadata (`Dict[str, str]`, *optional*, defaults to `None`):
            Optional text only metadata you might want to save in your header.
            For instance it can be useful to specify more about the underlying
            tensors. This is purely informative and does not affect tensor loading.

    Returns:
        `bytes`: The raw bytes representing the format

    Example:

    ```python
    from bintensors.tensorflow import save
    import tensorflow as tf

    tensors = {"embedding": tf.zeros((512, 1024)), "attention": tf.zeros((256, 256))}
    byte_data = save(tensors)
    ```
    """
    np_tensors = _tf2np(tensors)
    return numpy.save(np_tensors, metadata=metadata)


def save_file(
    tensors: Dict[str, tf.Tensor],
    filename: Union[str, os.PathLike],
    metadata: Optional[Dict[str, str]] = None,
) -> None:
    """
    Saves a dictionary of tensors into raw bytes in bintensors format.

    Args:
        tensors (`Dict[str, tf.Tensor]`):
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
    from bintensors.tensorflow import save_file
    import tensorflow as tf

    tensors = {"embedding": tf.zeros((512, 1024)), "attention": tf.zeros((256, 256))}
    save_file(tensors, "model.bintensors")
    ```
    """
    np_tensors = _tf2np(tensors)
    return numpy.save_file(np_tensors, filename, metadata=metadata)


def save_with_checksum(
    tensor_dict: Dict[str, tf.Tensor],
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
    from bintensors.tensorflow import save_with_checksum
    import tensorflow as tf

    tensors = {"embedding": tf.zeros((512, 1024)), "attention": tf.zeros((256, 256))}
    checksum, byte_data = save_with_checksum(tensors)
    ```
    """
    np_tensors = _tf2np(tensor_dict)
    return numpy.save_with_checksum(np_tensors, metadata, hasher)


def load(data: bytes) -> Dict[str, tf.Tensor]:
    """
    Loads a bintensors file into tensorflow format from pure bytes.

    Args:
        data (`bytes`):
            The content of a bintensors file

    Returns:
        `Dict[str, tf.Tensor]`: dictionary that contains name as key, value as `tf.Tensor` on cpu

    Example:

    ```python
    from bintensors.tensorflow import load

    file_path = "./my_folder/bert.bintensors"
    with open(file_path, "rb") as f:
        data = f.read()

    loaded = load(data)
    ```
    """
    flat = numpy.load(data)
    return _np2tf(flat)


def load_file(filename: Union[str, os.PathLike]) -> Dict[str, tf.Tensor]:
    """
    Loads a bintensors file into tensorflow format.

    Args:
        filename (`str`, or `os.PathLike`)):
            The name of the file which contains the tensors

    Returns:
        `Dict[str, tf.Tensor]`: dictionary that contains name as key, value as `tf.Tensor`

    Example:

    ```python
    from bintensors.tensorflow import load_file

    file_path = "./my_folder/bert.bintensors"
    loaded = load_file(file_path)
    ```
    """
    result = {}
    with safe_open(filename, framework="tf") as f:
        for k in f.offset_keys():
            result[k] = f.get_tensor(k)
    return result


def _np2tf(numpy_dict: Dict[str, np.ndarray]) -> Dict[str, tf.Tensor]:
    for k, v in numpy_dict.items():
        numpy_dict[k] = tf.convert_to_tensor(v)
    return numpy_dict


def _tf2np(tf_dict: Dict[str, tf.Tensor]) -> Dict[str, np.array]:
    for k, v in tf_dict.items():
        tf_dict[k] = v.numpy()
    return tf_dict
