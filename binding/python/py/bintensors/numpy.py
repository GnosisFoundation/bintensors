import os
import sys
from typing import Dict, Optional, Union, Tuple


try:
    import numpy as np
except ImportError:
    raise ImportError(
        "Could not find the 'numpy' module. To use this part of the package, please install numpy: `pip install numpy`."
    )


from bintensors import deserialize, safe_open, serialize, serialize_file, serialize_checksum

__all__ = ["save", "save_file", "load", "load_file"]


def _tobytes(tensor: np.ndarray) -> bytes:
    if not _is_little_endian(tensor):
        tensor = tensor.byteswap(inplace=False)
    return tensor.tobytes()


def save(tensor_dict: Dict[str, np.ndarray], metadata: Optional[Dict[str, str]] = None) -> bytes:
    """
    Saves a dictionary of tensors into raw bytes in bintensors format.

    Args:
        tensor_dict (`Dict[str, np.ndarray]`):
            The incoming tensors. Tensors need to be contiguous and dense.
        metadata (`Dict[str, str]`, *optional*, defaults to `None`):
            Optional text only metadata you might want to save in your header.
            For instance it can be useful to specify more about the underlying
            tensors. This is purely informative and does not affect tensor loading.

    Returns:
        `bytes`: The raw bytes representing the format

    Example:

    ```python
    from bintensors.numpy import save
    import numpy as np

    tensors = {"embedding": np.zeros((512, 1024)), "attention": np.zeros((256, 256))}
    byte_data = save(tensors)
    ```
    """
    flattened = {k: {"dtype": v.dtype.name, "shape": v.shape, "data": _tobytes(v)} for k, v in tensor_dict.items()}
    serialized = serialize(flattened, metadata=metadata)
    result = bytes(serialized)
    return result


def save_file(
    tensor_dict: Dict[str, np.ndarray], filename: Union[str, os.PathLike], metadata: Optional[Dict[str, str]] = None
) -> None:
    """
    Saves a dictionary of tensors into raw bytes in bintensors format.

    Args:
        tensor_dict (`Dict[str, np.ndarray]`):
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
    from bintensors.numpy import save_file
    import numpy as np

    tensors = {"embedding": np.zeros((512, 1024)), "attention": np.zeros((256, 256))}
    save_file(tensors, "model.bintensors")
    ```
    """
    flattened = {k: {"dtype": v.dtype.name, "shape": v.shape, "data": _tobytes(v)} for k, v in tensor_dict.items()}
    serialize_file(flattened, filename, metadata=metadata)


def save_with_checksum(
    tensor_dict: Dict[str, np.ndarray], metadata: Optional[Dict[str, str]] = None
) -> Tuple[bytes, bytes]:
    """
    Saves a dictionary of tensors into raw bytes in bintensors format.

    Args:
        tensors (`Dict[str, torch.Tensor]`):
            The incoming tensors. Tensors need to be contiguous and dense.
        metadata (`Dict[str, str]`, *optional*, defaults to `None`):
            Optional text only metadata you might want to save in your header.
            For instance it can be useful to specify more about the underlying
            tensors. This is purely informative and does not affect tensor loading.

    Returns:
        `bytes`: The raw bytes representing the format

    Example:

    ```python
    from bintensors.torch import save
    import torch

    tensors = {"embedding": torch.zeros((512, 1024)), "attention": torch.zeros((256, 256))}
    checksum, byte_data = save_with_checksum(tensors)
    ```
    """
    flattened = {k: {"dtype": v.dtype.name, "shape": v.shape, "data": _tobytes(v)} for k, v in tensor_dict.items()}
    checksum, serialized = serialize_checksum(flattened, metadata=metadata)
    result = bytes(checksum), bytes(serialized)
    return result


def load(data: bytes) -> Dict[str, np.ndarray]:
    """
    Loads a bintensors file into numpy format from pure bytes.

    Args:
        data (`bytes`):
            The content of a bintensors file

    Returns:
        `Dict[str, np.ndarray]`: dictionary that contains name as key, value as `np.ndarray` on cpu

    Example:

    ```python
    from bintensors.numpy import load

    file_path = "./my_folder/bert.bintensors"
    with open(file_path, "rb") as f:
        data = f.read()

    loaded = load(data)
    ```
    """
    flat = deserialize(data)
    return _view2np(flat)


def load_file(filename: Union[str, os.PathLike]) -> Dict[str, np.ndarray]:
    """
    Loads a bintensors file into numpy format.

    Args:
        filename (`str`, or `os.PathLike`)):
            The name of the file which contains the tensors

    Returns:
        `Dict[str, np.ndarray]`: dictionary that contains name as key, value as `np.ndarray`

    Example:

    ```python
    from bintensors.numpy import load_file

    file_path = "./my_folder/bert.bintensors"
    loaded = load_file(file_path)
    ```
    """
    result = {}
    with safe_open(filename, framework="np") as f:
        for k in f.offset_keys():
            result[k] = f.get_tensor(k)
    return result


# np.float8 formats require 2.1; we do not support these dtypes on earlier versions
_float8_e4m3fn = getattr(np, "float8_e4m3fn", None)
_float8_e5m2 = getattr(np, "float8_e5m2", None)

_TYPES = {
    "F64": np.float64,
    "F32": np.float32,
    "F16": np.float16,
    "BF16": np.bfloat16,
    "I64": np.int64,
    "U64": np.uint64,
    "I32": np.int32,
    "U32": np.uint32,
    "I16": np.int16,
    "U16": np.uint16,
    "I8": np.int8,
    "U8": np.uint8,
    "BOOL": np.bool,
    "F8_E4M3": _float8_e4m3fn,
    "F8_E5M2": _float8_e5m2,
}


def _getdtype(dtype_str: str) -> np.dtype:
    return _TYPES[dtype_str]


def _view2np(safeview) -> Dict[str, np.ndarray]:
    result = {}
    for k, v in safeview:
        dtype = _getdtype(v["dtype"])
        arr = np.frombuffer(v["data"], dtype=dtype).reshape(v["shape"])
        result[k] = arr
    return result


def _is_little_endian(tensor: np.ndarray) -> bool:
    byteorder = tensor.dtype.byteorder
    if byteorder == "=":
        if sys.byteorder == "little":
            return True
        else:
            return False
    elif byteorder == "|":
        return True
    elif byteorder == "<":
        return True
    elif byteorder == ">":
        return False
    raise ValueError(f"Unexpected byte order {byteorder}")
