import os
from typing import Union

import safetensors as sf
from safetensors import numpy as numpy_sf

import bintensors as bt
from bintensors import numpy

__all__ = ["sf2bt", "bt2sf"]


def sf2bt(filename: Union[os.PathLike, str]) -> None:
    """
    convert safetensors file bintensors file.
    """
    with sf.safe_open(filename, framework="numpy") as model:
        layers = model.keys()
        tensor_dict = {layer: model.get_tensor(layer) for layer in layers}
        numpy.save_file(tensor_dict, filename, metadata=model.metadata())


def bt2sf(filename: Union[os.PathLike, str]) -> None:
    """
    convert bintensors file safetensors file.
    """
    with bt.safe_open(filename, framework="numpy") as model:
        layers = model.keys()
        tensor_dict = {layer: model.get_tensor(layer) for layer in layers}
        numpy_sf.save_file(tensor_dict, filename, metadata=model.metadata())
