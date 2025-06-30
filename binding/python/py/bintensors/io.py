import os
import io
from ._bintensors_rs import _validate

from typing import Union

__all__ = ["is_btfile"]


def is_btfile(name: Union[os.PathLike, str, bytes, io.BufferedReader]):
    """

     Return True if `name` is a valid bintensors file or buffer, else False.

    Args:
        name (`str`, `os.PathLike`, `bytes`, or `BufferedReader`):
            A path to a file, a file-like object, or a bytes buffer.

    Returns:
        (`bool`): True if it's a valid bintensors file, False otherwise.
    """
    try:
        if isinstance(name, bytes):
            return _validate(name)

        elif hasattr(name, "read"):
            pos = name.tell()
            name.seek(0)
            chunk = name.read(-1)
            name.seek(pos)
            return _validate(chunk)

        elif isinstance(name, (os.PathLike, str)):
            file = open(name, "rb")
            chunk = file.read(-1)
            file.close()
            return _validate(chunk)
        else:
            raise ValueError("Invalid supported type.")
    except Exception:
        return False
