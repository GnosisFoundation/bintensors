import pytest
import struct

import torch
from typing import List, Dict, Tuple
from itertools import chain

from bintensors import BintensorError
from bintensors.torch import save, load

_DTYPE = {
    "BOL": 0,
    "U8": 1,
    "I8": 2,
    "F8_E5M2": 3,
    "F8_E4M3": 4,
    "I16": 5,
    "U16": 6,
    "F16": 7,
    "BF16": 8,
    "I32": 9,
    "U32": 10,
    "F32": 11,
    "F64": 12,
    "I64": 13,
    "F64": 14,
}


def encode_unsigned_variant_encoding(number: int) -> bytes:
    """Encodes an unsigned integer into a variable-length format."""
    if number > 0xFFFFFFFF:
        return b"\xfd" + number.to_bytes(8, "little")
    elif number > 0xFFFF:
        return b"\xfc" + number.to_bytes(4, "little")
    elif number > 0xFA:
        return b"\xfb" + number.to_bytes(2, "little")
    else:
        return number.to_bytes(1, "little")


def encode_header(id: str, dtype: str, shape: Tuple[int, ...], offset: Tuple[int, int]) -> bytes:
    """Encodes the struct TensorInfo into byte buffer with string ID prefix."""
    if dtype not in _DTYPE:
        raise ValueError(f"Unsupported dtype: {dtype}")

    encoded_id = encode_unsigned_variant_encoding(len(id)) + id.encode("utf-8")

    # Compose numeric fields
    numeric_layout = chain(
        [_DTYPE[dtype], len(shape)],
        shape,
        offset
    )

    encoded_tensor_info = b"".join(encode_unsigned_variant_encoding(x) for x in numeric_layout)

    return encoded_id + encoded_tensor_info


def test_empty_file():
    "bintensors allows empty dictonary"
    tensor_dict = {}
    buffer = save(tensor_dict)
    # decouple first 8 bytes part of the buffer unsinged long long
    header_size = struct.unpack("<Q", buffer[0:8])[0]
    # header size + metadata + empty tensors
    MAX_FILE_SIZE = 8 + header_size
    assert header_size == 8, "expected packed buffer shoudl be unsinged interger 8."
    assert buffer[8:] == b"\x00\x00      ", "expected empty metadata fields."
    assert MAX_FILE_SIZE == len(buffer), "These should  be equal"


def test_man_cmp():
    size = 2
    shape = (2, 2)
    tensor_chunk_length = shape[0] * shape[1] * 4  # Size of a tensor buffer

    length = encode_unsigned_variant_encoding(size)

    # Create tensor info buffer
    tensor_info_buffer = b"".join(
        encode_header(
            f"weight_{i}",
            "F32",
            shape,
            (i * tensor_chunk_length, i * tensor_chunk_length + tensor_chunk_length),
        )
        for i in range(size)
    )
    layout = length + tensor_info_buffer
    layout = b"\0" + layout
    layout += b" " * (((8 - len(layout)) % 8) % 8)
    n = len(layout)
    n_header = n.to_bytes(8, "little")
    
    expected = n_header + layout + (b"\0" * tensor_chunk_length * size)

    tensor_dict = {"weight_0": torch.zeros(shape), "weight_1": torch.zeros(shape)}

    buffer = save(tensor_dict)
    # we need to check both since there is no order in the hashmap
    assert buffer == expected, f"got {buffer}, and expected {expected}"


def test_missmatch_length_of_metadata_large():
    size = 2
    shape = (2, 2)
    tensor_chunk_length = shape[0] * shape[1] * 4  # Size of a tensor buffer

    length = encode_unsigned_variant_encoding(size * 1000)

    # Create tensor info buffer
    tensor_info_buffer = b"".join(
        encode_header(
            f"weight_{i}",
            "F32",
            shape,
            (i * tensor_chunk_length, i * tensor_chunk_length + tensor_chunk_length),
        )
        for i in range(size)
    )
    layout = length + tensor_info_buffer
    layout = b"\0" + layout
    layout += b" " * (((8 - len(layout)) % 8) % 8)
    n = len(layout)
    n_header = n.to_bytes(8, "little")
    
    # layout together
    buffer = n_header + layout + b"\0" * (tensor_chunk_length * size)

    with pytest.raises(BintensorError):
        # this is not a valid since the metadata
        # size doe not match as it too big
        _ = load(buffer)


def test_missmatch_length_of_metadata_small():
    size = 2
    shape = (2, 2)
    tensor_chunk_length = shape[0] * shape[1] * 4  # Size of a tensor buffer

    length = encode_unsigned_variant_encoding(size - 1)

    # Create tensor info buffer
    tensor_info_buffer = b"".join(
        encode_header(
            f"weight_{i}",
            "F32",
            shape,
            (i * tensor_chunk_length, i * tensor_chunk_length + tensor_chunk_length),
        )
        for i in range(size)
    )
    layout = length + tensor_info_buffer
    layout = b"\0" + layout
    layout += b" " * (((8 - len(layout)) % 8) % 8)
    n = len(layout)
    n_header = n.to_bytes(8, "little")

    # layout together
    buffer = n_header + layout + b"\0" * (tensor_chunk_length * size)

    with pytest.raises(BintensorError):
        # this is not a valid since the metadata
        # size doe not match as it too big
        _ = load(buffer)

