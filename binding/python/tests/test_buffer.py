import pytest
import struct

import torch
import tempfile
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


def encode_tensor_info(dtype: str, shape: Tuple[int, ...], offset: Tuple[int, int]) -> List[bytes]:
    """Encodes the struct TensorInfo into byte buffer"""
    if dtype not in _DTYPE:
        raise ValueError(f"Unsupported dtype: {dtype}")

    # flatten out the tensor info
    layout = chain([_DTYPE[dtype], len(shape)], shape, offset)
    return b"".join(list(map(encode_unsigned_variant_encoding, layout)))


def encode_hash_map(index_map: Dict[str, int]) -> List[bytes]:
    """Encodes a dictionary of string keys and integer values."""
    length = encode_unsigned_variant_encoding(len(index_map))

    hash_map_layout = chain.from_iterable(
        (
            encode_unsigned_variant_encoding(len(k)),
            k.encode("utf-8"),
            encode_unsigned_variant_encoding(v),
        )
        for k, v in index_map.items()
    )

    return b"".join(chain([length], hash_map_layout))


def test_empty_file():
    "bintensors allows empty dictonary"
    tensor_dict = {}
    buffer = save(tensor_dict)
    # decouple first 8 bytes part of the buffer unsinged long long
    header_size = struct.unpack("<Q", buffer[0:8])[0]
    # header size + metadata + empty tensors
    MAX_FILE_SIZE = 8 + header_size
    assert header_size == 8, "expected packed buffer shoudl be unsinged interger 8."
    assert buffer[8:] == b"\x00\x00\x00     ", "expected empty metadata fields."
    assert MAX_FILE_SIZE == len(buffer), "These should  be equal"


def test_man_cmp():
    size = 2
    shape = (2, 2)
    tensor_chunk_length = shape[0] * shape[1] * 4  # Size of a tensor buffer

    length = encode_unsigned_variant_encoding(size)

    # Create tensor info buffer
    tensor_info_buffer = b"".join(
        encode_tensor_info(
            "F32",
            shape,
            (i * tensor_chunk_length, i * tensor_chunk_length + tensor_chunk_length),
        )
        for i in range(size)
    )
    layout_tensor_info = length + tensor_info_buffer

    expected = []
    for (start, end, step) in [(0, size, 1), (size - 1, -1, -1)]:
        # Create hash map layout
        hash_map_layout = encode_hash_map({f"weight_{i}": i for i in range(start, end, step)})

        # Construct full layout
        layout = b"\0" + layout_tensor_info + hash_map_layout
        layout += b" " * (((8 - len(layout)) % 8) % 8)
        n = len(layout)
        n_header = n.to_bytes(8, "little")

        # layout together
        buffer = n_header + layout + b"\0" * (tensor_chunk_length * 2)
        expected.append(buffer)

    tensor_dict = {"weight_0": torch.zeros(shape), "weight_1": torch.zeros(shape)}

    buffer = save(tensor_dict)
    # we need to check both since there is no order in the hashmap
    assert buffer in expected, f"got {buffer}, and expected {expected}"


def test_missmatch_length_of_metadata_large():
    size = 2
    shape = (2, 2)
    tensor_chunk_length = shape[0] * shape[1] * 4  # Size of a tensor buffer

    length = encode_unsigned_variant_encoding(size * 1000)

    # Create tensor info buffer
    tensor_info_buffer = b"".join(
        encode_tensor_info(
            "F32",
            shape,
            (i * tensor_chunk_length, i * tensor_chunk_length + tensor_chunk_length),
        )
        for i in range(size)
    )
    layout_tensor_info = length + tensor_info_buffer

    expected = [0] * 2

    # Create hash map layout
    hash_map_layout = encode_hash_map({f"weight_{i}": i for i in range(0, 2, 1)})

    # Construct full layout
    layout = b"\0" + layout_tensor_info + hash_map_layout
    layout += b" " * (((8 - len(layout)) % 8) % 8)
    n = len(layout)
    n_header = n.to_bytes(8, "little")

    # layout together
    buffer = n_header + layout + b"\0" * (tensor_chunk_length * 2)

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
        encode_tensor_info(
            "F32",
            shape,
            (i * tensor_chunk_length, i * tensor_chunk_length + tensor_chunk_length),
        )
        for i in range(size)
    )
    layout_tensor_info = length + tensor_info_buffer

    # Create hash map layout
    hash_map_layout = encode_hash_map({f"weight_{i}": i for i in range(0, 2, 1)})

    # Construct full layout
    layout = b"\0" + layout_tensor_info + hash_map_layout
    layout += b" " * (((8 - len(layout)) % 8) % 8)
    n = len(layout)
    n_header = n.to_bytes(8, "little")

    # layout together
    buffer = n_header + layout + b"\0" * (tensor_chunk_length * 2)

    with pytest.raises(BintensorError):
        # this is not a valid since the metadata
        # size doe not match as it too big
        _ = load(buffer)


def test_missmatch_length_of_metadata():
    size = 2
    shape = (2, 2)
    tensor_chunk_length = shape[0] * shape[1] * 4  # Size of a tensor buffer

    # convert usize or unsigned long long into variant encoding
    length = encode_unsigned_variant_encoding(size * 1000)

    # Create tensor info byte buffer
    tensor_info_buffer = b"".join(
        encode_tensor_info(
            "F32",
            shape,
            (i * tensor_chunk_length, i * tensor_chunk_length + tensor_chunk_length),
        )
        for i in range(size)
    )
    layout_tensor_info = length + tensor_info_buffer

    # Create hash map layout
    hash_map_layout = encode_hash_map({f"weight_{i}": i for i in range(0, 2, 1)})

    # Construct full layout
    # metadata empty + tensor_info + hash_map_index_map
    layout = b"\0" + layout_tensor_info + hash_map_layout

    # empty padding
    layout += b" " * (((8 - len(layout)) % 8) % 8)
    n = len(layout)

    # size of full header (metadata + tensors info + index map)
    n_header = n.to_bytes(8, "little")

    # layout together into buffer
    buffer = n_header + layout + b"\0" * (tensor_chunk_length * 2)

    with pytest.raises(BintensorError):
        # this is not a valid since the metadata
        # size doe not match as it too big
        _ = load(buffer)

