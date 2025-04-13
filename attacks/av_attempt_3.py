# Safetensors attack proposal #4
import os
from bintensors.torch import load_file
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

from typing import Tuple


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

filename = "bintensors_abuse_attempt_3.bt"


def create_payload(size: int):
    """Generates a binary payload with tensor metadata and hash map layout."""
    shape = (2, 2)
    tensor_chunk_length = shape[0] * shape[1] * 4  # Size of a tensor buffer

    length = encode_unsigned_variant_encoding(size)

    # Create tensor info buffer
    header = b"".join(encode_header(f"weight_{i}", "F32", shape, (0, tensor_chunk_length)) for i in range(size))

    # Construct full layout
    layout = b"\x00" + length + header
    layout += b" " * (((8 - len(layout)) % 8) % 8)
    n = len(layout)
    n_header = n.to_bytes(8, "little")

    # Write payload to file
    with open(filename, "wb") as f:
        f.write(n_header)
        f.write(layout)
        f.write(b"\0" * tensor_chunk_length)

    
    print(f"[✓] Payload written: {filename}")


if __name__ == "__main__":
    create_payload(100)
    print(f"[✓] Size: {os.path.getsize(filename) / 1_000_000:.5f} MB")
    load_file(filename)

    
