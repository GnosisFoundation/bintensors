# Safetensors attack proposal #1 (https://github.com/huggingface/safetensors/blob/main/attacks/safetensors_abuse_attempt_1.py)
import sys
import os
import torch
from bintensors import BintensorError
from bintensors.torch import load_file, save_file

filename = "bintensors_abuse_attempt_1.bt"


def create_payload():
    weights = {"weight": torch.zeros((2, 2))}
    save_file(weights, filename)

    with open(filename, "r+b") as f:
        f.seek(0)
        # Now the header claims 2**32 - xx even though the file is small
        n = 1000
        n_bytes = n.to_bytes(8, "little")
        f.write(n_bytes)


create_payload()
# This properly crashes with an out of bounds exception.
try:
    test = load_file(filename)
except BintensorError as e:
    print(f"This is expected; {e}")
    os.remove(filename)  # clean up file
    sys.exit(0)
