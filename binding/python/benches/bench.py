#!/usr/bin/env python3
import time
import torch
import pyperf
import logging

from typing import Dict, Tuple

from bintensors.torch import serialize, load, _flatten

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

py_runner = pyperf.Runner()


def create_gpt2(n_layers: int = 20) -> Dict[str, torch.Tensor]:
    tensors = {}
    tensors["wte"] = torch.zeros((50257, 768))
    tensors["wpe"] = torch.zeros((1024, 768))

    for i in range(n_layers):
        tensors[f"h.{i}.ln_1.weight"] = torch.zeros((768,))
        tensors[f"h.{i}.ln_1.bias"] = torch.zeros((768,))
        tensors[f"h.{i}.attn.bias"] = torch.zeros((1, 1, 1024, 1024))
        tensors[f"h.{i}.attn.c_attn.weight"] = torch.zeros((768, 2304))
        tensors[f"h.{i}.attn.c_attn.bias"] = torch.zeros((2304))
        tensors[f"h.{i}.attn.c_proj.weight"] = torch.zeros((768, 768))
        tensors[f"h.{i}.attn.c_proj.bias"] = torch.zeros((768))
        tensors[f"h.{i}.ln_2.weight"] = torch.zeros((768))
        tensors[f"h.{i}.ln_2.bias"] = torch.zeros((768))
        tensors[f"h.{i}.mlp.c_fc.weight"] = torch.zeros((768, 3072))
        tensors[f"h.{i}.mlp.c_fc.bias"] = torch.zeros((3072))
        tensors[f"h.{i}.mlp.c_proj.weight"] = torch.zeros((3072, 768))
        tensors[f"h.{i}.mlp.c_proj.bias"] = torch.zeros((768))

    tensors["ln_f.weight"] = torch.zeros((768))
    tensors["ln_f.bias"] = torch.zeros((768))

    return tensors


def serialize_gpt2() -> Tuple[bytes, float]:
    """Serialize small GPT-2 model and return the buffer and time taken."""
    logger.info(f"Creating GPT-2 model with 20 layers")
    tensor_dict = create_gpt2()

    # Calculate size
    space = _flatten(tensor_dict)
    bytes_used = sum(len(v["data"]) for _, v in space.items())
    logger.info(f"{bytes_used * 1e-6:.2f} MB allocated (not including metadata)")

    t0 = time.perf_counter()
    serialized_buffer = serialize(space)
    serialization_time = time.perf_counter() - t0

    logger.info(f"Serialization took {serialization_time:.4f} seconds")
    logger.info(f"Serialized size: {len(serialized_buffer) * 1e-6:.2f} MB")

    return serialized_buffer, serialization_time


def load_model_bench(buffer: bytes) -> float:
    """Load model from buffer and return the time taken."""
    t0 = time.perf_counter()
    _ = load(buffer)
    loading_time = time.perf_counter() - t0

    logger.info(f"Loading took {loading_time:.4f} seconds")

    return loading_time


def main():
    # Run the "serialize_gpt2" benchmark
    py_runner.bench_func("serialize_gpt2_bench", serialize_gpt2)

    # Run the "load_gpt2" benchmark
    serialized_buffer, _ = serialize_gpt2()
    py_runner.bench_func("load_gpt2_bench", load_model_bench, *[serialized_buffer])


if __name__ == "__main__":
    main()
