import torch

import tempfile
from typing import Dict

import bintensors as bt
from bintensors.io import is_btfile
from bintensors.torch import save_file as bt_save_file
from bintensors.safetensors import sf2bt, bt2sf

from safetensors.torch import save_file


def create_gpt2_tensors_dict(n_layers: int) -> Dict[str, torch.Tensor]:
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


def test_conversion_sf2bt():
    with tempfile.NamedTemporaryFile() as file:
        tensor_dict = create_gpt2_tensors_dict(1)
        save_file(tensor_dict, file.name)
        sf2bt(file.name)
        assert is_btfile(file.name)


def test_conversion_bt2sf():
    with tempfile.NamedTemporaryFile() as file:
        tensor_dict = create_gpt2_tensors_dict(1)
        bt_save_file(tensor_dict, file.name)
        bt2sf(file.name)
        assert not is_btfile(file.name)
