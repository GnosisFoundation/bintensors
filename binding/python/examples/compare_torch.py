import os
import time
import torch
from typing import Dict
from bintensors.torch import save_file, load_file

def create_gpt2(n_layers: int) -> Dict[str, torch.Tensor]:
    """Create GPT-2 model architecture with specified number of layers."""
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

# Create the GPT-2 model tensors
tensor_dict = create_gpt2(20)

# Measure the time to save the model in custom .btf format
start_time = time.perf_counter()
save_file(tensor_dict, "gpt_tensor.bt")
save_time_btf = time.perf_counter() - start_time
print(f"Custom Binary Tensor Format saving time: {save_time_btf:.4f} seconds")

# Measure the time to save the model in PyTorch .pt format
start_time = time.perf_counter()
torch.save(tensor_dict, "gpt_tensor.pt")
save_time_pt = time.perf_counter() - start_time
print(f"PyTorch saving time: {save_time_pt:.4f} seconds")

# Measure the time to load the tensor from the PyTorch .pt file
start_time = time.perf_counter()
loaded_tensor_pt = torch.load("gpt_tensor.pt")
load_time_pt = time.perf_counter() - start_time
print(f"PyTorch loading time: {load_time_pt:.4f} seconds")

# Measure the time to load the tensor from the custom .btf file
start_time = time.perf_counter()
loaded_tensor_btf = load_file("gpt_tensor.bt")
load_time_btf = time.perf_counter() - start_time
print(f"Custom Binary Tensor Format loading time: {load_time_btf:.4f} seconds")

# clean up
os.remove("gpt_tensor.bt")
os.remove("gpt_tensor.pt")