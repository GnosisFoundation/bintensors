import os
import re
import warnings
import logging
from itertools import islice

import torch

from typing import Dict, Set, List, Union

from bintensors.torch import save_file, load_file

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__file__)

__dir__ = os.path.dirname(os.path.realpath(__file__))

def create_gpt2(n_layers: int) -> Dict[str, torch.Tensor]:
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


def save_file_chunk(filename: str, tensor_dict: Dict[str, torch.Tensor], metadata: Dict[str, str] = None):
    save_file(tensor_dict, filename, metadata)

def load_file_chunks(files : List[Union[os.PathLike, str]], device : Union[int,str] = "cpu") -> Dict[str, torch.Tensor]:
    result = {}
    for file in files:
        result |= load_file(file, device)
    
    return result

def chunk_model(model: Dict[str, torch.Tensor], n_chunks: int, dir_out : os.PathLike) -> Set[Dict[str, torch.Tensor]]:
    offset_chunks = len(model) % n_chunks
    if offset_chunks != 0:
        warnings.warn(f"model archetecture is mod of {n_chunks}")


    try:
        total = int(len(model)/n_chunks)
        if offset_chunks > 0:
            total += int(n_chunks / offset_chunks)
        offest = 0
        items = model.items()
        files = []
        while len(model) > offest:
            # convert modle into chunk
            chunk = dict(islice(items, offest, offest + n_chunks))
            
            # determine stdout file
            filename = f"chunk-{offest}-{offest + n_chunks}.bt"
            output = os.path.join(__dir__, dir_out, filename)
            files.append(output)
            # write to file
            save_file_chunk(output, chunk)
            
            iteration = f"{len(files)}/{total}"
            out_dir = '/'.join(re.split(r'[\\/]', output)[-3:-1])
            logger.info(f"{iteration:<8}: {filename} writen to path ../{out_dir}/")
            
            offest += n_chunks
    except Exception as e:
        raise e
    
    return files


model = create_gpt2(10)
files = chunk_model(model, 14, "output")

gpt = load_file_chunks(files, device="cpu")
print(gpt.keys())
