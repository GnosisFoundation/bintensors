import pytest

import os
import tempfile
import torch

from typing import Dict, Tuple
from bintensors.torch import load, save, save_file, load_file, safe_open, save_with_checksum


def _compare_torch_tensors(lhs: torch.Tensor, rhs: torch.Tensor) -> bool:
    return torch.equal(lhs, rhs)


def _ASSERT_SHAPE(value: Tuple[int, ...], expected: Tuple[int, ...]) -> bool:
    assert value == expected, f"assert mismatch of shapes of a tensor, found {value}, expected {expected}"


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


class ToyRegressionModel(torch.nn.Module):
    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.ln_1 = torch.nn.Linear(n_in, n_out)
        self.output = torch.nn.Linear(n_in, 1)

    def forward(self, x: torch.Tensor):
        x = self.ln_1(x)
        x = torch.nn.functional.sigmoid(x)
        x = self.output(x)
        return x


def test_pt_save_and_load_gpt2_tensors_dict():
    small_gpt2 = create_gpt2_tensors_dict(2)
    buffer = save(small_gpt2)
    model = load(buffer)
    all((_compare_torch_tensors(small_gpt2[key], model[key]) for key in small_gpt2.keys()))


def test_pt_save_and_load_zero_sized_tensors():
    _SHAPES_A, _SHAPES_B = (0,), (0, 0, 0)
    tensor_dict = {"ln.weight": torch.zeros(_SHAPES_A), "ln.bias": torch.zeros(_SHAPES_B)}

    buffer = save(tensor_dict)
    loaded_tensor_dict = load(buffer)
    _ASSERT_SHAPE(tuple(loaded_tensor_dict["ln.weight"].shape), _SHAPES_A)
    _ASSERT_SHAPE(tuple(loaded_tensor_dict["ln.bias"].shape), _SHAPES_B)


def test_pt_tensor_dtype_roundtrip():
    tensor_dict = {
        "float32": torch.zeros((10, 10), dtype=torch.float32),
        "float64": torch.zeros((10, 10), dtype=torch.float64),
        "int32": torch.zeros((10, 10), dtype=torch.int32),
        "int64": torch.zeros((10, 10), dtype=torch.int64),
    }
    buffer = save(tensor_dict)
    loaded_dict = load(buffer)

    for key, value in tensor_dict.items():
        assert _compare_torch_tensors(loaded_dict[key], value)
        assert loaded_dict[key].dtype == value.dtype


def test_nn_module_toy_model():
    model = ToyRegressionModel(10, 10)

    tensor_dict = model.state_dict()
    buffer = save(tensor_dict)
    loaded_dict = load(buffer)

    assert all((_compare_torch_tensors(loaded_dict[key], value) for (key, value) in tensor_dict.items()))

    model.load_state_dict(state_dict=loaded_dict, strict=True)


def test_pt_invalid_tensor_dict_raises_error():
    invalid_dict = {"valid": torch.zeros((5, 5)), "invalid": "string_value"}

    with pytest.raises(Exception):
        _ = save(invalid_dict)


def test_pt_save_file_and_load_file_consistency():
    tensor_dict = create_gpt2_tensors_dict(1)
    filename = ""
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        filename = tmp.name
        save_file(tensor_dict, filename)
        loaded_dict = load_file(filename)

        for key, value in tensor_dict.items():
            assert _compare_torch_tensors(loaded_dict[key], value)


def test_pt_safe_open_access_and_metadata():
    tensor_dict = create_gpt2_tensors_dict(1)
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        filename = tmp.name
        save_file(tensor_dict, filename)
        with safe_open(filename, "pt") as model:
            assert model.get_tensor("h.0.ln_1.weight") is not None
            assert model.get_tensor("h.0.ln_1.bias") is not None
            assert model.metadata() is None


def test_checksum_two_diffrent_models():
    model_1 = { "ln.weight" : torch.rand((10,10)), "ln.bias" : torch.rand((10)) }
    model_2 = { "ln.weight" : torch.rand((10,10)), "ln.bias" : torch.rand((10)) }

    checksum1, _ = save_with_checksum(model_1)
    checksum2, _ = save_with_checksum(model_2)

    assert checksum1 != checksum2, "These checksum are not equivilent"


def test_checksum_two_same_models():
    model_1 = { "ln.weight" : torch.zeros((2,2)), "ln.bias" : torch.zeros((10)) }
    model_2 = { "ln.weight" : torch.zeros((2,2)), "ln.bias" : torch.zeros((10)) }

    for _ in range(1000):
        checksum1, _ = save_with_checksum(model_1)
        checksum2, _ = save_with_checksum(model_2)
        assert checksum1 == checksum2, "These checksum are equivilent"
