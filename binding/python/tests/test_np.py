import pytest

import tempfile
import numpy as np

from typing import Dict, Tuple
from bintensors.numpy import load, load_file, save, save_file, safe_open, save_with_checksum


def _compare_np_array(lhs: np.ndarray, rhs: np.ndarray) -> bool:
    return np.array_equal(lhs, rhs)


def _ASSERT_SHAPE(value: Tuple[int, ...], expected: Tuple[int, ...]) -> bool:
    assert value == expected, f"assert mismatch of shapes of a tensor, found {value}, expected {expected}"


def create_gpt2_numpy_dict(n_layers: int) -> Dict[str, np.ndarray]:
    tensors = {}
    tensors["wte"] = np.zeros((50257, 768))
    tensors["wpe"] = np.zeros((1024, 768))
    for i in range(n_layers):
        tensors[f"h.{i}.ln_1.weight"] = np.zeros((768,))
        tensors[f"h.{i}.ln_1.bias"] = np.zeros((768,))
        tensors[f"h.{i}.attn.bias"] = np.zeros((1, 1, 1024, 1024))
        tensors[f"h.{i}.attn.c_attn.weight"] = np.zeros((768, 2304))
        tensors[f"h.{i}.attn.c_attn.bias"] = np.zeros((2304))
        tensors[f"h.{i}.attn.c_proj.weight"] = np.zeros((768, 768))
        tensors[f"h.{i}.attn.c_proj.bias"] = np.zeros((768))
        tensors[f"h.{i}.ln_2.weight"] = np.zeros((768))
        tensors[f"h.{i}.ln_2.bias"] = np.zeros((768))
        tensors[f"h.{i}.mlp.c_fc.weight"] = np.zeros((768, 3072))
        tensors[f"h.{i}.mlp.c_fc.bias"] = np.zeros((3072))
        tensors[f"h.{i}.mlp.c_proj.weight"] = np.zeros((3072, 768))
        tensors[f"h.{i}.mlp.c_proj.bias"] = np.zeros((768))
    tensors["ln_f.weight"] = np.zeros((768))
    tensors["ln_f.bias"] = np.zeros((768))
    return tensors


def test_save_and_load_gpt2_tensors_dict():
    small_gpt2 = create_gpt2_numpy_dict(2)
    buffer = save(small_gpt2)
    model = load(buffer)
    assert all((_compare_np_array(small_gpt2[key], model[key]) for key in small_gpt2.keys()))


def test_save_and_load_zero_sized_tensors():
    _SHAPES_A, _SHAPES_B = (0,), (0, 0, 0)
    tensor_dict = {"ln.weight": np.zeros(_SHAPES_A), "ln.bias": np.zeros(_SHAPES_B)}

    buffer = save(tensor_dict)
    loaded_tensor_dict = load(buffer)
    _ASSERT_SHAPE(tuple(loaded_tensor_dict["ln.weight"].shape), _SHAPES_A)
    _ASSERT_SHAPE(tuple(loaded_tensor_dict["ln.bias"].shape), _SHAPES_B)


def test_tensor_dtype_roundtrip():
    tensor_dict = {
        "float32": np.zeros((10, 10), dtype=np.float32),
        "float64": np.zeros((10, 10), dtype=np.float64),
        "int32": np.zeros((10, 10), dtype=np.int32),
        "int64": np.zeros((10, 10), dtype=np.int64),
    }
    buffer = save(tensor_dict)
    loaded_dict = load(buffer)

    for key, value in tensor_dict.items():
        assert _compare_np_array(loaded_dict[key], value)
        assert loaded_dict[key].dtype == value.dtype


def test_save_and_load_large_gpt2_dict():
    tensor_dict = create_gpt2_numpy_dict(20)
    buffer = save(tensor_dict)
    loaded_dict = load(buffer)
    assert all((_compare_np_array(loaded_dict[key], value) for key, value in tensor_dict.items()))


def test_tensor_dict_with_mixed_shapes():
    tensor_dict = {
        "scalar": np.array(5.0),
        "vector": np.array([1.0, 2.0, 3.0]),
        "matrix": np.array([[1.0, 2.0], [3.0, 4.0]]),
        "tensor3d": np.random.rand(2, 3, 4),
    }

    buffer = save(tensor_dict)
    loaded_dict = load(buffer)

    for key, value in tensor_dict.items():
        assert _compare_np_array(loaded_dict[key], value)
        assert loaded_dict[key].shape == value.shape


def test_invalid_tensor_dict_raises_error():
    invalid_dict = {"valid": np.zeros((5, 5)), "invalid": "string_value"}

    with pytest.raises(Exception):
        _ = save(invalid_dict)


def test_save_file_and_load_file_consistency():
    tensor_dict = create_gpt2_numpy_dict(1)
    filename = ""
    loaded_dict = {}
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        filename = tmp.name
        try:
            save_file(tensor_dict, filename)
            loaded_dict = load_file(filename)
        finally:
            for key, value in tensor_dict.items():
                assert _compare_np_array(loaded_dict[key], value)


def test_safe_open_access_and_metadata():
    tensor_dict = create_gpt2_numpy_dict(1)
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        filename = tmp.name
        # save file into tempfile
        save_file(tensor_dict, filename)

        # contex manager numpy framework allocation on bintensors file
        with safe_open(filename, "numpy") as model:
            assert model.get_tensor("h.0.ln_1.weight") is not None
            assert model.get_tensor("h.0.ln_1.bias") is not None
            assert model.metadata() is None


def test_safe_open_access_with_metadata():
    tensor_dict = create_gpt2_numpy_dict(1)
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        filename = tmp.name

        save_file(tensor_dict, filename, metadata={"hello": "world"})
        with safe_open(filename, "numpy") as model:
            assert model.get_tensor("h.0.ln_1.weight") is not None
            assert model.get_tensor("h.0.ln_1.bias") is not None
            assert model.metadata()["hello"] == "world"


def test_checksum_two_diffrent_models():
    model_1 = { "ln.weight" : np.random.random((10,10)), "ln.bias" : np.random.random((10)) }
    model_2 = { "ln.weight" : np.random.random((10,10)), "ln.bias" : np.random.random((10)) }

    checksum1, _ = save_with_checksum(model_1)
    checksum2, _ = save_with_checksum(model_2)

    assert checksum1 != checksum2, "These checksum are not equivilent"


def test_checksum_two_same_models():
    model_1 = { "ln.weight" : np.zeros((2,2)), "ln.bias" : np.zeros((10)) }
    model_2 = { "ln.weight" : np.zeros((2,2)), "ln.bias" : np.zeros((10)) }

    for _ in range(1000):
        checksum1, _ = save_with_checksum(model_1)
        checksum2, _ = save_with_checksum(model_2)
        assert checksum1 == checksum2, "These checksum are equivilent"


def test_checksum_two_same_models_with_diffrent_framework():
    import torch
    from bintensors.torch import save_with_checksum as save_with_checksum_pt
    model_1 = { "ln.weight" : np.zeros((2,2), dtype=np.float32), "ln.bias" : np.zeros((10), dtype=np.float32) }
    model_2 = { "ln.weight" : torch.zeros((2,2), dtype=torch.float32), "ln.bias" : torch.zeros((10), dtype=torch.float32) }

    for _ in range(1000):
        checksum1, _ = save_with_checksum(model_1)
        checksum2, _ = save_with_checksum_pt(model_2)
        assert checksum1 == checksum2, "These checksum are equivilent"

