import pytest

import tempfile
import numpy as np

import mlx.core as mlx


from typing import Dict, Tuple
from bintensors.mlx import load, load_file, save, save_file, safe_open, save_with_checksum


def _compare_mlx_array(lhs: mlx.array, rhs: mlx.array) -> bool:
    return mlx.array_equal(lhs, rhs)


def _ASSERT_SHAPE(value: Tuple[int, ...], expected: Tuple[int, ...]) -> bool:
    assert value == expected, f"assert mismatch of shapes of a tensor, found {value}, expected {expected}"


def create_gpt2_numpy_dict(n_layers: int) -> Dict[str, mlx.array]:
    tensors = {}
    tensors["wte"] = mlx.zeros((50257, 768))
    tensors["wpe"] = mlx.zeros((1024, 768))
    for i in range(n_layers):
        tensors[f"h.{i}.ln_1.weight"] = mlx.zeros((768,))
        tensors[f"h.{i}.ln_1.bias"] = mlx.zeros((768,))
        tensors[f"h.{i}.attn.bias"] = mlx.zeros((1, 1, 1024, 1024))
        tensors[f"h.{i}.attn.c_attn.weight"] = mlx.zeros((768, 2304))
        tensors[f"h.{i}.attn.c_attn.bias"] = mlx.zeros((2304))
        tensors[f"h.{i}.attn.c_proj.weight"] = mlx.zeros((768, 768))
        tensors[f"h.{i}.attn.c_proj.bias"] = mlx.zeros((768))
        tensors[f"h.{i}.ln_2.weight"] = mlx.zeros((768))
        tensors[f"h.{i}.ln_2.bias"] = mlx.zeros((768))
        tensors[f"h.{i}.mlp.c_fc.weight"] = mlx.zeros((768, 3072))
        tensors[f"h.{i}.mlp.c_fc.bias"] = mlx.zeros((3072))
        tensors[f"h.{i}.mlp.c_proj.weight"] = mlx.zeros((3072, 768))
        tensors[f"h.{i}.mlp.c_proj.bias"] = mlx.zeros((768))
    tensors["ln_f.weight"] = mlx.zeros((768))
    tensors["ln_f.bias"] = mlx.zeros((768))
    return tensors

def test_save_and_load_gpt2_tensors_dict_mlx():
    small_gpt2 = create_gpt2_numpy_dict(2)
    buffer = save(small_gpt2)
    model = load(buffer)
    assert all((_compare_mlx_array(small_gpt2[key], model[key]) for key in small_gpt2.keys()))


def test_save_and_load_zero_sized_tensors_mlx():
    _SHAPES_A, _SHAPES_B = (0,), (0, 0, 0)
    tensor_dict = {"ln.weight": mlx.zeros(_SHAPES_A), "ln.bias": mlx.zeros(_SHAPES_B)}

    buffer = save(tensor_dict)
    loaded_tensor_dict = load(buffer)
    _ASSERT_SHAPE(tuple(loaded_tensor_dict["ln.weight"].shape), _SHAPES_A)
    _ASSERT_SHAPE(tuple(loaded_tensor_dict["ln.bias"].shape), _SHAPES_B)


def test_tensor_dtype_roundtrip_mlx():
    tensor_dict = {
        "float32": mlx.zeros((10, 10), dtype=mlx.float32),
        "float64": mlx.zeros((10, 10), dtype=mlx.float32),
        "int32": mlx.zeros((10, 10), dtype=mlx.int32),
        "int64": mlx.zeros((10, 10), dtype=mlx.float32),
    }
    buffer = save(tensor_dict)
    loaded_dict = load(buffer)

    for key, value in tensor_dict.items():
        assert _compare_mlx_array(loaded_dict[key], value)
        assert loaded_dict[key].dtype == value.dtype


def test_save_and_load_large_gpt2_dict_mlx():
    tensor_dict = create_gpt2_numpy_dict(20)
    buffer = save(tensor_dict)
    loaded_dict = load(buffer)
    assert all((_compare_mlx_array(loaded_dict[key], value) for key, value in tensor_dict.items()))


def test_tensor_dict_with_mixed_shapes_mlx():
    tensor_dict = {
        "scalar": mlx.array(5.0),
        "vector": mlx.array([1.0, 2.0, 3.0]),
        "matrix": mlx.array([[1.0, 2.0], [3.0, 4.0]]),
        "tensor3d": mlx.random.normal((2, 3, 4)),
    }

    buffer = save(tensor_dict)
    loaded_dict = load(buffer)

    for key, value in tensor_dict.items():
        assert _compare_mlx_array(loaded_dict[key], value)
        assert loaded_dict[key].shape == value.shape


def test_invalid_tensor_dict_raises_error_mlx():
    invalid_dict = {"valid": mlx.zeros((5, 5)), "invalid": "string_value"}

    with pytest.raises(Exception):
        _ = save(invalid_dict)


def test_save_file_and_load_file_consistency_mlx():
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
                assert _compare_mlx_array(loaded_dict[key], value)


def test_safe_open_access_and_metadata_mlx():
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


def test_safe_open_access_with_metadata_mlx():
    tensor_dict = create_gpt2_numpy_dict(1)
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        filename = tmp.name

        save_file(tensor_dict, filename, metadata={"hello": "world"})
        with safe_open(filename, "numpy") as model:
            assert model.get_tensor("h.0.ln_1.weight") is not None
            assert model.get_tensor("h.0.ln_1.bias") is not None
            assert model.metadata()["hello"] == "world"


def test_checksum_two_diffrent_models_mlx():
    model_1 = {"ln.weight": mlx.random.normal((10, 10)), "ln.bias": mlx.random.normal((10,))}
    model_2 = {"ln.weight": mlx.random.normal((10, 10)), "ln.bias": mlx.random.normal((10,))}

    checksum1, _ = save_with_checksum(model_1)
    checksum2, _ = save_with_checksum(model_2)

    assert checksum1 != checksum2, "These checksum are not equivilent"


def test_checksum_two_same_models_mlx():
    model_1 = {"ln.weight": mlx.zeros((2, 2)), "ln.bias": mlx.zeros((10))}
    model_2 = {"ln.weight": mlx.zeros((2, 2)), "ln.bias": mlx.zeros((10))}

    for _ in range(1000):
        checksum1, _ = save_with_checksum(model_1)
        checksum2, _ = save_with_checksum(model_2)
        assert checksum1 == checksum2, "These checksum are equivilent"
