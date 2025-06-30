import torch
from bintensors.io import is_btfile
from bintensors.torch import save, save_file

import tempfile


def test_buffer_is_btfile():
    tensor_dict = {
        "float32": torch.zeros((10, 10), dtype=torch.float32),
        "float64": torch.zeros((10, 10), dtype=torch.float64),
        "int32": torch.zeros((10, 10), dtype=torch.int32),
        "int64": torch.zeros((10, 10), dtype=torch.int64),
    }
    buffer = save(tensor_dict)
    assert is_btfile(buffer)


def test_filename_is_btfile():
    with tempfile.NamedTemporaryFile() as file:
        tensor_dict = {
            "float32": torch.zeros((10, 10), dtype=torch.float32),
            "float64": torch.zeros((10, 10), dtype=torch.float64),
            "int32": torch.zeros((10, 10), dtype=torch.int32),
            "int64": torch.zeros((10, 10), dtype=torch.int64),
        }
        save_file(tensor_dict, file.name)
        assert is_btfile(file.name)


def test_reader_is_btfile():
    with tempfile.NamedTemporaryFile() as file:
        tensor_dict = {
            "float32": torch.zeros((10, 10), dtype=torch.float32),
            "float64": torch.zeros((10, 10), dtype=torch.float64),
            "int32": torch.zeros((10, 10), dtype=torch.int32),
            "int64": torch.zeros((10, 10), dtype=torch.int64),
        }
        save_file(tensor_dict, file.name)
        file = open(file.name, "rb")
        assert is_btfile(file)
