@staticmethod
def deserialize(bytes):
    """
    Opens a bintensors lazily and returns tensors as asked

    Args:
        data (`bytes`):
            The byte content of a file

    Returns:
        (`List[str, Dict[str, Dict[str, any]]]`):
            The deserialized content is like:
                [("tensor_name", {"shape": [2, 3], "dtype": "float32", "data": b"..." }), (...)]
    """
    pass

@staticmethod
def serialize(tensor_dict, metadata=None):
    """
    Serializes raw data.

    Args:
        tensor_dict (`Dict[str, Dict[Any]]`):
            The tensor dict is like:
                {"tensor_name": {"dtype": "float32", "shape": [2, 3], "data": b"...."}}
        metadata (`Dict[str, str]`, *optional*):
            The optional purely text annotations

    Returns:
        (`bytes`):
            The serialized content.
    """
    pass

@staticmethod
def serialize_file(tensor_dict, filename, metadata=None):
    """
    Serializes raw data into file.

    Args:
        tensor_dict (`Dict[str, Dict[Any]]`):
            Dictorary of tensor data types, and there assoisated layer key
        filename (`str`, or `os.PathLike`):
            The name of the file to write into.
        metadata (`Dict[str, str]`, *optional*):
            The optional purely text annotations

    Returns:
        (`NoneType`):
            On success return `None`.
    """
    pass

class safe_open:
    """
    Opens a bintensors lazily and returns tensors as asked

    Args:
        filename (`str`, or `os.PathLike`):
            The filename to open

        framework (`str`):
            The framework you want you tensors in. Supported values:
            `pt`, `numpy`.

        device (`str`, defaults to `"cpu"`):
            The device on which you want the tensors.
    """

    def __init__(self, filename, framework, device=...):
        pass
    def __enter__(self):
        """
        Start the context manager
        """
        pass
    def __exit__(self, _exc_type, _exc_value, _traceback):
        """
        Exits the context manager
        """
        pass
    def get_slice(self, name):
        """
        Returns a full slice view object

        Args:
            name (`str`):
                The name of the tensor you want

        Returns:
            (`PySafeSlice`):
                A dummy object you can slice into to get a real tensor
        Example:
        ```python
        from bintensors import safe_open

        with safe_open("model.bintensors", framework="pt", device=0) as f:
            tensor_part = f.get_slice("embedding")[:, ::8]

        ```
        """
        pass
    def get_tensor(self, name):
        """
        Returns a full tensor

        Args:
            name (`str`):
                The name of the tensor you want

        Returns:
            (`Tensor`):
                The tensor in the framework you opened the file for.

        Example:
        ```python
        from bintensors import safe_open

        with safe_open("model.bintensors", framework="pt", device=0) as f:
            tensor = f.get_tensor("embedding")

        ```
        """
        pass
    def keys(self):
        """
        Returns the names of the tensors in the file.

        Returns:
            (`List[str]`):
                The name of the tensors contained in that file
        """
        pass
    def metadata(self):
        """
        Return the special non tensor information in the header

        Returns:
            (`Dict[str, str]`):
                The freeform metadata.
        """
        pass

class BintensorError(Exception):
    """
    Custom Python Exception for Safetensor errors.
    """
