import datetime
import os

from bintensors import BintensorError
from bintensors.torch import load_file, serialize

filename = "bintensors_abuse_attempt_2.bt"


def create_payload():
    """
    payload is too will be too large to deserilize mitigating simple DDos attack
    """
    shape = [2, 2]
    n = shape[0] * shape[1] * 4

    metadata = {
        f"weight_{i}": {"dtype": "float32", "shape": shape, "data_offsets": [n * i, (n * i) + n], "data": b"\0" * n}
        for i in range(1000 * 1000 * 10)
    }

    buffer = serialize(metadata)
    with open(filename, "wb") as f:
        f.write(buffer)


if not os.path.isfile(filename):
    create_payload()

print(f"The file {filename} is {os.path.getsize(filename) / 1000/ 1000} Mo")
start = datetime.datetime.now()
try:
    test = load_file(filename)
except BintensorError as e:
    os.remove(filename) 
    print(f"Throw an expected error: {e}")
# Header file exception should be thrown and should not reach this code
# This code below should not run...
print(f"Loading the file took {datetime.datetime.now() - start}")