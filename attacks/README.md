# Open Call to Break  

Before integrating this file format into a production-level system, it is crucial to rigorously test its security and resilience. As a single individual, my ability to uncover all potential vulnerabilities is inherently limited. Therefore, I encourage you to analyze and attempt to break the format, assuming you can.  

## Where to Begin  

The first step is understanding the underlying encoding used within this file format. I recommend starting with the [Bintensor specifications](https://github.com/GnosisFoundation/bintensors/blob/master/specs/encoding.md) and the [Bincode specifications](https://github.com/bincode-org/bincode/blob/trunk/docs/spec.md).  

If you are already familiar with how `bincode` serializes Rust’s data structures into raw bytes, you may skip this section and proceed directly to my current approaches or the latest findings from previous attack attempts. However, if you are not yet comfortable with these encoding methods, I strongly encourage you to review the binary layout of each data type before proceeding.  

### Attempts

first 3 of attempts where applyied are from the `safetensors` repo on how they may attack there on file format.

### Attempt 1

We swap the size of the metadata header, in the hope of possibly exceeding the the size of the header, though this will be caught by the BinTensorError exception.

### Attempt 2

This attempt assumes the existence of a service that hosts such files. In an effort to delay loading similar to tactics used in DDoS attacks—attackers could create excessively large files, allocating a significant portion of memory to file metadata. This can be easily mitigated by the enforced upper memory limit for metadata encoding, the encoder will trigger an error, even if the attacker manages to bypass the MAX_HEADER_SIZE check.


### Attempt 3

One of my favourite proposed attacks against `SafeTensors` and this format involves overlapping offsets—a simple yet elegant exploit. The mitigation is straightforward: enforce strict validation of tensor offsets. Any attempt to exceed buffer boundaries or define overlapping regions should be rejected. The tensor buffer should function as a contiguous sequence over a subset set of $\{0, \dots N\}$, ensuring data integrity and preventing unintended memory access.

### Attempt 4 

> ⚠️ **Note:** The exploit is no longer applicable in the release version (Rust: `0.1.0`, Python: `0.1.0`) due to significant changes in the file layout compared to the pre-release versions (`0.0.1-alpha.3` / `0.0.5`).

This format vulnerability was unexpected and prompted a thorough review, resulting in [commit](https://github.com/GnosisFoundation/bintensors/commit/032826e369d301b49eb264090581e24198d3a4ed) to properly validate the issue. The root cause lies in tensor_info entries exceeding their index_map counterparts, leading to potential memory allocation mismatches. To mitigate this risk, we introduced a validation check: if the size of tensor_info exceeds that of index_map, the format is immediately deemed invalid.

This issue is specific to BinTensors and can only arise if the file has been manually altered.

### Attempt 5

> ⚠️ **Note:** The exploit is no longer applicable in the release version (Rust: `0.1.0`, Python: `0.1.0`) due to significant changes in the file layout compared to the pre-release versions (`0.0.1-alpha.3` / `0.0.5`).


I had a moment of clarity that led to identifying a critical flaw in the metadata encoding of the format. Specifically, I realized that an attacker could craft an oversized `index_map` to maliciously reference the **same tensor buffer repeatedly**, resulting in the deserialization of **tens or even hundreds of megabytes** from a single tensor entry — all without increasing the actual tensor data footprint.  

This vulnerability exposes the system to resource exhaustion attacks and bypasses intended memory boundaries.

### Proposed Mitigation

To resolve this issue, the internal `Metadata` structure was redesigned into a more organized and deterministic format, minimizing the risk of errors. Details of the new format can be found in the [specification](https://github.com/GnosisFoundation/bintensors/blob/master/specs/encoding.md#-header-reconstruction).
