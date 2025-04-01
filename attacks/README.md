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
