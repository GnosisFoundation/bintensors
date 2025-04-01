# Open Call to Break  

Before integrating this file format into a production-level system, it is crucial to rigorously test its security and resilience. As a single individual, my ability to uncover all potential vulnerabilities is inherently limited. Therefore, I encourage you to analyze and attempt to break the format, assuming you can.  

## Where to Begin  

The first step is understanding the underlying encoding used within this file format. I recommend starting with the [Bintensor specifications](https://github.com/GnosisFoundation/bintensors/blob/master/specs/encoding.md) and the [Bincode specifications](https://github.com/bincode-org/bincode/blob/trunk/docs/spec.md).  

If you are already familiar with how `bincode` serializes Rust’s data structures into raw bytes, you may skip this section and proceed directly to my current approaches or the latest findings from previous attack attempts. However, if you are not yet comfortable with these encoding methods, I strongly encourage you to review the binary layout of each data type before proceeding.  
