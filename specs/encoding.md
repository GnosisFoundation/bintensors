# Standard BinTensors Specification

NOTE: This is a pre-release specification detailing the serialization format used within `BinTensors`, within that format we use default data types serialization methods, which you can find in `bincode` [specification](https://github.com/bincode-org/bincode/blob/trunk/docs/spec.md).

As we refine the format, changes may be introduced to improve performance and add new features. While we aim to ensure backward compatibility, adjustments to previous file versions may be necessary.

## Preface

This document serves as an introduction to the serialization techniques used within BinTensors. If you are already familiar with Bincode’s encoding [specification](https://github.com/bincode-org/bincode/blob/trunk/docs/spec.md), you may skip ahead to the relevant definitions and encoding details.


## Definitions

<!-- bincode reused definitions -->

- **Variant**: A specific constructor or case of an enum type.
- **Variant Payload**: The associated data of a specific enum variant.
- **Discriminant**: A unique identifier for an enum variant, typically represented as an integer.

## Endianness

By default the seralization of the data types used are little-edian byte order. Meaning the multi bytes data structues and types are encoded with there least significant byte first.

<p align="center">
<div style="background-color:white; padding:8px">

![](https://upload.wikimedia.org/wikipedia/commons/thumb/5/5d/32bit-Endianess.svg/880px-32bit-Endianess.svg.png)

</div>
<span> <a src="https://en.wikipedia.org/wiki/Endianness">Wikipedia</a> diagram demonstrating big-endianness versus little-endianness</span>
</p>

## IntEncoding

Within `BinTensors` we are using `Varint Encoding` aside from `Int Encoding`.

### Varint Encoding

Varint Encoding is a specific method for encoding unsigned and signed integers. It uses a predefined set of discriminants $\{∅, 251, 252, 253\}$ to differentiate between byte representations.

```rust
// unsigned 8 bit integer.
let data = u8::MAX;
let encoded =  bincode::encode_to_vec(&data, bincode::config::standard()).unwrap();
assert_eq!(encoded, &[
    0xFF,
]);

// unsigned 16 bit integer.
let data = u16::MAX;
let encoded =  bincode::encode_to_vec(&data, bincode::config::standard()).unwrap();
assert_eq!(encoded, &[
    251, 0xFF, 0xFF
]);


// unsigned 32 bit integer.
let data = u32::MAX;
let encoded =  bincode::encode_to_vec(&data, bincode::config::standard()).unwrap();
assert_eq!(encoded, &[
    252, 0xFF, 0xFF, 0xFF, 0xFF
]);

// unsigned 64 bit integer.
let data = u64::MAX;
let encoded =  bincode::encode_to_vec(&data, bincode::config::standard()).unwrap();
assert_eq!(encoded, &[
    253, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF
]);

// unsigned 64 bit integer.
let data = usize::MAX;
let encoded =  bincode::encode_to_vec(&data, bincode::config::standard()).unwrap();
assert_eq!(encoded, &[
    253, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF
]);
// We don't handle signed integers but if you would like how to know
// there encoded.
// https://docs.rs/bincode/2.0.1/bincode/config/struct.Configuration.html#method.with_variable_int_encoding
```

## Tuple Encoding

Tuples, such as those used within the `TensorInfo` data structure (e.g., for offset counters), are serialized in first-to-last order without any additional metadata.

- No length prefix is required.
- Fields are encoded sequentually
- order of serialization is deterministic and matches the tuple's declaration order.

```rust
// Simple example of tuple encoding
let data_offset = (0, 3000);
let encoded = bincode::encode_to_vec(&data_offset, bincode::config::standard()).unwrap();
assert_eq!(encoded, &[
    0,             // First element (0)
    251, 0xB8, 0xB0 // Second element (3000) with Varint prefix
]);
```

## Vector

`Vec<T>` are encoded with a length prefix using `Int Encoding`, followed by the serialized elements.

```rust
let data: Vec<String> = vec!["hello".into(), "world".into()];
let encoded = bincode::encode_to_vec(&data, bincode::config::standard()).unwrap();
assert_eq!(
    encoded,
    &[
        2, 5, b'h', b'e', b'l', b'l', b'o', 5, b'w', b'o', b'r', b'l', b'd'
    ]
);
```

## String and `&str` Encoding

In `bincode`, strings are encoded similarly to `Vec<T>`, which aligns with their representation in Rust’s standard library.

### Encoding Characteristics

- **No null terminator is added.**
- **No Byte Order Mark (BOM) is written.**
- **Unicode non-characters are preserved.**
- **The length is encoded first** using the configured `Int Encoding`.
- **Raw UTF-8 bytes** follow the length.
- **Supports the full range of valid UTF-8 sequences.**
- **Code points like U+0000** can appear freely within the string.

### Unicode Handling

- **Serialization:** Strings are encoded as a sequence of raw UTF-8 bytes.
- **No normalization or transformation** of text is performed.
- **Decoding:** If an invalid UTF-8 sequence is encountered, a `DecodeError::Utf8` is raised.

### Example

```rust
// UTF-8 encoding
let data = "hello world";
let encoded = bincode::encode_to_vec(data, bincode::config::standard()).unwrap();
assert_eq!(
    encoded,
    &[11, 104, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100] // 11-byte length prefix + UTF-8 bytes
);

// This will return an error because "hello world 🌎" contains non-UTF-8 sequences in bincode's context
let data = "hello world 🌎";
let encoded = bincode::encode_to_vec(data, bincode::config::standard());
assert!(encoded.is_err()); // Invalid UTF-8 sequences cause DecodeError::Utf8
```

## Enums

Enums are broken down into **variants**, which follow an intuitive encoding structure. To differentiate between different enum types, we allocate space for a **discriminant**, which acts as an identifier for each variant.

This approach is similar to `Varint Encoding` for `unsigned` and `signed` integers, where a predefined set of discriminants `{∅, 251, 252, 253}` is used. However, the discriminant for enums does not follow the same allocation pattern.

Since `Option<T>` and `Dtype` are the only enums being deserialized in our file format, we focus on their encoding. Enum variants may also include an **optional variant payload**, depending on their definition.

### Option<T>

`Option<T>` is always serialized using a `Variant Encoding` for the **discriminant**. If the variant contains data like `Option<T>` or a `Struct`, the serialized **variant payload** follows the variant **discriminant**.

```rust
// Encoding None (no payload needed)
let data: Option<u8> = None;
let encoded = bincode::encode_to_vec(data, bincode::config::standard()).unwrap();
assert_eq!(&encoded, &[0]); // Only the discriminant byte

// Encoding Some(42)
let data = Some(42);
let encoded = bincode::encode_to_vec(data, bincode::config::standard()).unwrap();
assert_eq!(&encoded, &[1, 42]); // Discriminant (1) + payload (42)
```

### Dtype

```rust
let data = Dtype::BOOL;
let encoded = bincode::encode_to_vec(data, bincode::config::standard()).unwrap();
assert_eq!(&encoded, &[0]); // Discriminant (1)
// rest enums value follow..
```

## HashMap

If you're familiar with the `HashMap<K, V>` data structure, its serialization should be relatively straightforward. Like `Vec<T>`, we first encode the number of entries in the map. Then, each key-value pair is serialized sequentially.

```rust
let mut data: HashMap<String, String> = HashMap::new();
data.insert("hello".to_string(), "world".to_string());
data.insert("hello1".to_string(), "world".to_string());

let encoded = bincode::encode_to_vec(data, bincode::config::standard()).unwrap();
assert_eq!(
    &encoded,
    &[
        // Number of key-value pairs
        2,
        // Key: "hello"
        5, b'h', b'e', b'l', b'l', b'o',
        // Value: "world"
        5, b'w', b'o', b'r', b'l', b'd',
        // Key: "hello1"
        6, b'h', b'e', b'l', b'l', b'o', b'1',
        // Value: "world"
        5, b'w', b'o', b'r', b'l', b'd'
    ]
);
```


## Assembling the Format

```
┌────────────────┬────────────┬───────────────────────┬──────────────────────┐
│ SofM (8 bytes) │  Metadata  |  padding : min(n, 8)  │ Tensor N Chunks...   │
└────────────────────────────────────────────────────────────────────────────┘
```

Now that we've outlined the core serialization structures used in the `BinTensors` file format, let's examine how to properly interpret and reconstruct an actual binary buffer. The majority of serialization complexity is concentrated in the metadata portion.

Consider the following binary buffer:

```rust
b"\x10\x00\x00\x00\x00\x00\x00\x00\x00\x01\x09\x02\x01\x04\x00\x10\x01\x04\x74\x65\x73\x74\x00\x20\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0";
```

### Size Of Metadata (SofM)

The first 8 bytes represent the Size of Metadata (SofM), which indicates how many bytes to read for the metadata section:

```rust
// Extract the Size of Metadata (SofM) from the first 8 bytes
let sofm = usize::from_le_bytes(buffer[..8].try_into().expect("expected 8 bytes"));
// This yields 16 bytes as the metadata size
```
### Metadata Reconstruction

After the SofM header, we encounter the actual metadata structure. For reference, here's the structure we're reconstructing:

```rust
pub struct TensorInfo {
    /// The type of each element of the tensor
    pub dtype: Dtype,
    /// The shape of the tensor
    pub shape: Vec<usize>,
    /// The offsets to find the data within the byte-buffer array
    pub data_offsets: (usize, usize),
}

pub struct Metadata {
    metadata: Option<HashMap<String, String>>,
    tensors: Vec<TensorInfo>,
    index_map: HashMap<String, usize>,
}
```

Let's examine the metadata bytes in detail:

```rust
// This represents only the metadata portion, excluding the SofM header and tensor buffer
// [0, 1, 9, 2, 1, 4, 0, 16, 1, 4, 116, 101, 115, 116, 0, 32]
let metadata_bytes = b"\x00\x01\x09\x02\x01\x04\x00\x10\x01\x04\x74\x65\x73\x74\x00\x20";
```

#### Byte-by-Byte Breakdown

Let's analyze the metadata byte sequence step-by-step:

1. `\x00` - Discriminant for `Option<HashMap<String, String>>` (the value `None`) if this where Not None then we would follow Discriminant `1` with the **variant payload** `HashMap<String, String>` serialization. 
2. `\x01` - Length of the `Vec<TensorInfo>` (1 element)
3. `\x09` - Discriminant for enum `Dtype` (representing `Dtype::I32`)
4. `\x02` - Length of shape `Vec<usize>` (2 elements)
5. `\x01\x04` - The shape values `[1, 4]`
6. `\x00\x10` - The tuple data_offsets `(0, 16)`
7. `\x01` - Length of `HashMap<String, usize>` (1 key-value pair)
8. `\x04` - Length of the tensor name `String`
9. `\x74\x65\x73\x74` - UTF-8 encoding of 'test' (the HashMap key)
10. `\x00` - Value associated with the key in the HashMap (index 0)
11. `\x20` - Padding byte to keep the metadata chunk divisible by 8.

This metadata structure contains a single I32 tensor with shape [1, 4], located at byte positions 0-16 in the tensor buffer. The tensor can be accessed using the "test" key in the index map, which points to position 0 in the tensors vector.

Result:
```
Metadata {
    metadata: None,
    tensors: [
        TensorInfo {
            dtype: I32,
            shape: [1, 4],
            data_offsets: (0, 16)
        }
    ],
    index_map: {
        "test": 0
    }
}
```


### Tensor Buffer

The remainder of the file represents the tensor data itself. According to the metadata, our tensor is located at the offset immediately following the padding:

```
\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0
```

This tensor buffer contains the actual values for the tensor identified as "test" in the index map. Based on the metadata, we know:

1. The data type is `I32` (32-bit integer)
2. The shape is [1, 4], indicating a 1×4 matrix or array
3. The data occupies positions 0-16 in this buffer

In this example, the tensor buffer appears to contain zeroes, which would represent a 1×4 array of 32-bit integers, all initialized to zero. Each 32-bit integer occupies 4 bytes in the buffer, and with 4 elements total (based on the shape [1, 4]), this accounts for the full 16 bytes indicated by the `data_offsets` field (0, 16).

The tensor data can be accessed and interpreted by:
1. Reading the appropriate number of bytes from the buffer
2. Converting those bytes to the correct data type (I32 in this case)
3. Reshaping the resulting values according to the tensor's shape

