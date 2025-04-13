# Standard BinTensors Specification

<!-- > [!NOTE]
>This is a pre-release specification detailing the serialization format used within `BinTensors`, within that format we use default data types serialization methods, which you can find in `bincode` [specification](https://github.com/bincode-org/bincode/blob/trunk/docs/spec.md). As we refine the format, changes may be introduced to improve performance and add new features. While we aim to ensure backward compatibility, adjustments to previous file versions may be necessary. -->

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
  <picture>
    <img alt="bintensors" src="https://raw.githubusercontent.com/GnosisFoundation/bintensors/refs/heads/master/.github/assets/endian.png" style="max-width: 100%;">
  </picture>
  <br/>

  <sub>
     <a href="https://en.wikipedia.org/wiki/Endianness">Wikipedia</a> diagram demonstrating big-endianness versus little-endianness
  </sub>
  <br/>
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

Here’s a refined and clearer version of your section with an emphasis on the deterministic ordering of `BTreeMap`:

---

## HashMap vs BTreeMap

When working with map-like data structures such as `HashMap<K, V>` or `BTreeMap<K, V>`, their serialization behavior is similar in structure. Like a `Vec<T>`, the encoder first writes the number of key-value pairs, followed by each key and its associated value in sequence.

However, an important distinction lies in ordering:  
- `HashMap` does **not** guarantee the order of its entries.
- `BTreeMap`, by contrast, maintains a **deterministic, sorted order** based on the keys.

This deterministic ordering makes `BTreeMap` particularly useful when consistent serialization output is required, such as in checksumming, binary diffing, or reproducible builds.

Here's an example of how a simple `HashMap<String, String>` would be serialized using `bincode`:

```rust
use std::collections::HashMap;

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

> ⚠️ Note: The actual key-value order in the encoded output may vary between runs if using `HashMap`. If reproducibility is important, prefer `BTreeMap`.

## Assembling the Format

<p align="center">
  <picture>
    <img alt="bintensors" src="https://github.com/GnosisFoundation/bintensors/blob/master/.github/assets/bt-format.png" style="max-width: 100%;">
  </picture>
  <br/>

  <sub>
    Visual representation of bintensors (bt) file format
  </sub>
  <br/>
</p>


Now that we've outlined the core serialization structures used in the `BinTensors` file format, let's examine how to properly interpret and reconstruct an actual binary buffer. The majority of the serialization complexity is concentrated in the **metadata section**.

Consider the following binary buffer:

```rust
b"\x18\x00\x00\x00\x00\x00\x00\x00\x00\x01\x08weight_1\x00\x02\x02\x02\x00\x04       \x00\x00\x00\x00";
```

---

### 📦 Size Of Metadata Header (SofMH)

The first 8 bytes represent the **Size of Metadata (SofMH)**, which tells us how many bytes to read for the metadata section:

```rust
// Extract the Size of Metadata (SofMH) from the first 8 bytes
let sofm = usize::from_le_bytes(buffer[..8].try_into().expect("expected 8 bytes"));
// This yields 24 bytes as the header size
```

---

### 🧠 Header Reconstruction

After the SofM header field, we encounter the metadata — a serialized structure that contains a combination of optional user metadata and a set of tensor descriptors. This metadata is stored in a compact format composed of a `BTreeMap<String, String>` and a `Vec<(String, TensorInfo)>` over just encoding, and decoding of `Metadata`.

Notably, the use of `BTreeMap` instead of `HashMap` ensures that the entries maintain a consistent and deterministic order across serialization and deserialization, which is essential for reproducibility and avoids the non-deterministic layout of a standard HashMap. This layout allows us to store a fixed state efficiently and predictably.

For reference, here is the structure we're reconstructing:

```rust
pub struct TensorInfo {
    pub dtype: Dtype,                   // Data type of the tensor
    pub shape: Vec<usize>,             // Tensor dimensions
    pub data_offsets: (usize, usize),  // Start and end offsets in the buffer
}

pub struct Metadata {
    metadata: Option<HashMap<String, String>>, // Optional user-defined key-value metadata
    tensors: Vec<TensorInfo>,                  // List of tensor metadata entries
    index_map: HashMap<String, usize>,         // Maps tensor names to indices in `tensors`
}

```

---

### 🔍 Byte-by-Byte Analysis

We'll analyze just the **metadata portion**, excluding the SofMH and tensor buffer:

```rust
let metadata_bytes = b"\x00\x01\x08weight_1\x00\x02\x02\x02\x00\x04       ";
```

#### Breakdown:

| Byte(s)        | Meaning                                                                 |
|----------------|-------------------------------------------------------------------------|
| `\x00`         | Discriminant for `None` (`Option<BTreeMap<...>>`)                       |
| `\x01`         | Length of the vector of tensor entries (1 tensor)                       |
| `\x08`         | Length of the key string `"weight_1"`                                   |
| `weight_1`     | UTF-8 string name of the tensor                                         |
| `\x00`         | Discriminant for `Dtype::Bool`                                          |
| `\x02`         | Length of the shape vector (2 dimensions)                               |
| `\x02\x02`     | Shape dimensions `[2, 2]`                                               |
| `\x00\x04`     | Data offsets: `(0, 4)`                                                  |
| `\x20...`      | Padding (spaces `0x20`) to align the metadata to 8-byte boundary        |

---

### ✅ Final Decoded Metadata

```rust
Metadata {
    metadata: None,
    tensors: [
        TensorInfo {
            dtype: BOOL,
            shape: [2, 2],
            data_offsets: (0, 4),
        }
    ],
    index_map: {
        "weight_1": 0,
    }
}
```

---

### 📊 Tensor Buffer

The remaining bytes in the buffer represent the actual tensor data:

```rust
b"\x00\x00\x00\x00"
```

From the metadata, we know:

1. `dtype` is `BOOL` (likely 1 byte per value)
2. Shape is `[2, 2]` = 4 total elements
3. Data spans offset `(0, 4)`, so 4 bytes total

These 4 bytes appear to be zeroes, representing a 2×2 boolean tensor initialized to `false`.
