//! Module Containing verification structure
use sha1::{Digest, Sha1};

#[cfg(feature = "std")]
use std::{io::Write, io, fmt};

#[cfg(not(feature = "std"))]
use core::{io::Write, io, fmt};


#[allow(unused_macros)]
/// Converts a byte slice to a hexadecimal string representation.
///
/// # Arguments
/// 
/// * `$buf` - A byte slice (`&[u8]`) to be converted to a hexadecimal string.
///
/// # Returns
/// A `String` containing the hexadecimal representation of the input byte slice.
macro_rules! bytes_to_hex {
    ($buf:expr) => {{
        let bytes: &[u8] = $buf.as_ref();
        bytes.iter().map(|b| format!("{:02x}", b)).collect::<String>()
    }};
}

/// The size of the digest in bytes.
/// 
/// This constant represents the size of a cryptographic hash (e.g., SHA-1), which is 20 bytes.
pub const DIGEST_SIZE: usize = 20;

/// A flexible enum structure for possible later implementation of other
/// hashes such as SHA-256, to give associated id to a tensor
#[derive(Debug)]
pub enum ObjectId {
    /// hash function that digest bytes and produces 160 bit (20 bytes)
    SHA1([u8; DIGEST_SIZE])
}


impl ObjectId {
    /// Give sa byte-buffer sha1 digest the tensors
    /// and produces a [`ObjectId::SHA1`].
    pub fn sha1(buf : impl AsRef<[u8]>) -> ObjectId {
        let result = Sha1::digest(buf);
        Self::SHA1(result[..].try_into().expect("SHA1 output must be 20 bytes"))
    }
}

impl ObjectId {

    /// Write hash id into writer trait.
    pub fn write<W : Write>(&self, writer : &mut W) -> io::Result<()> {
        write!(writer, "{}", self)
    }

    /// Segments the hashed id into prefix, and suffix.
    pub fn segment(&self) -> (&[u8], &[u8]) {
        match self {
            Self::SHA1(bytes) => (&bytes[0..2], &bytes[2..])
        }
    }

}

/// Converts a discrete slice [u8; DIGEST_SIZE] value into `ObjectId`
impl From<[u8; DIGEST_SIZE]> for ObjectId {
    fn from(value: [u8; DIGEST_SIZE]) -> Self {
        ObjectId::SHA1(value)
    }
}

/// Converts a discrete slice &str value into `ObjectId`
impl From<&str> for ObjectId {
    fn from(value: &str) -> Self {
        ObjectId::sha1(value.as_bytes())
    }
}

/// Converts a discrete slice &[u8] value into `ObjectId`
impl From<&[u8]> for ObjectId {
    fn from(value: &[u8]) -> Self {
        ObjectId::sha1(value)
    }
}

// TODO: uncomment if needed
// /// Converts a discrete slice TensorView<'_> value into `ObjectId`
// impl From<TensorView<'_>> for ObjectId {
//     fn from(value: TensorView) -> Self {
//         ObjectId::sha1(value.data())
//     }
// }

impl From<ObjectId> for Vec<u8> {
    fn from(value: ObjectId) -> Self {
        match value {
            ObjectId::SHA1(bytes) => bytes.to_vec()
        }
    }
}

/// Implemnts the `Display` trait for `ObjectId`
impl fmt::Display for ObjectId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SHA1(b) => write!(f, "{}", bytes_to_hex!(b))
        }
    }
}


#[cfg(test)]
mod test {
    use crate::oid::ObjectId;

    #[cfg(feature = "std")]
    use std::io::Cursor;

    #[cfg(not(feature = "std"))]
    use core::io::Cursor;

    #[test]
    fn test_sh1_haser_basic() {
        let hash = ObjectId::sha1(b"hello world");
        assert_eq!(hash.to_string(), "2aae6c35c94fcfb415dbe95f408b9ce91ee846ed")
    }

    #[test]
    fn test_sha1_hasher_empty() {
        let hash = ObjectId::sha1(b"");
        assert_eq!(hash.to_string(), "da39a3ee5e6b4b0d3255bfef95601890afd80709")
    }

    #[test]
    fn test_sha1_hasher_unicode() {
        let hash = ObjectId::sha1("你好，世界".as_bytes());
        assert_eq!(hash.to_string(), "3becb03b015ed48050611c8d7afe4b88f70d5a20"); // SHA1 of "你好，世界"
    }

    #[test]
    fn test_sha1_write() {
        let hash = ObjectId::sha1(b"test data");
        let mut buffer = Cursor::new(Vec::new());

        hash.write(&mut buffer).unwrap();
        let result = String::from_utf8(buffer.into_inner()).unwrap();

        assert_eq!(result, hash.to_string());
    }
    
}