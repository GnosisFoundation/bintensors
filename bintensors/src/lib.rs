#![deny(missing_docs)]
#![doc(
    html_logo_url = "https://github.com/GnosisFoundation/bintensors/blob/master/.github/assets/bintensors-logo.png?raw=true"
)]
#![doc = include_str!("../DOC_README.md")]
#![cfg_attr(not(feature = "std"), no_std)]
pub mod slice;
pub mod tensor;
/// serialize_to_file only valid in std
#[cfg(feature = "std")]
pub use tensor::serialize_to_file;
pub use tensor::{BinTensorError, BinTensors, Dtype, View, serialize, serialize_with_checksum};

// TODO: uncomment when all of no_std is ready
// #[cfg(feature = "alloc")]
// #[macro_use]
// extern crate alloc;

#[cfg(not(feature = "std"))]
compile_error!("must have the `std` feature");

/// A facade around all the types we need from the `std`, `core`, and `alloc`
/// crates. This avoids elaborate import wrangling having to happen in every
/// module.
mod lib {
    // TODO: uncomment when add non-std
    // #[cfg(not(feature = "std"))]
    // mod no_stds {
    //     pub use alloc::borrow::Cow;
    //     pub use alloc::string::{String, ToString};
    //     pub use alloc::vec::Vec;
    //     pub use hashbrown::HashMap;
    // }
    #[cfg(feature = "std")]
    mod stds {
        pub use std::borrow::Cow;
        pub use std::collections::HashMap;
        pub use std::string::{String, ToString};
        pub use std::vec::Vec;
    }

    /// choose std or no_std to export by feature flag
    // TODO: uncomment when add non-std
    // #[cfg(not(feature = "std"))]
    // pub use no_stds::*;
    #[cfg(feature = "std")]
    pub use stds::*;
}
