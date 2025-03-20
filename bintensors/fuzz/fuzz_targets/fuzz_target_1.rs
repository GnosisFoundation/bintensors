#![no_main]

use libfuzzer_sys::fuzz_target;
use bintensors::tensor::BinTensors;

fuzz_target!(|data: &[u8]| {
    let _ = BinTensors::deserialize(data);
});
