# audio2face3d-types

Owned Rust data for audio-to-face inference, with no external dependencies or
asynchronous runtime requirement. This crate does not expose wire or GPU types.

`PcmBuffer::from_vec` transfers ownership; `copy_from_slice` copies borrowed
input. Curve frames share their ordered layout through `Arc` and own their
weights. Times use integer nanoseconds, with exact audio sample positions kept
separately. Configuration uses `Option` to preserve absent values and explicit
zero/false, and validates before use.

This is the common data layer. Inference and transport are provided separately.
