//! Internal ACE wire bindings and transport-boundary conversions.
//!
//! Default features generate messages only. Enable `client` or `server` for
//! the corresponding gRPC bindings. Inference data lives in audio2face3d-types.
pub mod convert;
pub mod wire;
