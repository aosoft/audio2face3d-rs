//! Unmodified upstream schemas. Generated bindings are not public domain types.
#![allow(
    clippy::all,
    rustdoc::broken_intra_doc_links,
    rustdoc::invalid_html_tags
)]
include!(concat!(env!("OUT_DIR"), "/ace.rs"));
#[cfg(feature = "grpc-server")]
pub use nvidia_ace::services::a2f_controller::v1::a2f_controller_service_server::{
    A2fControllerService, A2fControllerServiceServer,
};
pub use nvidia_ace::{
    a2f::v1 as a2f, animation_data::v1 as animation, audio::v1 as audio,
    controller::v1 as controller, status::v1 as status,
};
pub const DESCRIPTOR: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/ace_descriptor.bin"));
pub const SERVICE_NAME: &str = "nvidia_ace.services.a2f_controller.v1.A2FControllerService";
