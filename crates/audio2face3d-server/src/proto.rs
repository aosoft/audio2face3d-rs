//! Upstream wire types, generated without changing NVIDIA's schemas.
#![allow(
    clippy::all,
    rustdoc::broken_intra_doc_links,
    rustdoc::invalid_html_tags
)]
include!(concat!(env!("OUT_DIR"), "/ace.rs"));
pub use nvidia_ace::services::a2f_controller::v1::a2f_controller_service_server::{
    A2fControllerService, A2fControllerServiceServer,
};
pub use nvidia_ace::{
    a2f::v1 as a2f, animation_data::v1 as animation, audio::v1 as audio,
    controller::v1 as controller, status::v1 as status,
};
pub const DESCRIPTOR: &[u8] = tonic::include_file_descriptor_set!("ace_descriptor");
pub const SERVICE_NAME: &str = "nvidia_ace.services.a2f_controller.v1.A2FControllerService";
