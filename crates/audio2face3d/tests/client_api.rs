//! Compile the public session surface without access to private fake/driver types.
use audio2face3d::client::{
    Client, Control, Limits,
    types::{InputChunk, PcmBuffer, RequestOptions, Result},
};
fn require_send<T: Send>(_: T) {}
fn application(client: &Client) -> Result<()> {
    let (mut input, mut output, control) = client.start(RequestOptions::default())?.split();
    require_send(input.send(InputChunk::new(PcmBuffer::from_vec(vec![0, 0])?, vec![])));
    require_send(output.recv());
    require_send(control.closed());
    require_send(input.finish());
    control.cancel();
    require_send(client.shutdown());
    Ok(())
}
#[test]
fn public_surface_compiles_without_an_adapter_or_async_runtime() {
    fn shared<T: Send + Sync + Clone>() {}
    shared::<Client>();
    shared::<Control>();
    Limits::default().validate().unwrap();
    #[cfg(any(feature = "mock", feature = "native"))]
    require_send(Client::direct(Default::default()));
    #[cfg(feature = "client-grpc")]
    require_send(Client::server(audio2face3d::client::ServerConfig::new(
        "http://127.0.0.1:1",
    )));
    let _application: fn(&Client) -> Result<()> = application;
}
