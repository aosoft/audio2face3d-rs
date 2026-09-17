#![cfg(feature = "native")]
use audio2face3d_server::{
    Server, ServerError,
    config::{BackendKind, Config},
};
/// Run with AUDIO2FACE3D_LOG_MODEL pointing to a prepared Regression descriptor.
#[tokio::test]
#[ignore = "requires native SDK, GPU and prepared Regression model"]
async fn stopping_during_native_prepare_keeps_cleanup_owned() {
    let model = std::env::var_os("AUDIO2FACE3D_LOG_MODEL").expect("prepared model");
    let server = Server::builder(Config {
        backend: BackendKind::Regression,
        model: Some(model.into()),
        shutdown_timeout_ms: 1,
        ..Default::default()
    })
    .build()
    .unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let result = server.serve(listener, std::future::ready(())).await;
    let report = match result {
        Ok(report) => report,
        Err(ServerError::ShutdownTimeout { completion, .. }) => completion.await.unwrap(),
        Err(error) => panic!("{error}"),
    };
    assert_eq!(report.inference_requests, 0);
    assert_eq!(report.inference_workers_started, 0);
}
