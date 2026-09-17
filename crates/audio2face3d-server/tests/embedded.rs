#![cfg(feature = "mock")]
use audio2face3d_server::{
    Server,
    config::{BackendKind, Config},
};
#[tokio::test]
async fn embedded_server_uses_caller_listener_and_stop() {
    let config = Config {
        backend: BackendKind::Mock,
        ..Config::default()
    };
    let server = Server::builder(config).build().unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let (stop, stopped) = tokio::sync::oneshot::channel();
    let serving = tokio::spawn(server.serve(listener, async {
        let _ = stopped.await;
    }));
    let channel = tonic::transport::Endpoint::from_shared(format!("http://{address}"))
        .unwrap()
        .connect()
        .await
        .unwrap();
    let mut health = tonic_health::pb::health_client::HealthClient::new(channel);
    let reply = health
        .check(tonic_health::pb::HealthCheckRequest {
            service: String::new(),
        })
        .await
        .unwrap();
    assert_eq!(reply.into_inner().status, 1);
    stop.send(()).unwrap();
    serving.await.unwrap().unwrap();
}
