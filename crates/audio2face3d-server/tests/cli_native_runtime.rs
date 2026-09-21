#![cfg(all(feature = "cli", feature = "native"))]
use audio2face3d::client::{
    Client, ServerConfig,
    types::{AudioFormat, InputChunk, OutputEvent, PcmBuffer, RequestOptions},
};
use std::{
    path::PathBuf,
    process::{Child, Command, Stdio},
    time::Duration,
};
struct Process(Child);
impl Drop for Process {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}
#[tokio::test]
#[ignore = "requires AUDIO2FACE3D_TEST_PLATFORM_CONFIG and A2F_MODEL"]
async fn native_cli_serves_with_config_file_and_no_sdk_environment() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../temp/native-cli-tests");
    std::fs::create_dir_all(&root).unwrap();
    let log =
        std::fs::File::create(root.join(format!("server-cli-{}.log", std::process::id()))).unwrap();
    let reservation = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let address = reservation.local_addr().unwrap();
    drop(reservation);
    let mut command = Command::new(env!("CARGO_BIN_EXE_audio2face3d-server"));
    command
        .args(["--platform-config"])
        .arg(std::env::var_os("AUDIO2FACE3D_TEST_PLATFORM_CONFIG").unwrap())
        .arg("--model")
        .arg(std::env::var_os("A2F_MODEL").unwrap())
        .args(["--listen", &address.to_string()]);
    for (name, _) in std::env::vars_os() {
        let text = name.to_string_lossy().to_ascii_uppercase();
        if text.starts_with("CUDA")
            || text.starts_with("TENSORRT")
            || text.starts_with("CUDARC")
            || text == "TRTEXEC"
            || text == "AUDIO2FACE3D_API_KEY"
        {
            command.env_remove(name);
        }
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        let windows = std::env::var_os("SystemRoot").unwrap();
        command
            .env("PATH", PathBuf::from(windows).join("System32"))
            .creation_flags(0x08000000);
    }
    #[cfg(unix)]
    command.env_remove("LD_LIBRARY_PATH");
    command
        .stdout(Stdio::from(log.try_clone().unwrap()))
        .stderr(Stdio::from(log));
    let mut process = Process(command.spawn().unwrap());
    std::fs::write(
        root.join(format!("server-cli-{}.manifest", std::process::id())),
        format!(
            "pid={}\naddress={address}\ncommand={command:?}\n",
            process.0.id()
        ),
    )
    .unwrap();
    let client = tokio::time::timeout(Duration::from_secs(60), async {
        loop {
            assert!(
                process.0.try_wait().unwrap().is_none(),
                "server exited before accepting gRPC"
            );
            let config = ServerConfig::builder(format!("http://{address}"))
                .connect_timeout(Duration::from_millis(250))
                .build()
                .unwrap();
            if let Ok(client) = Client::server(config).await {
                break client;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    })
    .await
    .expect("server prepare timed out");
    let (mut input, mut output, control) = client
        .start(
            RequestOptions::builder(AudioFormat::MONO_16KHZ)
                .build()
                .unwrap(),
        )
        .unwrap()
        .split();
    let sending = tokio::spawn(async move {
        input
            .send(InputChunk::new(
                PcmBuffer::from_vec(vec![0; 3200]).unwrap(),
                vec![],
            ))
            .await
            .unwrap();
        input.finish().await.unwrap();
    });
    let (mut curves, mut complete) = (0, false);
    tokio::time::timeout(Duration::from_secs(60), async {
        while let Some(event) = output.recv().await.unwrap() {
            match event {
                OutputEvent::Curves(_) => curves += 1,
                OutputEvent::Completed(_) => complete = true,
                _ => {}
            }
        }
    })
    .await
    .unwrap();
    sending.await.unwrap();
    control.closed().await.unwrap();
    client.shutdown().await.unwrap();
    assert!(curves > 0 && complete);
}
