#![cfg_attr(
    all(target_os = "windows", not(debug_assertions)),
    windows_subsystem = "windows"
)]
use clap::Parser;
fn main() -> std::process::ExitCode {
    let console = attach_parent_console();
    let conversion_command = std::env::args_os()
        .nth(1)
        .is_some_and(|arg| arg == "obj2morph" || arg == "model");
    let args = match audio2face3d_gui::startup::Args::try_parse() {
        Ok(args) => args,
        Err(error) => {
            let _ = error.print();
            if !console && !conversion_command {
                show_message(&error.to_string());
            }
            return std::process::ExitCode::from(error.exit_code() as u8);
        }
    };
    #[cfg(any(feature = "obj2morph", feature = "model-management"))]
    if args.command.is_some()
        && let Err(error) = args.validate_subcommand()
    {
        eprintln!("{error}");
        return std::process::ExitCode::from(2);
    }
    #[cfg(any(feature = "obj2morph", feature = "model-management"))]
    let mut args = args;
    #[cfg(any(feature = "obj2morph", feature = "model-management"))]
    if let Some(command) = args.command.take() {
        match command {
            #[cfg(feature = "model-management")]
            audio2face3d_gui::startup::Command::Model(input) => {
                let result = (|| -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
                    let context = audio2face3d::Audio2Face3DContext::builder()
                        .native_runtime(args.platform.resolve()?)
                        .build();
                    audio2face3d::logging::integration::LogScope::new(context)
                        .in_scope(|| input.run())
                })();
                return match result {
                    Ok(()) => std::process::ExitCode::SUCCESS,
                    Err(error) => {
                        eprintln!("{error}");
                        std::process::ExitCode::FAILURE
                    }
                };
            }
            #[cfg(feature = "obj2morph")]
            audio2face3d_gui::startup::Command::Obj2morph(input) => {
                return match audio2face3d_gui::obj2morph::cli::run(input) {
                    Ok(()) => std::process::ExitCode::SUCCESS,
                    Err(error) => {
                        eprintln!("{error}");
                        std::process::ExitCode::from(error.exit_code() as u8)
                    }
                };
            }
        }
    }
    let options = match args.resolve() {
        Ok(options) => options,
        Err(error) => {
            eprintln!("{error}");
            if !console {
                show_message(&error.to_string());
            }
            return std::process::ExitCode::FAILURE;
        }
    };
    match audio2face3d_gui::desktop::run_with_options(options) {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            if !console {
                show_message(&error.to_string());
            }
            std::process::ExitCode::FAILURE
        }
    }
}
fn show_message(message: &str) {
    rfd::MessageDialog::new()
        .set_title("Audio2Face-3D")
        .set_description(message)
        .show();
}
fn attach_parent_console() -> bool {
    #[cfg(target_os = "windows")]
    {
        use windows_sys::Win32::System::Console::{
            ATTACH_PARENT_PROCESS, AttachConsole, GetStdHandle, STD_OUTPUT_HANDLE,
        };
        // SAFETY: called once at process startup; no pointers or owned handles are passed.
        unsafe {
            let _ = AttachConsole(ATTACH_PARENT_PROCESS);
            let output = GetStdHandle(STD_OUTPUT_HANDLE);
            !output.is_null() && output as isize != -1
        }
    }
    #[cfg(not(target_os = "windows"))]
    {
        true
    }
}
