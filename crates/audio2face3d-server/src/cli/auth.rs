use audio2face3d_server::auth::{
    AuthError, AuthRequest, AuthResult, Authenticator, Principal, validate_api_key,
};
use std::ffi::OsString;
use subtle::ConstantTimeEq;

/// The concrete single-key policy belongs to this executable.
pub(super) fn resolve(cli: Option<OsString>) -> Result<Option<impl Authenticator>, &'static str> {
    resolve_with(cli, || std::env::var_os("AUDIO2FACE3D_API_KEY"))
}
fn resolve_with(
    cli: Option<OsString>,
    environment: impl FnOnce() -> Option<OsString>,
) -> Result<Option<impl Authenticator>, &'static str> {
    let selected = cli.or_else(environment);
    let selected = selected
        .map(|value| {
            let key = value.into_string().map_err(|_| "API key must be Unicode")?;
            validate_api_key(&key).map_err(|_| "invalid API key format")?;
            Ok::<_, &'static str>(key)
        })
        .transpose()?;
    Ok(selected.map(|expected| {
        move |request: AuthRequest<'_>| -> AuthResult {
            if bool::from(
                expected
                    .as_bytes()
                    .ct_eq(request.api_key.expose().as_bytes()),
            ) {
                Principal::new("default")
            } else {
                Err(AuthError::InvalidCredential)
            }
        }
    }))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn precedence_missing_and_invalid_values() {
        assert!(resolve_with(None, || None).unwrap().is_none());
        assert!(
            resolve_with(Some("valid".into()), || panic!(
                "environment must not be read"
            ))
            .unwrap()
            .is_some()
        );
        assert!(
            resolve_with(None, || Some("valid".into()))
                .unwrap()
                .is_some()
        );
        for key in ["", " bad", "bad ", "bad,key", "bad=key"] {
            assert!(resolve_with(Some(key.into()), || Some("valid".into())).is_err());
            assert!(resolve_with(None, || Some(key.into())).is_err());
        }
        assert!(resolve_with(Some("x".repeat(4097).into()), || None).is_err());
        #[cfg(windows)]
        {
            use std::os::windows::ffi::OsStringExt;
            assert!(resolve_with(None, || Some(OsString::from_wide(&[0xd800]))).is_err());
        }
    }
}
