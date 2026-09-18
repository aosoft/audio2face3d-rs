use super::SecretApiKey;
use tonic::{Status, metadata::MetadataMap};
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InvalidApiKey;
impl std::fmt::Display for InvalidApiKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("invalid API key format")
    }
}
impl std::error::Error for InvalidApiKey {}
/// RFC 6750 b64token syntax, without decoding or normalizing.
pub fn validate_api_key(key: &str) -> Result<(), InvalidApiKey> {
    if key.is_empty() || key.len() > 4096 {
        return Err(InvalidApiKey);
    }
    let mut padding = false;
    let mut body = false;
    for c in key.bytes() {
        if c == b'=' {
            padding = true;
        } else if !padding && (c.is_ascii_alphanumeric() || b"-._~+/".contains(&c)) {
            body = true;
        } else {
            return Err(InvalidApiKey);
        }
    }
    if body { Ok(()) } else { Err(InvalidApiKey) }
}
pub(super) fn parse(metadata: &MetadataMap) -> Result<SecretApiKey<'_>, Status> {
    let invalid = || Status::unauthenticated("missing or invalid bearer credential");
    let values = metadata.get_all("authorization");
    let mut values = values.iter();
    let value = values.next().ok_or_else(invalid)?;
    if values.next().is_some() {
        return Err(invalid());
    }
    let text = value.to_str().map_err(|_| invalid())?;
    let (scheme, rest) = text.split_once(' ').ok_or_else(invalid)?;
    if !scheme.eq_ignore_ascii_case("Bearer") {
        return Err(invalid());
    }
    let key = rest.trim_start_matches(' ');
    validate_api_key(key).map_err(|_| invalid())?;
    Ok(SecretApiKey(key))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn strict_bearer_and_key_grammar() {
        for key in ["abc", "nvapi-test", "a-._~+/", "a=="] {
            assert!(validate_api_key(key).is_ok());
        }
        for key in ["", "=", "=abc", "ab=c", " a", "a ", "a\tb", "a,b", "日本"] {
            assert!(validate_api_key(key).is_err());
        }
        assert!(validate_api_key(&"a".repeat(4096)).is_ok());
        assert!(validate_api_key(&"a".repeat(4097)).is_err());
        for value in ["Bearer abc", "bEaReR   abc"] {
            let mut m = MetadataMap::new();
            m.insert("authorization", value.parse().unwrap());
            assert_eq!(parse(&m).unwrap().expose(), "abc");
        }
        for value in [
            "Bearer ",
            " Bearer abc",
            "Basic abc",
            "Bearer abc ",
            "Bearer a,b",
            "Bearer a=b",
        ] {
            let mut m = MetadataMap::new();
            m.insert("authorization", value.parse().unwrap());
            assert!(parse(&m).is_err());
        }
        let mut m = MetadataMap::new();
        m.append("authorization", "Bearer a".parse().unwrap());
        m.append("authorization", "Bearer b".parse().unwrap());
        assert!(parse(&m).is_err());
        assert_eq!(format!("{:?}", SecretApiKey("secret")), "[REDACTED]");
    }
}
