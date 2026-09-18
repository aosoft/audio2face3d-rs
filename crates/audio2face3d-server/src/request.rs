//! Per-RPC state, independent of the shared application context.
use crate::auth::RequestId;
use std::{future::Future, time::Duration};
use tonic::{Status, metadata::MetadataMap};
#[derive(Clone, Copy)]
pub(crate) struct RequestContext {
    pub(crate) id: RequestId,
    pub(crate) deadline: Option<tokio::time::Instant>,
}
impl RequestContext {
    pub(crate) fn new(id: RequestId, metadata: &MetadataMap) -> Result<Self, Status> {
        let now = tokio::time::Instant::now();
        let values = metadata.get_all("grpc-timeout");
        let mut values = values.iter();
        let deadline = if let Some(value) = values.next() {
            if values.next().is_some() {
                return Err(invalid());
            }
            let value = value.to_str().map_err(|_| invalid())?;
            let duration = parse_timeout(value)?;
            Some(now.checked_add(duration).ok_or_else(invalid)?)
        } else {
            None
        };
        Ok(Self { id, deadline })
    }
    pub(crate) async fn expired(self) {
        match self.deadline {
            Some(deadline) => tokio::time::sleep_until(deadline).await,
            None => std::future::pending().await,
        }
    }
    pub(crate) async fn run<T>(
        self,
        future: impl Future<Output = Result<T, Status>>,
    ) -> Result<T, Status> {
        if self
            .deadline
            .is_some_and(|d| d <= tokio::time::Instant::now())
        {
            return Err(Status::deadline_exceeded("RPC deadline exceeded"));
        }
        tokio::select! { biased; _ = self.expired() => Err(Status::deadline_exceeded("RPC deadline exceeded")), result = future => result }
    }
}
fn invalid() -> Status {
    Status::invalid_argument("invalid grpc-timeout")
}
fn parse_timeout(value: &str) -> Result<Duration, Status> {
    if !(2..=9).contains(&value.len()) || !value.is_ascii() {
        return Err(invalid());
    }
    let (number, unit) = value.split_at(value.len() - 1);
    if !number.bytes().all(|b| b.is_ascii_digit()) {
        return Err(invalid());
    }
    let number = number.parse::<u64>().map_err(|_| invalid())?;
    Ok(match unit {
        "H" => Duration::from_secs(number * 3600),
        "M" => Duration::from_secs(number * 60),
        "S" => Duration::from_secs(number),
        "m" => Duration::from_millis(number),
        "u" => Duration::from_micros(number),
        "n" => Duration::from_nanos(number),
        _ => return Err(invalid()),
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn grammar_units_and_duplicates() {
        for (s, d) in [
            ("1H", 3_600_000_000_000),
            ("1M", 60_000_000_000),
            ("1S", 1_000_000_000),
            ("1m", 1_000_000),
            ("1u", 1000),
            ("1n", 1),
            ("0S", 0),
        ] {
            assert_eq!(parse_timeout(s).unwrap().as_nanos(), d);
        }
        for s in [
            "",
            "H",
            "123456789S",
            "-1S",
            "+1S",
            "1s",
            "1.0S",
            " 1S",
            "1S ",
        ] {
            assert!(parse_timeout(s).is_err());
        }
        let mut m = MetadataMap::new();
        m.append("grpc-timeout", "1S".parse().unwrap());
        m.append("grpc-timeout", "2S".parse().unwrap());
        assert!(RequestContext::new(RequestId(1), &m).is_err());
    }
}
