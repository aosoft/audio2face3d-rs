pub fn gencode_flags(value: &str) -> Result<Vec<String>, String> {
    let mut result = Vec::new();
    for architecture in value.split(',').map(str::trim) {
        let number = architecture
            .strip_prefix("sm_")
            .or_else(|| architecture.strip_prefix("compute_"))
            .unwrap_or(architecture);
        if number.len() < 2 || !number.bytes().all(|byte| byte.is_ascii_digit()) {
            return Err(format!("invalid CUDA architecture {architecture:?}"));
        }
        let flag = format!("-gencode=arch=compute_{number},code=sm_{number}");
        if !result.contains(&flag) {
            result.push(flag);
        }
    }
    if result.is_empty() {
        return Err("at least one CUDA architecture is required".into());
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_and_deduplicates_architectures() {
        assert_eq!(
            gencode_flags("86, sm_89,compute_86").unwrap(),
            [
                "-gencode=arch=compute_86,code=sm_86",
                "-gencode=arch=compute_89,code=sm_89"
            ]
        );
    }

    #[test]
    fn rejects_invalid_architecture() {
        assert!(gencode_flags("native").is_err());
    }
}
