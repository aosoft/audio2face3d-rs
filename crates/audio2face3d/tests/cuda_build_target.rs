#[path = "../build_cuda_arch.rs"]
mod build_cuda_arch;

#[test]
fn selects_lowest_generic_target_supported_by_kernels_and_toolkit() {
    for (supported, expected) in [
        ("compute_86\ncompute_52\ncompute_50\ncompute_89", 50),
        ("compute_120\ncompute_90a\ncompute_75\ncompute_100f", 75),
        ("compute_35\ncompute_80\ncompute_50", 50),
        (" compute_100 \r\ncompute_100\r\ncompute_120", 100),
    ] {
        assert_eq!(
            build_cuda_arch::minimum_target(supported).unwrap(),
            expected
        );
    }
}

#[test]
fn rejects_missing_or_specialized_only_targets() {
    for supported in [
        "",
        "compute_35",
        "compute_90a\ncompute_100f",
        "sm_50\ncompute_8.6",
        "compute_\ncompute_75 extra",
    ] {
        assert!(
            build_cuda_arch::minimum_target(supported).is_err(),
            "{supported}"
        );
    }
}
