# License text fallbacks for GUI packaging

Prefer license/notice files included with each locked Cargo package. A few
upstream archives omit them; `package-gui.ps1` preserves package manifests and
README attribution and supplies these texts. MIT/Apache dual-license packages
without either text are distributed under the Apache-2.0 option.

- Apache-2.0: standard text from `crates/audio2face3d/LICENSE-APACHE`.
- BSL-1.0: standard text from error-code 3.4.0's LICENSE, for clipboard-win.
- CC0-1.0: standard text from blake3 1.8.7's LICENSE_CC0, for hexf-parse.
- tonic-MIT: tonic 0.14.6 LICENSE (Lucio Franco); tonic-prost and
  tonic-prost-build 0.14.6 belong to the same upstream tonic workspace.
- protoc-MIT: upstream rust-protoc-bin-vendored LICENSE.txt at
  `895c0433c3727a552970ce961e398e20e52d6353`, the source revision recorded by
  protoc-bin-vendored 3.2.0 (`https://github.com/stepancheg/rust-protoc-bin-vendored`).
  These packages supply build-time tools, not redistributed executables.

Recheck fallback attribution when upgrading the corresponding dependency.
