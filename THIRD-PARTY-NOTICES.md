# Third-party source notices

This project is an independently maintained Rust port of NVIDIA's MIT-licensed
[Audio2Face-3D-SDK](https://github.com/NVIDIA/Audio2Face-3D-SDK). Its upstream
source and reference revision is recorded in the
[root README's License section](README.md#license). NVIDIA attribution and the
MIT permission text are retained in the root and crate `LICENSE` files.
The Rust port implements the pipelines without linking the original Audio2Face
SDK libraries; CUDA and TensorRT remain native dependencies.

`crates/audio2face3d/src/animation/blendshape/bvls/svd.rs` is a Rust adaptation
of Eigen 3.4's `JacobiSVD.h`, `RealSvd2x2.h`, `Jacobi.h`,
`ColPivHouseholderQR.h`, `Householder.h`, `SVDBase.h`, `Redux.h`,
`GeneralMatrixVector.h`, `GeneralBlockPanelKernel.h`, `Memory.h`, and SSE
`PacketMath.h`, as supplied with that SDK. This file, including its Rust
modifications, is licensed under MPL-2.0 and retains the original attribution.

The MPL text is in the root `LICENSE-MPL-2.0`. The library's `LICENSE` also
contains it so the standalone Cargo source archive retains the full text.
The library package declares `MIT AND MPL-2.0 AND Apache-2.0`; independently written MIT
files keep their MIT license. Source distributions retain the SVD source and
its notices. Binary distributions must provide access to the corresponding
MPL-covered source, including any modifications; the release archive or an
exact source revision can be used for this purpose. See the
[Mozilla MPL FAQ](https://www.mozilla.org/en-US/MPL/2.0/FAQ/).

External CUDA/TensorRT installations remain subject to their own terms.
These notices describe embedded SDK/Eigen source;
binary releases must also retain notices required by their other dependencies.

The ACE protocol definitions under `crates/audio2face3d/proto` retain their
upstream notices and use Apache-2.0. The license text is included in
[the library package](crates/audio2face3d/LICENSE-APACHE).
