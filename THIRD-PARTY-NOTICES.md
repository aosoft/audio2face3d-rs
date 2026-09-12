# Third-party source notices

The CPU BlendShape BVLS algorithm in
`crates/audio2face3d/src/animation/blendshape/bvls.rs` follows NVIDIA
Audio2Face-3D-SDK `audio2face-sdk/source/audio2face-core/bvls.cpp`, revision
`1ca0f02535ed774f5dbcd724a31cd486368dc783` (MIT). The source retains its NVIDIA
copyright notice; the MIT text is in `LICENSE` and the library's `LICENSE`.

`crates/audio2face3d/src/animation/blendshape/bvls/svd.rs` is a Rust adaptation
of Eigen 3.4's `JacobiSVD.h`, `RealSvd2x2.h`, `Jacobi.h`,
`ColPivHouseholderQR.h`, `Householder.h`, `SVDBase.h`, `Redux.h`,
`GeneralMatrixVector.h`, `GeneralBlockPanelKernel.h`, `Memory.h`, and SSE
`PacketMath.h`, as supplied with that SDK. This file, including its Rust
modifications, is licensed under MPL-2.0 and retains the original attribution.

The MPL text is in the root `LICENSE-MPL-2.0`. The library's `LICENSE` also
contains it so the standalone Cargo source archive retains the full text.
The library package declares `MIT AND MPL-2.0`; independently written MIT
files keep their MIT license. Source distributions retain the SVD source and
its notices. Binary distributions must provide access to the corresponding
MPL-covered source, including any modifications; the release archive or an
exact source revision can be used for this purpose. See the
[Mozilla MPL FAQ](https://www.mozilla.org/en-US/MPL/2.0/FAQ/).

Models and external CUDA/TensorRT installations are not included in these
source licenses; their own terms continue to apply.
