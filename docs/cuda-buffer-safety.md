# CUDA バッファの安全性と API 移行

R1 の最初の安全性修正では、公開された安全な API が返る前に、初期化と借用資源への GPU アクセスを完了させる。非同期処理の所有型への移行と初期化コストの削減は、後続の性能改善として扱う。

## 転送できる型と初期化

`cuda::DeviceElement` は外部から実装できない sealed trait である。対応する型は、現行コードが使用する `u8`、`u32`、`u64`、`i32`、`f32`。いずれも所有ポインタ・参照・padding を持たず、すべてのビット列が有効な Rust の値となる。`String`、`bool`、`NonZeroU32`、参照型は転送対象にできない。

`GpuDevice::allocate` は `allocate_zeroed` へ委譲し、ゼロ書込みとその完了待ちを終えてからバッファを返す。`from_slice` も転送の完了まで入力 slice を借用する。ゼロ長と byte count の overflow はエラーにする。

`DeviceView` は読み取り用。書込み用の `DeviceViewMut` は `DeviceBuffer::view_mut(&mut self)` から生成し、`Copy` / `Clone` を実装しない。`as_view` は書込み view を短く共有借用するための入口であり、元の排他借用を解除しない。

## TensorRT の入力と出力

従来の公開 `BindingBuffer` と汎用 `DeviceBindings::insert` は、入力と出力を区別する API に置き換えた。

```rust
use audio2face3d::tensorrt::{DeviceBindings, InputBinding, OutputBinding};

let mut bindings = DeviceBindings::new();
bindings.insert_input("input", InputBinding::from_view(input.view()))?;
bindings.insert_output("result", OutputBinding::from_view(output.view_mut()))?;
session.enqueue(0, &mut bindings, &stream)?.synchronize()?;
drop(bindings); // 出力バッファを読み戻す前に排他借用を終了する。
```

型から TensorRT の dtype を決定する `TensorElement` は、現在 `f32` に実装している。`OutputBinding::from_view` は排他的な view を消費する。`enqueue` / `enqueue_with_postprocess` は `&mut DeviceBindings` を借用し、別 session との同時共有も禁止する。

encoded tensor の出力には `OutputBinding::from_bytes(DeviceViewMut<u8>, dtype)` を使える。入力の `InputBinding::from_bytes` は `unsafe` であり、呼出し側が native dtype とビット表現の対応を保証する。特に boolean の入力を任意の byte 列から安全に構築できる API は提供しない。

推論前には device、入力/出力の用途、dtype、runtime shape と byte count を照合する。TensorRT の `setTensorAddress` は256バイトの整列を要求するため、要素型の自然な整列とは別に検査する。仕様の根拠は [NVIDIA IExecutionContext API](https://docs.nvidia.com/deeplearning/tensorrt/latest/_static/c-api/classnvinfer1_1_1_i_execution_context.html)。

この device binding API は device 上の dense な `kLINEAR` 格納だけを扱う。host tensor は Rust の metadata 境界で、packed / vectorized format は native shim の metadata 境界で拒否する。これらの対応には別の所有・容量計算の契約が必要になる。native dtype の `kBOOL = 4`、`kINT64 = 8` も正しく復号し、使用するタグと device location は C++ の `static_assert` で SDK 定義と照合する。

## 借用処理と失敗時の完了保証

安全な host/device 転送、TensorRT の実行、cuBLAS PCA、cuRAND の乱数生成、GPU の teeth・blendshape・postprocess は、完了を待ってから返る。既存の fence は互換のため残すが、返却時点で処理は完了している。`mem::forget(fence)` で借用を早期解放しても、未完了の処理がその資源へアクセスする経路を作らない。

内部 `CompletionGuard` は native 呼出しの前に作成する。正常終了では明示的に完了させ、途中の submission error、event record の失敗、postprocess のエラー、panic/unwind でも stream を完了させてから借用を解放する。event は可能な範囲で投入前に作成する。

**完了待ちが失敗した場合はプロセスを abort する。** エラーを返すだけでは、転送先の借用 host slice や借用 device buffer に GPU がアクセスし続ける可能性を排除できないためである。公開の単独 `CudaStream::synchronize` は引き続き `Result` を返すが、借用資源を保持する安全な実行境界では、完了不能を通常の回復可能エラーとして返さない。

完了関数自身が panic した場合も内部で捕捉して abort し、通常の unwind で借用を解放しない。この経路とエラー返却の経路は、子プロセスが呼出し元に戻らず終了するテストで検証する。

`copy_from_async`、raw kernel launch、raw pointer の memset、cuBLAS の低水準 enqueue などは引き続き `unsafe`。利用者は全資源の寿命、排他性、実行順序、native ABI を保証する必要がある。

## stream の所有

`CudaStream` と cuBLAS/cuRAND handle は内部 `Arc<StreamInner>` を共有する。元の `CudaStream` を落としても handle が native stream を保持し、最終所有者だけが destroy する。handle の実行に渡した stream は、作成時に設定した stream と同一であることも検証する。同じ device の別 stream を渡して、実処理と完了 event の記録先を食い違わせることを防ぐ。

## 互換性と性能

- `DeviceBuffer<T>` / `DeviceView<T>` の型制約と、公開 `BindingBuffer` の置換はソース互換性に影響する。
- model buffer contract の出力引数と diffusion state は `&mut` を要求する。
- `solve_async` など既存の名前が残る入口も、現在は完了してから戻る。非同期の戻り時点を前提にしたコードは移行が必要である。
- allocation に zero-fill と専用 stream の生成・完了待ちが追加される。推論・後処理にも完了待ちが追加されるため、性能の互換性は保証しない。
- 後続の最適化では、初期化状態を区別する内部バッファと、資源を実所有する submission を導入する。安全な借用 API の完了保証は弱めない。

## 検証

2026-10-10 に Windows / RTX 3070 Ti Laptop GPU、CUDA 12.9、TensorRT 10.16.1.11 で検証した。既存の Claire、diffusion、emotion のモデルを使用した。

- native ライブラリテスト: 212 passed / 0 failed / 1 ignored。regression / diffusion / classifier の実モデル推論、GPU の数値照合、初期化直後の読戻し、ビット列保存、stream 所有と同一性、fence の忘却、postprocess の error / unwind を含む。
- 追加の read-only output / 不正な pointer alignment の拒否を含む TensorRT session テスト5件も、実モデルを有効にして成功した。
- native コンパイル契約: 5グループ成功。禁止型・sealed trait・view と output binding の排他性・寿命・thread safety を確認した。期待診断の作成後、自動更新なしでも実行する。
- CUDA 無効の `animation` 構成: sealed な型契約のコンパイルテストが成功。
- CUDA/TensorRT 全ターゲットの Clippy（`-D warnings`）、全 feature の workspace 全ターゲット型検査、format 検査を実施。

SDK capture を要求する ignored の BVLS parity テスト、2台目の GPU を必要とする実機間の検証、性能の before/after 比較は今回の成功結果に含めない。TensorRT の host / packed tensor の拒否は実装と SDK 契約の照合で確認しており、それら専用の engine による実機テストは未実施である。
