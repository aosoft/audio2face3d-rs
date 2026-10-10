# 実行 Context の所有と伝播（R8）

CUDA、TensorRT、モデル生成ツールの実行環境は、資源を所有する `Audio2Face3DContext` から取得する。内部処理でログの thread-local scope を読み直して SDK を選択しない。

- `GpuDevice` は Context とログ scope を別々に保持する。`new_with_context` は渡された Context を直接初期化に使い、`new` は互換入口で一度だけ capture する。Context の clone は既存の資源を共有する。
- `TensorRtSession` は device の Context で native API を初期化する。ログの request fields は同じ Context を共有する scope からだけ継承する。
- inference `Factory` は Context を保持し、Direct client、native worker の初期化 closure、Regression と Classifier の読み込みまで明示的に渡す。
- geometry / classifier / post-process の各 factory と bundle の `load_with_context` は、Context を worker に直接渡す。compatibility `load` の同期入口だけが初回 capture を行う。
- `EngineBuilder::build_with_context` と新しい `ModelEngineBuildRequest::execute_with_context(&context)` は、その Context の `NativeRuntimeConfig::tool_command` を使う。生成、既存成果物の照合、`trtexec` / `nvcc` のバージョン確認で同じ設定を共有する。公開の差し替え用 trait は追加していない。

`LogScope` は初期化ログ、要求ごとの fields、future の poll/drop、native cleanup のログ伝播に引き続き使う。同期区間の guard を `.await` の間保持せず、非同期区間には既存の `wrap_future` を使う。既存の `activate()` と公開 `LogScope::context()` は維持する。`GpuDevice::new_with_context` の初期化は従来どおり新規 scope を使う。

`TensorRtSession::load_with_context` でも SDK の選択は渡された device の所有者に従う。外側の scope が別 Context を指していても device の資源を切り替えない。

テストは、Context A の factory/device を Context B の scope 内で作成し、別スレッドで利用しても A の logger と資源を使うことを検証する。mock は native 初期化なしで実行できる。実 CUDA の検証は SDK roots を指定した ignored test を別プロセスで実行する。emotion feature がある場合は post-process bundle と interactive factory の明示 Context も確認する。

モデル生成の portable テストでは小さな Rust 製の `trtexec` / `nvcc` fixture を使い、生成時の子プロセス環境と provenance が A の設定に揃うこと、既存成果物の照合が同じ identity を使うことを確認する。既存のログ poll/drop、Context clone、探索設定、子プロセスだけの環境変更に関するテストも維持する。

統合検証では CUDA / TensorRT の実行、Context clone の資源共有と呼び出し元破棄後の利用、実モデルの worker cleanup と再利用も成功した。全 feature の診断に tonic の同名 blanket trait の補足が混ざる問題に対して、CUDA の非公開 marker を `DeviceElementSealed` に改名した。許可型と sealing の契約は変えていない。更新した compile-fail 期待値は最小 feature と全 feature の両方で確認した。

検証のコマンドと参照比較の結果は [R5〜R8 の検証記録](refactoring-r5-r8-validation.md) にまとめる。
