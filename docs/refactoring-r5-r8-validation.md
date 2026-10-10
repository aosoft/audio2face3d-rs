# R5〜R8 の変更と検証

2026-10-10〜11 に、workspace 0.4.0、Windows、CUDA 12.9、TensorRT 10.16.1.11 と取得済みモデルで確認した。

| 項目 | 変更 | 詳細 |
|---|---|---|
| R5 | host / device adapter のモデル別 driver と geometry の interactive 反復を共有 | [native scheduler](native-schedulers.md) |
| R6 | server が cleanup を所有し、client の状態遷移と lock 外の副作用を集約 | [request lifecycle](request-lifecycle.md) |
| R7 | Clip / Player をカプセル化し、収集完了と job の最終結果を分離 | [GUI API の移行](gui-playback-data.md) |
| R8 | device、factory、worker、SDK tools に所有者の Context を明示的に渡す | [Context 所有](runtime-context-ownership.md) |

R7 は公開フィールドを利用する側の修正が必要な API 変更を含む。getter と検証付き mutation、controller の `Finalizing` / `Ready` を使う。workspace version、モデルの数式、seed、wire の正常系の順序は変更していない。

## 成功した検証

- `ci/run-tier.ps1 -Tier portable`：workspace の portable テスト、feature 境界、GUI、Clippy、fmt、rustdoc。R8 の追加経路は `mock,model-management` の library / logging / Context / runtime-tools / Direct client テストで再確認した。
- `ci/run-tier.ps1 -Tier cuda-lifetime -PlatformConfig platform.toml`：実 CUDA のバッファ・callback lifetime 契約と feature 別テスト。
- `ci/run-tier.ps1 -Tier tensorrt-model -PlatformConfig platform.toml`：最終コードの全 feature / 全 target の Clippy、release の workspace テスト、実モデルの `acquired_models_execute_through_completed_facades`、警告を拒否する rustdoc。
- `native_runtime_context` の ignored test：Context A の device と post-process factory を Context B の scope 内で生成し、スレッド移動して利用。別プロセスで、明示初期化と lazy 初期化の資源共有・呼び出し元破棄後の利用も確認した。
- server の `native_lifecycle` / `native_reuse` の ignored test：prepare 中の停止、利用できない SDK 設定の拒否、成功・キャンセル後の再利用と出力の再現性。3件成功。
- CUDA の sealing を確認する compile-fail 契約：`animation` の最小構成と全 feature の release 構成で同じ期待値が成功。任意依存の同名 trait による診断差分を除くため、非公開 marker の名前を固有にした。
- 最終の `cargo fmt --all -- --check` と `git diff --check`。

## C++ SDK との参照比較

`ci/run-tier.ps1 -Tier reference-parity -PlatformConfig platform.toml` と `reference/run-sdk-compatibility.ps1` を使った。全ケース FP32、seed 0、同じ mono PCM16 / 16 kHz の入力で、SDK と Rust の両方から新しく capture した。

| Pipeline / execution | Tracks | Records | 最大絶対誤差 |
|---|---:|---:|---:|
| Emotion / standard | 1 | 147 | 0.0009782314 |
| Regression / teeth-standalone | 2 | 2 | 0.0000271201 |
| Regression / standard | 2 | 1470 | 0.0002961159 |
| Regression / interactive-all | 1 | 735 | 0.0000572056 |
| Diffusion / standard | 2 | 2940 | 0.0005340576 |
| Diffusion / interactive-all | 1 | 1470 | 0.0004234314 |
| Emotion / standard | 2 | 294 | 0.0009782314 |
| Emotion / interactive-all | 1 | 147 | 0.0009782910 |

全8件で metadata が完全一致し、数値は `reference/tolerances.json` の成分別の許容差を満たした。レポートは `temp/reference-comparison/<case>/comparison.json` に保存する。入力音声・モデル・生成物は commit に含めない。

これらは正しさの検証であり、性能改善、bitwise の CPU/GPU 一致、GUI の実機での可聴性やデバイス抜き差し、全モデル・全 precision の網羅を主張するものではない。
