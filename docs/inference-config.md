# 推論設定と server 設定の共通化（R2）

server の `Config` は完成した `inference::Config` と通信設定を保持する。backend は `config.engine().clone()` を使用し、推論フィールドを転記しない。共通の検証は inference に集約した。

```rust
let engine = audio2face3d::inference::Config::builder(
    audio2face3d::inference::BackendKind::Mock,
)
.mock_curve("TongueOut")
.mock_value(0.2)
.mock_jaw_open(0.8)
.build()?;

let server = audio2face3d_server::ServerConfig::builder(
    audio2face3d_server::config::BackendKind::Mock,
)
.engine(engine)
.max_streams(2)
.build()?;
```

`engine(config)` は推論設定全体を置き換え、通信設定を保持する。後から呼んだ個別 setter は該当する値だけを上書きする。`into_builder().build()` は全設定を保持し、`optional_*` の `None` と `Some(0.0)` を区別する。公開の builder/getter/error 型と `_ms: u64` は維持した。

## 入力受理条件の変更

直接利用でも、従来 server が拒否していた次の指定を拒否する。

- `mock_curve` を指定せずに `mock_value` / `mock_jaw_open` を指定する。
- `JawOpen` に `mock_jaw_open` を併用し、選択カーブの値を上書きする。
- mock backend にモデル、感情モデル、非ゼロ device を指定する。
- regression backend に `mock_curve` / `mock_value` / `mock_jaw_open` を指定する。

カーブ名、重みの有限性と範囲、device ordinal、音声長、regression のモデル指定も inference が検証する。preset の `mock_pattern` は従来どおり保持でき、regression では使わない。

server は feature の利用可否、descriptor ファイルの存在、通信容量と timeout を追加検証する。直接利用の設定 builder はファイルを開かず、native runtime / GPU を初期化しない。モデルの内容検査は loader が行う。

## 公開 enum の互換性

server と inference の `BackendKind` / `MockPattern` は別型のまま維持した。両型に利用側の同じ trait を実装するコードとの互換性を保つためである。相互変換を用意し、server の `FromStr` は inference の解析へ委譲する。既存の文字列と解析 error は維持する。

## 検証対象

有効・不正な設定の共通表、mock の曖昧な指定、round trip、setter の順序、optional 値、全通信設定、別 enum への trait 実装、CLI / gRPC の回帰を確認する。native 構成でも設定構築だけでは descriptor の内容や GPU を初期化しないことを確認する。

2026-10-10 に次を確認した。

- server `cli,mock`: library 13件、config builder 6件、protocol 15件成功。
- server `native,mock`: config builder 7件成功。内容が不正な既存ファイルと実在しない device ordinal を設定しても、設定構築だけでは native 初期化を行わない。
- library `grpc-server`（mock/native 無効）: inference config 3件成功。
- server backend 無効: library 12件成功。
- server `cli,mock` 全ターゲット Clippy（`-D warnings`）、workspace 全 feature・全ターゲット型検査、format 検査成功。
