# 音声デバイスの障害通知（R4）

stream 世代ごとに `StreamEpoch` を作る。最初の CPAL error は `OnceLock` に保持し、状態を `HEALTHY → WRITING_FAULT → FAULT_READY → FAULT_HANDLED` と進める。後続の通知は回数に集約する。同じ slot を healthy に戻したり、最初の error を取り出して再利用したりしない。

CPAL error callback は Player/UI の Mutex を取得せず、文字列化も stream の操作も行わない。保持している worker の Thread を unpark する。通知がキュー満杯や Player lock 競合で失われる経路をなくした。CPAL error の保持・解放はあるため、確保・解放のない hard realtime 処理とは扱わない。

worker は request と現在の epoch を毎回確認し、park/unpark の token で待機直前の通知にも対応する。初期化・破棄などの処理中に token が消費された場合にも対応できるよう、すべての blocking 処理後に request / READY / shutdown を再確認してから park する。WRITING の段階では処理せず、READY の Release/Acquire 公開を待つ。start が返った直後にも fault を調べる。

起動失敗と実行中の障害は同じ `handle_failure` を通り、Player と command の同じ lock 内で現在世代を再照合してから Pause と status 更新を行う。Pause は audible position を保存する。OS stream の破棄は worker 上で、Player lock を解放した後に行う。

faulted epoch と旧世代の data callback は無音を返す。古い通知は新しい Player 状態や stream を変更しない。故障した世代を自動再起動せず、再生を再開する明示 command で新しい slot を作る。

UI は `AudioOutput::take_error()` で現在世代の障害だけを受け取る。従来の公開 `errors` 欄は互換用に残すが、世代を考慮する利用では新しいメソッドを使う。標準 Desktop はこのメソッドへ移行した。

## 検証方針と範囲

fake stream と同じ worker loop の待機関数を使い、channel/barrier で順序を制御する。起動・実行中・待機直前・WRITING 中の通知、lock 保持中の報告、同時 callback、旧世代の通知、再開、二重処理、worker 上かつ lock 外での stream 破棄を確認する。

これは OS のデバイス初期化・破棄が停止しないことを保証する変更ではない。物理デバイスの抜去や OS backend の実時間挙動は fake stream の機能テストとは別に確認する必要がある。

## 検証結果

2026-10-10 に音声 worker の10テストが成功した。Player と status の lock を保持しても callback の報告が戻り、worker が lock 解放後に Pause と通知を行い、stream を worker 上かつ lock 外で破棄することを確認した。

R2〜R4 の統合検証として `ci/run-tier.ps1 portable` を offline で実行し、全項目成功した。workspace tests、GUI feature 別 tests、Clippy（`-D warnings`）、format、rustdoc（`-D warnings`）を含む。さらに CUDA/TensorRT・GUI local/capture を含む workspace 全 feature・全ターゲットの locked/offline 型検査と最終 format 検査が成功した。
