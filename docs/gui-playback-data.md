# 描画履歴と音声再生の分離（R3 / R7）

標準 Desktop は `SessionController` が Clip、曲線、感情履歴を所有する。callback と共有する Player は immutable な PCM block の Arc handle と小さい再生情報を保持する。Player の内部 source は owned Clip または controller の音声に分かれ、標準 Desktop は曲線履歴を Player に格納しない。

controller は受信イベントを音声 lock 外で検証・反映し、バッチ終了時に新しい block handle と readiness を publish する。追加 handle の準備は lock 外で行う。最初の置換でも、音声 payload の複製や曲線履歴の移動を lock 内で行わない。

`PlaybackClockSnapshot` は Copy の時計情報。controller は時計の取得後に lock を解放し、ClipId の一致を確認してから曲線を補間する。Timeline は controller の Clip を直接集計し、感情表示にも音声 lock を使わない。旧 ClipId の更新と時計は拒否し、Timeline cache は ClipId、layout/curve revision、幅と表示範囲で失効する。同じ範囲への音声追加だけでは曲線 cache を捨てない。

## API とメモリ

`Clip::audio()` は `AudioTrack` を読み取り用に返す。`len`、`is_empty`、`get`、`iter` が利用でき、明示的な連続配列が必要な export などでは `to_vec()` を使う。JSON serialization は従来の平坦な sample 配列を維持する。PCM の decode 後は `push_audio_owned` で payload を一度だけ block 化する。

Player は block cursor を使い、通常の隣接 sample 読取りで全 block を走査しない。Seek の移動先は二分探索し、block 境界でも線形補間を維持する。

payload は一度だけ予算に加算する。共有 handle 配列、block 構造、Arc の overhead も Clip のメモリ検査へ含める。空追加は block を増やさず、極小 block は最大65,536件で制限する。従来の時間・メモリ上限も維持する。

`Player::replace(Clip)`、`snapshot`、`Timeline::show_player` は旧 owned-Clip 利用の互換入口として残る。新しい controller 利用は `clock_snapshot` / `with_view` / `snapshot` と `Timeline::show` または `show_clock` を使う。旧 `show_player` 自体は従来の lock を使うため、同じ改善を保証しない。

command、時計取得、block handle 追加と callback の lock 競合は残る。完全な lock-free 設計や hard realtime の保証は行わない。音声デバイスの実時間測定と長時間の GUI 操作による性能比較は、機能テストの成功とは別に扱う。

## R7 の移行

Clip と CurveSample のフィールド、Player の再生状態と設定フィールドは非公開にした。これは公開フィールドの利用側に修正が必要な API 変更である。

| 旧アクセス | 移行先 |
|---|---|
| `clip.names/frames/audio/emotions/revision` | 同名 getter。frame は `time()` / `values()` |
| `clip.session = Completed` | coverage を検証する `finish_collection()` |
| `clip.set_session(...)` | fallible な収集状態の遷移。完了後の再開や失敗への書換えは拒否 |
| `player.state/underruns/looping/streaming/stream_buffer` | 同名 getter。再生変更は `command()`、設定は `set_streaming()` |
| `player.clip` の参照 | owned 経路なら読み取り専用 `clip() -> Option<&Clip>`、標準経路なら controller の `clip()` |
| `player.clip.push_audio/push_frame` | owned 経路なら `append_audio/append_frame`。controller 経路は `apply_event` |
| `Player::from_clip` / `PlaybackUpdate` の外部利用 | `SessionController::replace` / `apply_event` / `publish` |

mutation は検証、予算確認、revision の checked increment を済ませてから反映する。名前の保有容量も予算に含める。終端後の追加は拒否し、新しい Clip へ置換して収集を再開する。

Clip の `SessionState` は収集状態として維持する。controller は別に Idle / Initializing / Receiving / Finalizing / Ready / Failed / Cancelled を持つ。出力収集の Completed は Finalizing であり、job の `job_finished(Ok(()))` を受けて Ready になる。収集済み Clip の受渡しには `replace_output`、既に成功済みのデータの表示には `replace` を使う。

Finalizing 中の streaming は従来の短い末尾を再生できる。offline Play は job 成功まで待つ。job 失敗・キャンセルは Clip の収集完了を保持したまま Player を現在の可聴位置で停止し、以後の Play を拒否する。後続の成功では失敗を解除せず、新しい Clip への置換でリセットする。旧 job 由来の入力には ClipId を確認する `apply_event_for` / `job_finished_for` を使う。

R7 のテストは不正 frame、非連続音声、予算超過、名前変更、coverage 不足、revision overflow の更新前拒否、終端後追加、job の遅い失敗とキャンセル、offline 開始条件、旧 ClipId、音声のみの cache 維持と無効な選択 channel を検証する。公開フィールドの書換えを拒否する compile-fail doctest も追加した。

## 検証

2026-10-10 に次を確認した。

- `session,mock,ui-egui,emotion`: library 18件、再生4件、streaming 3件、推論4件、backend 再利用1件成功。
- `standalone-app,grpc,mock,emotion`: 最終 library 35件と起動9件成功。全ターゲット Clippy（`-D warnings`）成功。
- グラフ処理を barrier で停止中の音声 lock 取得、payload の Arc 共有、16/48 kHz の block 境界補間、Pause/Seek/Loop、underrun 復帰、旧 ClipId と古い revision の拒否、同じ revision の別 Clip の cache 失効、極小 block 上限を確認した。
