# Imitation Learning Baseline (case3) — Risks and Dependencies

## リスクリスト

| # | リスク | 影響 | 発生確率 | 緩和策 |
|---|--------|------|----------|---------|
| 1 | 推論タイムアウト (Kaggle 1s 超過) | High | Low | Step 9 でローカル p95 計測。torch.no_grad + CPU 推論 + 小モデル (<1MB)。万一超える場合 torch.jit.script でグラフコンパイル化。Step 11 の評価で `turn_p95 < 1.0s` を必須化。 |
| 2 | Kaggle Validation Episode failed (モデル同梱 or import 経路) | High | Medium | `pipeline/.submitignore` に training/evaluation/configs を必ず追加、`Path.cwd()` ベース sys.path を厳守、相対 import 縛り。Step 9 完了時に `uv run python -m submit submit case3 --dry-run --skip-validation` で main.py ロード確認。 |
| 3 | 学習データ不足 (rating top25% で ~100 episodes, ~60K frames) | Medium | Medium | MVP は 60K で走らせ、検証 loss が epoch 5 以降で飽和するなら rating cutoff を緩和 (top50%) して再学習。最終手段として kaggle_episodes を追加スクレイプ (`src/env/kaggle/cli.py`)。 |
| 4 | BC compounding error (実対戦で分布外の状態に弱い) | Medium | Medium | vs baseline 100 戦で seed を多様化 (seed 0..99)。勝率が 50% を下回る場合、Step 2 の preprocess で「自陣劣勢時のフレーム」を up-sample する等で分布拡張。**根本対策は DAgger / self-play だが case3 スコープ外 → case4 に持ち越し**。 |
| 5 | Action 空間のミスマッチ (from/target 分類が Orbit Wars の戦術 crash-exploit 等を表現できない) | Medium | Low | eval で敗北パターンを分析し、必要なら複数手を同時に出す "multi-action" 拡張を検討 (現在の decoder は my_planet ごとに 1 action 出力)。 |
| 6 | torch 新規依存で環境構築が壊れる | Low | Low | `pyproject.toml` で CPU 版のみ許可 (`torch>=2.3.0` のみ指定、index URL 指定なし)。`dev/setup` で `uv sync` が成功するか Step 1 完了時に確認。 |
| 7 | 他 case (case0/1/2) への意図しない依存発生 | Medium | Low | Step 8 の geometry.py 独立コピー時にライセンス表記 (Apache 2.0) を付け、import 経路 grep で `pipeline.case[012]` への参照を 0 に維持。CI に `grep -r "from pipeline.case[012]" pipeline/case3/` が空になるチェックを入れる (オプション)。 |
| 8 | parquet 前処理が重すぎる (大量 replay をメモリ不足) | Low | Low | replay は 1 本ずつ stream 処理 (gzip 展開 → frame 抽出 → 即 dict append)。全 replay 並列展開はしない。 |
| 9 | 学習ハイパーパラメータが合わず勝率が頭打ち | Medium | Medium | lr, hidden, epochs を yaml で複数試せる前提にする。MVP は lr=1e-3, hidden=64, epochs=10, batch=256 から開始し、Step 11 で不合格なら一段ずつ調整。 |
| 10 | Kaggle runtime の torch バージョン不整合 | Medium | Low | Kaggle 標準は torch 2.x CPU 版。`pyproject.toml` の torch>=2.3.0 と互換範囲内を想定。万一 validation で fail したら Kaggle Notebook 側の `torch.__version__` を確認して `pyproject.toml` を pin する。 |
| 11 | ffa4 データの player perspective 正規化ミス | Medium | Low | preprocess で winner 側を player として取り出すロジックを Unit test でカバー (fixture: 4 人 FFA で player=2 が勝利するケース)。 |

## 外部依存

- **PyTorch (新規追加)**: CPU 版 torch>=2.3.0。`pyproject.toml` に追加、Kaggle ランタイムとは同梱不要。
- **Kaggle runtime**: torch / numpy / kaggle_environments プリインストール済み前提。
- **既存の `src/env/`, `src/submit/`**: 並列対戦、提出 packaging は既存を利用。新規変更は `src/env/agents.py` の AGENT_REGISTRY に 1 行追加のみ。
- **`data/kaggle_episodes/`**: 既に 798 replay 蓄積済み。追加取得は不要 (必要になったら `src/env/kaggle/cli.py` を走らせる)。
- **`data/lake/case3/`**: 本 case が生成する唯一のデータ出力先。他 case からは参照されない。

## 技術的負債 (本 case が導入し得るもの)

- **重複コード**: `pipeline/case3/policy/geometry.py` は `pipeline/case1/baseline/core/physics.py` の複製。case 独立原則のため意図的だが、将来的に `src/features/geometry.py` に集約する余地あり (case4 以降で検討)。
- **ハードコード定数**: MAX_PLANETS=36, ships_buckets=5 は model と preprocess の両方に現れる → `pipeline/case3/configs/il_baseline.yaml` に一元化し、両者が読む形に統一すべきだが、MVP では constants.py に定数モジュールを用意する簡易案を採用。
- **マルチモデル評価の不在**: 1 モデル 1 重みのみ。ベスト loss 以外のチェックポイントも保存しないため、チューニング時に複数 run の比較が面倒 (yaml の run_name を変えて手動管理)。

## Open Items (着手前に Step 1 で最終確認)

- [ ] `data/kaggle_episodes/matches/` の replay は 1v1/ffa4 両方を対象にするか、1v1 のみで MVP を回すか → 要件では両対応だが、MVP 初回学習は 1v1 のみで動作確認してから ffa4 を足す段階的進行を推奨。
- [ ] Kaggle ランタイムの torch バージョンは実機で確認 (現状の実績から 2.x 系想定)。
- [ ] `ships_buckets` の境界値 5 クラスをどう定義するか。案: `[=need, =all, =50%, =75%, =25%]`。デモデータの ships 分布を preprocess 時に観察してチューニング。
- [ ] `FROM_THRESHOLD` (推論時の from 確率閾値) の初期値 0.5 でよいか。vs baseline 評価で動きが少なすぎる場合 0.3 に下げる。
