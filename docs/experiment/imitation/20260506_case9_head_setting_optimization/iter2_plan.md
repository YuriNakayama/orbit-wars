# Imitation/case9 — dual head blend α=0.5 (iter2)

> 作成日: 2026-05-07
> 仮説 ID: H4
> hypotheses.md: docs/experiment/imitation/20260506_case9_head_setting_optimization/hypotheses.md
> 関連: docs/experiment/imitation/20260506_case9_head_setting_optimization/iter1_result.md
> スコープ: `bot/pipeline/imitation/case9/` の head 構造を dual head 化し、3-head と candidate head の loss を α=0.5 で同時学習する

## 仮説 (Hypothesis)

H4: dual head 全部入り (3-head + candidate, blend α=0.5) — H1 の from/target/ships 予測と H2/H3 の candidate slot 予測は失敗モードが異なる可能性があるため、共通 Set Transformer backbone 上で両 head を同時に学習し、loss を 0.5 / 0.5 で混ぜることで、表現学習と推論候補選択の安定性を改善できる。

## 既存コードの現状

- 主要モジュール: `bot/pipeline/imitation/case9/policy/model.py` は `three_head` / `candidate` / `candidate_ships` の 3 head mode を排他的に切り替える構造。
- head 実装: `policy/heads/three_head.py` と `policy/heads/candidate.py` は既に分離済みで、dual 化は新 head 追加または `Case9Policy` の mode 追加で対応可能。
- loss 実装: `training/losses.py` と `training/train.py` は head_mode ごとに loss dispatch しているため、`dual` 用 dispatch を追加する。
- 過去 iter: H1/H2/H3 はいずれも 10 ep = 0/10 で、n<300 ルールにより採否は inconclusive。val 指標は H1: `val_target_acc=0.928`、H2/H3: `val_cand_fire_acc=0.211`。

## スコープ (Scope)

- 変更ファイル:
  - `bot/pipeline/imitation/case9/policy/model.py`
  - `bot/pipeline/imitation/case9/policy/types.py`
  - `bot/pipeline/imitation/case9/training/losses.py`
  - `bot/pipeline/imitation/case9/training/train.py`
  - `bot/pipeline/imitation/case9/configs/il_case9_dual.yaml` (新規)
- ハイパーパラメータ / config:
  - `model.head_mode: dual`
  - `train.loss_weights.dual_alpha: 0.5`
  - backbone / featurizer / dataset / scheduler は既存 3 パターンと同一に固定
- データセット / 特徴量変更: なし
- 推論方針: 初期実装では candidate logits を主出力として decode に使い、3-head は補助 loss として扱う。必要なら result で 3-head blend inference を次仮説化する。

## 実装ステップ (Implementation outline)

1. `PolicyOutput` に dual mode で必要な 3-head 出力と candidate 出力を同時に保持できるフィールドを追加する。
2. `Case9Policy` に `head_mode="dual"` を追加し、`ThreeHead` と `CandidateHead` を同時に保持して forward で両方を返す。
3. `training/losses.py` に dual loss を追加する。式は `total = α * three_head_total + (1 - α) * candidate_total` とし、metric は両 head の主要値を同時に返す。
4. `training/train.py` の dispatch / metric map / best metric 周りに dual mode を追加する。best metric はまず `val_cand_fire_acc` を維持し、補助として `val_target_acc` を記録する。
5. `configs/il_case9_dual.yaml` を追加し、既存 config と同じ学習条件で `weights_out: pipeline/imitation/case9/policy/weights_dual.pt` に保存する。
6. inference の canonical 化はこの iter では行わず、評価時のみ `IL_CASE9_HEAD_MODE=dual` と dual weights で動作確認する。必要なら `agent.py` / `decoder.py` の dual 対応を最小追加する。

## 検証方法 (Validation method)

### スキップする検証 (from hypotheses.md skip list)

- Kaggle publicScore は引用しない。
- skill rating は使わない。
- 300 対戦による評価はしない。
- n<300 のローカル 10 / 30 ep 結果だけで確定結論を出さない。採否は val 指標 + ローカル挙動 + replay 定性で判断し、原則 inconclusive を許容する。

### 実施する検証

- ローカル smoke: 1 episode self-play で import / decode / action shape を確認する。
- ローカル test: `dev/test-bot` を実行する。
- リモート学習: `dev/runpod train <commit-sha> --case case9` 相当で RunPod 学習を実行する。想定所要時間は 30〜90 分。
- 評価: 学習中 `val_loss` / `val_cand_fire_acc` / `val_target_acc` / `val_ships_acc` を H1〜H3 と比較し、ローカル self-play 10 戦の挙動を baseline_v1 相手に確認する。
- replay 分析: 代表的な勝敗または loss seed を最大 2 試合 Markdown 化し、過剰発射 / no-op 偏り / target 選択の破綻を確認する。

## リスク / 既知の不確実性

- H1〜H3 がすべて 10 ep = 0/10 のため、dual 化だけでは挙動改善しない可能性が高い。
- 3-head と candidate head の supervision が競合すると、共通 backbone がどちらにも中途半端に最適化される可能性がある。
- 推論で candidate 側のみ使う場合、3-head は補助表現学習としてしか効かない。改善が見えた場合は H5 または follow-up で inference blend を切り分ける。
