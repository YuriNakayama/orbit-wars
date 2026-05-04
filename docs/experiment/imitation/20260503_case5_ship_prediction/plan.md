# imitation/case5 — Ship-Prediction Featurizer

> 作成日: 2026-05-03
> 関連:
> - `bot/pipeline/rulebase/case6/baseline/core/world_model.py:147` (`simulate_planet_timeline`) — ship-prediction の rulebase 実装
> - `bot/pipeline/rulebase/case6/baseline/missions/stay.py` — STAY 判定が timeline を消費する例
> - `bot/pipeline/imitation/case1/policy/featurizer.py:120` (`incoming[slot]` / `nearest_eta`) — imitation 側の現状 (集計のみ、時系列なし)
> スコープ: imitation/case5 を新設し、case1 featurizer に **per-planet × per-turn の ship 残存予測** を 4-6 列追加する

## 仮説 (Hypothesis)

Imitation BC は state→action mapping を学ぶが、現状 case1 の featurizer は
**敵 fleet の到着 ETA 別残存量** を持っていない (集計 incoming と nearest_eta のみ)。
rulebase case6 が STAY 判定で活用している `simulate_planet_timeline` 由来の
**per-planet timeline 特徴量** (turn ごとの owner / 残存 ships) を imitation 側に
取り込むと、特に **defensive hold** と **harass timing** の判断精度が上がるはず。

メカニズム:

- 現状 case1 は「いつ敵 fleet が来るか (ETA)」「総数どれくらい (incoming sum)」だけ知る
- rulebase case6 は「turn=1, 2, ... 30 までに planet 所有者がどう推移し ships が
  どう減るか」をシミュレーションして STAY 判定に使っている
- **このシミュレーション結果から 4-6 個の集約値** (loss_in_3turns, ttf=time-to-fall, min_owned,
  surplus_after_horizon, fall_predicted, keep_needed) を per-planet 特徴量として featurizer に
  追加すれば、policy は「この惑星はもうすぐ陥落する → 援軍 or 諦める」を学べる

成功指標 (300 ep self-play vs baseline_v1):

- **case1 baseline (ship-prediction なし)** の現状勝率を再測定 (既存 weights iter9 経路)
- **case5 (ship-prediction あり)** を training → 同条件で勝率比較
- 採否: case1 比 +5pp 以上の勝率改善 (300 ep) で採用、+0-5pp は非有意として保留、マイナスは破棄

評価メトリクス: ローカル self-play 1v1 vs `baseline_v1` 300 ep (memory `project_imitation_case1_phase3` に従い n<300 は信頼不可)。
**Kaggle publicScore は使わない**。

## 既存コードの現状 (from Step 1)

- **rulebase case6**: `bot/pipeline/rulebase/case6/baseline/core/world_model.py:147 simulate_planet_timeline` が
  per-planet 残存予測の本体。`{owner_at: {turn: int}, ships_at: {turn: float}, keep_needed: int, min_owned: int}` を返す。
  STAY mission (`missions/stay.py:_defense_risk_for_planet`) がこれを消費して防御 hold 判定。
- **imitation case1 featurizer**: `bot/pipeline/imitation/case1/policy/featurizer.py:75-115`。
  fleet ループで `_fleet_target_eta` を呼び `incoming[slot]` (my/enemy/neutral 別 ships 累積) と
  `nearest_eta[slot]` (最早到着 ETA) を集計するのみ。**時系列を持たない**。
- **case6 vs case4 比較**: `bot/pipeline/rulebase/case6/evaluation/compare_v4.py` が既存対戦スクリプト。
  case6 は STAY 経由で ship-prediction の効果を出すが、本 plan ではまずこの数値を再現してから
  imitation 取り込みの妥当性を判断する (Cycle 3 で実施)。

## スコープ (Scope)

### 変更ファイル

| Path | 変更内容 |
|------|----------|
| `bot/pipeline/imitation/case5/` (新規 case ディレクトリ) | case1 のコピーを起点に featurizer/preprocess/train を改造 |
| `bot/pipeline/imitation/case5/policy/featurizer.py` | 既存 11 列 → 15-17 列。新列: `loss_3turn`, `ttf_norm`, `min_owned_log`, `surplus_log`, `fall_predicted_flag`, `keep_needed_log` |
| `bot/pipeline/imitation/case5/policy/timeline.py` (新規) | `simulate_planet_timeline` を case5 内にコピー (`pipeline.rulebase.case6.*` への cross-case import 禁止のため、`.claude/rules/bot/pipeline.md`) |
| `bot/pipeline/imitation/case5/training/preprocess.py` | featurizer 拡張に伴い列定義更新 (parquet schema 互換性破壊 OK、case5 は新規) |
| `bot/pipeline/imitation/case5/training/train.py` | input_dim 変更のみ |
| `bot/src/dataset/selfplay/agents.py` | `il_v5: pipeline.imitation.case5.policy.agent:agent` を REGISTRY に追加 |
| `dvc.yaml` | `preprocess_imitation_case5 / train_imitation_case5 / eval_imitation_case5` stage を追加 |
| `bot/src/runpod_io/cli.py` の `CASE_DEFAULTS` | `case5` entry 追加 |
| 評価: `bot/pipeline/imitation/case5/evaluation/compare_baseline.py` | case1 と同形 |

### 変更なし

- case1 / case2 / case3 / case4 (imitation) — 触らない
- rulebase 全 case — 触らない (cross-case import 禁止に従い、`simulate_planet_timeline` は case5 内にコピー)
- Submit-shape (case1 が canonical なので影響なし)

## 実装ステップ

1. **Cycle 3 (このサイクル)**: rulebase case6 の `compare_v4.py` を 30 ep 走らせ、ship-prediction を
   STAY 経由で使う case6 が baseline_v4 にどれだけ効くかをまず計測 (baseline 取得)。
2. (Cycle 4 以降) `bot/pipeline/imitation/case5/` を `cp -r case1` で初期化、`__init__.py` 等の
   absolute import path を case5 へ書き換え。
3. `simulate_planet_timeline` を `pipeline/imitation/case5/policy/timeline.py` にコピー。
4. featurizer に timeline 由来 6 列を追加、PLANET_FEAT_DIM を 11→17 に変更、テストを書く。
5. preprocess に対応 (parquet schema は新規なので互換性問題なし)、training_dataset の input_dim 修正。
6. RunPod Step A → Step B で case5 学習を回す。
7. baseline_v1 vs il_v5 を 300 ep self-play、case1 vs il_v5 比較。
8. result.md / analysis.md。

## 検証方法 (Validation method)

- **ローカル**: pytest (case5 用 unit test) + `dev/test-bot`
- **リモート**: RunPod Step A (preprocess) → Step B (train, ~$1.0)
- **評価**: 300 ep self-play vs baseline_v1
- **採否しきい値**: case1 baseline 比 +5pp 以上の勝率

## Cycle 3 の縮小スコープ

このサイクルでは **plan.md + rulebase ship-prediction の baseline 取得** までに留める:

1. plan.md (this file) ✅
2. `compare_v4.py` を 30 ep 実行 → case6 vs baseline_v4 の勝率を測定
3. result.md に「case6 (ship-prediction 利用) は baseline_v4 にどれだけ効くか」を記録
4. 効くなら imitation case5 着手 (Cycle 4 以降)、効かないなら featurizer の設計を見直し
