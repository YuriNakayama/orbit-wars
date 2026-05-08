# Imitation/case9 — 3 head families behavior comparison (iter2)

> 作成日: 2026-05-08
> 仮説 ID: H4
> hypotheses.md: docs/experiment/imitation/20260506_case9_head_setting_optimization/hypotheses.md
> 関連: docs/experiment/imitation/20260506_case9_head_setting_optimization/iter1_result.md
> スコープ: `bot/pipeline/imitation/case9/` で head family を 3 パターンに揃え、学習時 train/val 精度推移とローカル対戦挙動を比較する

## 仮説 (Hypothesis)

H4: 3-head / candidate×ship数予測 / template×ship数予測の 3 パターンは、同じ Set Transformer backbone と同じ case9 データでも、**no-op/fire の表現方法**が異なるため学習曲線と対戦挙動が分かれる。

- **3-head**: `from` で fire/no-fire を独立判定し、fired source だけ `template` と `ships` を学習する。
- **candidate×ships**: candidate slot 0 を no-op、slot 1..K を target 候補として直接分類し、別 head で ships bucket を学習する。
- **template×ships**: template の最終 class を no-op とし、source ごとに `template incl no-op` と `ships bucket` を学習する。`from` head を持たず、no-op/fire は template 分類に内包する。

この比較で明らかにすることは、**candidate 空間での直接候補選択**と **template 空間での抽象行動選択**のどちらが、学習精度・fire精度・ローカル挙動の観点で安定するかである。

## 固定軸

| 軸 | 固定値 |
|---|---|
| data | `data/mart/imitation/case9/train.parquet`, `val.parquet` |
| featurizer | case9 既存 PLANET=41 / GLOBAL=20 |
| backbone | Set Transformer hidden=128, ISAB×3, PMA |
| optimizer | AdamW, lr=1e-3, cosine warmup, batch_size=128 |
| 評価相手 | `baseline_v1` |
| 対戦数 | local 10 ep smoke/挙動確認のみ。n<300 なので採否は inconclusive 固定 |

## 比較対象

| variant | head_mode | 学習ターゲット | best metric | 推論 |
|---|---|---|---|---|
| A: 3-head | `three_head` | from BCE + template CE + ships CE | `val_target_acc` | from sigmoid → template → ships |
| B: candidate×ships | `candidate_ships` | candidate slot CE/focal + ships CE | `val_cand_fire_acc` | candidate argmax, slot0=no-op → ships |
| C: template×ships | `template_ships` | template incl no-op CE + ships CE | `val_template_fire_acc` | template argmax, last=no-op → ships |

## 実装ステップ

1. `policy/heads/template_ships.py` を追加し、template incl no-op + ships bucket の head を実装する。
2. `Case9Policy` / `PolicyOutput` / decoder / loss dispatch / train logging に `head_mode="template_ships"` を追加する。
3. `configs/il_case9_template_ships.yaml` を追加し、既存 2 variant と同条件で学習できるようにする。
4. `dev/runpod` case registry と selfplay agent registry に `case9_template_ships` / `il_v9_template_ships` を追加する。
5. targeted tests と 1 episode smoke を通す。
6. 既存 run の A/B と新規 C を同じ表にまとめ、必要に応じて A/B も同 SHA で再学習して完全比較にする。

## 検証方法

### 実施する検証

- targeted tests: `uv run --directory bot pytest tests/pipeline/imitation/case9 -q --no-header -x`
- import sanity: `IL_CASE9_HEAD_MODE=template_ships` で agent import
- 1 episode smoke: `il_v9_template_ships` vs `baseline_v1`
- 学習: RunPod で `case9_template_ships` を実行。必要なら `case9_three_head` / `case9_candidate_ships` も同一 commit で再実行する。
- 可視化: 各 run の `history.jsonl` / `train.log` から `train_*_acc` / `val_*_acc` / loss 推移を PNG/CSV 化する。
- ローカル対戦: 各 variant の best.pt を canonical weights に差し替え、`baseline_v1` 相手に 10 ep 実行し、勝敗・平均turn・action発火傾向を比較する。

### スキップする検証

- Kaggle publicScore は引用しない。
- skill rating は使わない。
- 300 対戦は行わない。
- n<300 のローカル対戦だけでは採否を確定しない。

## 成功/観測基準

| 観測 | 意味 |
|---|---|
| `template_ships` の `val_template_fire_acc` が 3-head/candidate系より高い | no-opをtemplate分類に内包する方が fire判断に有利 |
| `candidate_ships` の `val_cand_fire_acc` が高いが対戦挙動が悪い | 候補分類精度と実行policy品質の乖離 |
| `3-head` が平均turnで粘る | from head の保守的fire判断が防御寄りに働く可能性 |
| 3者とも10戦0勝 | 現行backbone/headだけでは baseline_v1 を超える証拠なし。n<300なので inconclusive |

## リスク

- A/B は既存 run を流用すると commit差が残る。完全比較が必要なら再学習する。
- `template_ships` は no-op 多数派に寄り、fire recall が低くなる可能性がある。
- agent loader は現状 `weights.pt` を読むため、ローカル対戦時は variant ごとに一時的に `weights.pt` を差し替える必要がある。
