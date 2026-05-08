# Imitation/case9 — 3 head families behavior comparison (iter2) RESULT

> 関連: iter2_plan.md / hypotheses.md
> commit: `ba6228c` / full RunPod rerun: `7c6e9c1`
> case: `case9_three_head` / `case9_candidate_ships` / `case9_template_ships`
> 開始: 2026-05-08
> 終了: 2026-05-08
> コスト: RunPod A5000 Secure 3本並列、概算 $0.810/h

## Summary

ユーザー指示により、iter2 は dual head ではなく **3-head / candidate×ships / template×ships** の 3 パターン比較へ書き換えた。`template_ships` head を新規実装し、targeted tests / ruff / mypy / 1 episode smoke は通過した。RunPod full 学習は `case9_template_ships` で起動を試みたが、SECURE / ALL / A5000 系すべてで pod availability により確保失敗。代替としてローカル小データ (train=2048 / val=512, 3 epochs, CPU) で 3 head family の E2E 学習・曲線可視化・3戦挙動比較を実施した。

## Numbers

### Verification

| check | result |
|---|---|
| targeted pytest | 26 passed |
| ruff | passed |
| mypy | passed, 35 files |
| import sanity | passed: `IL_CASE9_HEAD_MODE=template_ships` |
| 1 episode smoke | passed: `il_v9_template_ships` vs `baseline_v1`, timeout 0 |
| RunPod launch | failed before pod creation: no instances available |

### Local small-data train/val curves

| variant | epoch | val_total | main val acc | fire/action val acc | val_ships_acc | note |
|---|---:|---:|---:|---:|---:|---|
| 3-head | 0 | 2.9245 | val_target_acc=0.9390 | val_target_acc=0.9390 | 0.9205 | best metric epoch |
| 3-head | 2 | 1.5131 | val_target_acc=0.9390 | val_target_acc=0.9390 | 0.9088 | loss は低下 |
| candidate×ships | 0 | 5144744.1250 | val_cand_acc=0.7816 | val_cand_fire_acc=0.0167 | 0.9070 | best fire epoch |
| candidate×ships | 2 | 5144743.6250 | val_cand_acc=0.9662 | val_cand_fire_acc=0.0132 | 0.9070 | no-op寄り |
| template×ships | 0 | 2.3257 | val_template_acc=0.9390 | val_template_fire_acc=0.0000 | 0.9205 | best fire epoch (=0) |
| template×ships | 2 | 1.0414 | val_template_acc=0.9390 | val_template_fire_acc=0.0000 | 0.8401 | no-op完全寄り |

### Local 3-game behavior on tiny weights

| variant | vs baseline_v1 | avg_turns | timeout | note |
|---|---:|---:|---:|---|
| 3-head | 0 / 3 | 265.3 | 0 | 最も粘る |
| candidate×ships | 0 / 3 | 163.7 | 0 | 早く崩れる |
| template×ships | 0 / 3 | 118.3 | 0 | no-op偏り + 早期崩壊 |

## Diagnosis

- `template_ships` は構造上、sourceごとの no-op を template 最終classに内包できるため実装はシンプルになった。
- ただし小データでは `val_template_noop_acc=1.0` / `val_template_fire_acc=0.0` になり、**template分類が no-op 多数派に完全に寄った**。
- `candidate×ships` も `val_cand_noop_acc` が高く、`val_cand_fire_acc` は 0.0〜0.0167 に低迷。candidateも同じく no-op偏りが強い。
- 3-head は fired source の template CE を separate に学習するため、小データでは `val_target_acc=0.939` を維持し、3戦挙動でも平均turnが最長だった。
- ただし train=2048 / val=512 / 3戦のため、採否は project rule 通り **inconclusive**。


## Full RunPod rerun (2026-05-08)

3パターンを A5000 Secure pod 3本で並列実行。いずれも `train_process_exit_0` / `artifacts_uploaded` に到達し、学習とS3成果物アップロードは成功。後処理の `dvc add` のみ失敗したため、成果物はS3 fallbackで取得した。

| variant | run_id | best epoch | best val_total | 主指標 | ships acc | local 5 ep vs baseline_v1 | avg_turns |
|---|---|---:|---:|---:|---:|---:|---:|
| template×ships | `20260508-115939__feature-imitation-case9-head-comparison__7c6e9c1__seed0` | 14 | 0.8758 | val_template_fire_acc=0.2888 | 0.9156 | 0/5 | 144.2 |
| 3-head | `20260508-120122__feature-imitation-case9-head-comparison__7c6e9c1__seed0` | 13 | 1.0772 | val_from_acc=0.9415 / val_target_acc=0.9275 | 0.9146 | 0/5 | 160.6 |
| candidate×ships | `20260508-120124__feature-imitation-case9-head-comparison__7c6e9c1__seed0` | 8 | 246719.4473 | val_cand_fire_acc=0.2115 | 0.8294 | 0/5 | 289.6 |

Full分析は `iter2_analysis.md` に記録。

## Decision

- 採否: inconclusive
- full学習の暫定観測: **template×ships は学習指標が最良**、candidate×ships はローカル5戦で最長生存だが candidate loss が異常大。3-head は安定だが勝ちに届かない。
- 次の一手: candidate×ships の loss 異常原因を確認しつつ、template/candidate の発射率・no-op率・ship分布をログ化して、30戦程度の挙動確認に進む。

## Artifacts

- small curve PNG: `data/output/experiment/imitation/case9/head_family_small/small_learning_curves.png`
- small metrics CSV: `data/output/experiment/imitation/case9/head_family_small/small_epoch_metrics.csv`
- behavior logs: `data/output/experiment/imitation/case9/head_family_small/behavior_{variant}.log`
- small weights: `data/output/experiment/imitation/case9/head_family_small/weights_{variant}.pt`
- full curve PNG: `data/output/experiment/imitation/case9/head_family_full/full_learning_curves.png`
- full metrics CSV: `data/output/experiment/imitation/case9/head_family_full/full_epoch_metrics.csv`
- full analysis: `docs/experiment/imitation/20260506_case9_head_setting_optimization/iter2_analysis.md`
- plan: `docs/experiment/imitation/20260506_case9_head_setting_optimization/iter2_plan.md`
