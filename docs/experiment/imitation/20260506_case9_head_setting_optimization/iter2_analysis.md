# Imitation/case9 — 3 head families behavior comparison (iter2) ANALYSIS

> 関連: `iter2_plan.md` / `iter2_result.md` / `hypotheses.md`
> 分析日: 2026-05-08
> commit: `7c6e9c1`
> 対象: 3-head / candidate×ships / template×ships

## 結論サマリ

3本とも RunPod full 学習は `train_process_exit_0` で成功し、成果物もS3へアップロード済み。後処理の `dvc add` のみ失敗したため、成果物は `dev/runpod pull --from s3` で取得した。

学習指標では **template×ships が最も低い val_total と最高の fire acc を示した**。ただしローカル5戦では3パターンすべて `baseline_v1` に 0/5。挙動上は candidate×ships が最も長く粘ったが、学習損失が異常に大きく candidate分類が崩れているため、採否は project rule 通り **inconclusive**。

## RunPod full 学習結果

| variant | run_id | best epoch | best val_total | 主指標 | ships acc | early stop | 備考 |
|---|---|---:|---:|---:|---:|---|---|
| template×ships | `20260508-115939__feature-imitation-case9-head-comparison__7c6e9c1__seed0` | 14 | 0.8758 | val_template_fire_acc=0.2888 | 0.9156 | yes | fire acc は3候補中最高 |
| 3-head | `20260508-120122__feature-imitation-case9-head-comparison__7c6e9c1__seed0` | 13 | 1.0772 | val_from_acc=0.9415 / val_target_acc=0.9275 | 0.9146 | yes | 安定だが fire recall 系の直接指標なし |
| candidate×ships | `20260508-120124__feature-imitation-case9-head-comparison__7c6e9c1__seed0` | 8 | 246719.4473 | val_cand_fire_acc=0.2115 | 0.8294 | yes | candidate loss が異常大 |

## ローカル対戦挙動確認

各best.ptを一時的に `bot/pipeline/imitation/case9/policy/weights.pt` に差し替え、`baseline_v1` 相手に5戦実行した。

| variant | agent | vs baseline_v1 | avg_turns | timeout | 解釈 |
|---|---|---:|---:|---:|---|
| template×ships | `il_v9_template_ships` | 0/5 | 144.2 | 0 | 学習指標は良いが実戦では早めに崩れる |
| 3-head | `il_v9_three_head` | 0/5 | 160.6 | 0 | templateより少し粘るが勝てない |
| candidate×ships | `il_v9_candidate_ships` | 0/5 | 289.6 | 0 | 最長生存。ただし candidate head 自体は学習不安定 |

## 診断

1. **template×ships は学習上は最有望**
   - `val_total=0.8758` と `val_template_fire_acc=0.2888` が3候補中最良。
   - 小データ時の `fire_acc=0.0` 崩壊からは改善しており、full data では no-op 完全偏りではない。
   - ただし `val_template_noop_acc=0.9943` と no-op 優勢は残っている。

2. **candidate×ships は損失スケール/ラベル対応に問題がある可能性が高い**
   - `val_total=246719`、`avg_grad_norm_pre_clip` も非常に大きい。
   - `val_cand_acc=0.0673` に対し `val_cand_fire_acc=0.2115` なので、全体分類としては大きく崩れている。
   - ローカルでは最長生存だが、これは「良い攻撃」ではなく消極/遅延挙動の可能性がある。

3. **3-head は安定だが勝ち筋に届いていない**
   - `val_from_acc=0.9415`, `val_target_acc=0.9275`, `val_ships_acc=0.9146` は安定。
   - 一方、ローカル5戦では 0/5 で、baseline_v1 を破るには行動選択・発射量・no-op/fire balance の追加改善が必要。

## 採否

- H4 採否: **inconclusive**
- 暫定順位:
  1. 学習指標: template×ships > 3-head >>> candidate×ships
  2. 5戦挙動: candidate×ships > 3-head > template×ships
- ただし n=5 かつ baseline_v1 への 0勝のため、結論は出さない。

## 次に見るべき指標

- template/candidate の `fire_acc` だけでなく、実際の発射率・発射ship分布・no-op率
- candidate×ships の candidate loss 異常値の原因: ラベルindex範囲、mask、ignore_index、CE入力スケール
- 5戦ではなく最低30戦の挙動比較。ただし採否判定は n<300 では固定で inconclusive

## Artifacts

- full curve PNG: `data/output/experiment/imitation/case9/head_family_full/full_learning_curves.png`
- full metrics CSV: `data/output/experiment/imitation/case9/head_family_full/full_epoch_metrics.csv`
- summary CSV: `data/output/experiment/imitation/case9/head_family_full/full_summary.csv`
- behavior logs: `data/output/experiment/imitation/case9/head_family_full/behavior_{variant}.log`
- raw behavior summary: `data/output/experiment/imitation/case9/head_family_full/behavior_summary_raw.json`
