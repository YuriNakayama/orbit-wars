# case7 「ルールベースに勝つ」ループ — iter05 PLAN

時刻: 2026-06-03 03:49 (cron tick 6 続き)

## 前提 (確定)
- 5/5 variant が vs baseline_v1 = 0/10。parity test 81 件 pass = 変換健全。
- ボトルネックは compute scale。CPU 10-30 iter では届かない。
- memory: case1 は RunPod GPU 300 iter で初めて self-play 0.50。

## 方針変更 (04:02): GPU 前に「本物 v1 で直接学習」を試す (小規模・無料・本筋)
ユーザーの「小規模」志向 + train/eval ギャップが 0/10 の真因の可能性を踏まえ、
**GPU スケール前に opponent=python_v1 (本物 baseline_v1, host callback) で直接学習**を試す。
- これまで全 run は JAX 近似 rule (lite/full) 相手 → 本物 v1 を一度も見ていない。
- memory `case6_pool_v1_rejected` は python_v1 で「reward sparse 勾配消失」と棄却したが、
  **BC warm-start + winnable curriculum (noop warmup) と組み合わせていなかった**。今回は併用。
- 起動: `loop_iter05_vs_real_v1.yaml` (BC warmstart, curriculum noop(2)→python_v1,
  kl0.2, 12 iter ep6 h200)。python_v1 は callback で重いので small に。
- 完走 → 10戦 vs baseline_v1。これで動けば「本物相手の経験」が鍵だった証明。
  動かなければ → 下記 GPU スケールへ。

## (保留) iter06+ 方針: GPU スケールアップ (loop の GPU 許可方針)
RunPod GPU で iter 数を 1 桁増やす (100-200 iter)。ただしコスト/在庫に注意:
- memory `project_reinforce_self_snapshot_cost`: self_snapshot/PFSP は rollout 2倍重、
  case6 H1 が A100 で $7 超過 (cap $1.5 の 4.7倍)。→ **iterations 抑制 + 軽い opponent**。
- memory `project_runpod_3090_4090_stockout`: 3090/4090 は枯渇あり。pod 未作成=課金0。

## GPU run 設計 (コスト最小)
- BC warm-start ON + ratio shaping + curriculum noop → self_snapshot (lite/full は
  host callback で GPU でも遅い → 少量 or 後半のみ)。
- iterations 100-150、episodes 16、horizon 500 (GPU は CPU 比 17-18x なので 100 iter ~25min)。
- config: 既存 `kaggle_jax_train*.yaml` を base に上記反映 (case7 用に新規 yaml)。

## 起動手順 (次 tick で実施)
1. case7 一式 + loop config を commit (RunPod は commit SHA から学習)。
   ※ docs/config/コード変更が未 commit。`git add` → commit (push は branch へ)。
2. `dev/runpod train <sha> --case case7 --watch` で起動 (cost cap 既定 $1.5)。
   在庫無ければ 40min backoff (memory)。est cost を確認してから本起動。
3. 完了 → `dev/runpod pull` → best.pt → jax_to_torch → 10戦 vs baseline_v1。

## 留意 / 判断
- GPU は許可済 (認証不要) だが課金発生 → 1 run のコスト見積りを log して起動。
- それでも 0/10 が続くなら、問題は「この model 容量/特徴量では v1 に勝てない」可能性 →
  imitation 側 (case9) の強化や、rule を教師にした別アプローチへ範囲を広げる検討。
- GPU 起動は重い一歩なので、次 tick 冒頭で現状を要約しユーザーに一報してから実行。
