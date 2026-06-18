# Phase 1 — PPO 実験条件の探索計画 (H1)

> 作成日: 2026-06-09
> 仮説 ID: H1 (hypotheses.md Phase 1)
> 関連: iter1_result.md (H0 scaffold), hypotheses.md
> スコープ: PPO のパラメータを 1 つずつ A/B で振り、Phase 2/3 が使う「凍結 config」を確定する

## このフェーズのゴール (3 条件を同時に満たす config を見つける)

| # | 条件 | 測定 | 合格ライン |
|---|---|---|---|
| ① | pool 内 per-iter 勝率が ~0.5 | 学習後半(pool 区間)の per-iter `win_rate` 平均 | 0.40–0.60 に収束 |
| ② | 固定相手 held-out 勝率が **0 付近から滑らかに増加** | held-out (in-JAX `baseline_jax_full`, 固定 seed) を **毎 iter** 記録した曲線 | 単調 or 概ね右肩上がり、序盤低→終盤高 |
| ③ | 全 JAX・GPU で **20 iter ≈ 30 分** | run の wall-clock | 20 iter ≤ 30 min (iter0 compile 込) |

3 つ揃った config を **`ppo_frozen.yaml`** として確定し、Phase 2 (V-MPO 無調整) / Phase 3 (V-MPO HP sweep) はこれを **algo / V-MPO HP 以外一切変更せず**流用する。

## 実験規律 (厳守)

- **1 run = 1 パラメータのみ変更**の A/B。基準 config (`ppo_base`) を起点に、下表の順で 1 軸ずつ振る。
- 採用した変更は次の run の基準 config に取り込み、累積的に良い config を育てる(逐次改善)。
- seed は固定 (seed=0)。分散が気になる軸のみ seed を 2-3 振って確認(③速度に響くので最小限)。
- held-out は **毎 iter (every=1)** 記録に変更して②の曲線を密に観察する(eval は in-JAX なので安価)。

## iter1 (H0) で判明した出発点の課題

- pool=[full, lite, self] で iter4 の pool 切替時に win 0.25 + entropy 48→14 急落。
- 観察: f_var が pool 勝率を 0.5 に保てていない(rulebase proxy 相手で 0.25 に張り付き)。
- 方針(ユーザー指定): **プールは full+lite+self を常に含む**まま、**f_var の優先度パラメータ
  (priority_p)** を調整して、(a) プール対戦勝率を ~0.5 に保ち、(b) 固定相手 (held-out full) との
  勝率が 0 から滑らかに増加する設定を見つける。f_var は弱い時は self/lite を、強くなれば full を
  選ぶよう勝率0.5付近を重み付けするので、priority_p の集中度が鍵。

## 振るパラメータと順序 (各 run の A/B 軸 + 判定)

> プール構成 (full+lite+self) は全 run 固定で探索しない。コード改修不要。

**プール構成は全 run で固定**: full + lite + self を常に含む(探索しない)。探索は優先度・curriculum・
最適化系のパラメータで行う。

| run | 変更軸 (1つ) | base → 候補 | 狙う条件 | 判定 |
|---|---|---|---|---|
| **R1** | f_var priority_p | `2.0` → `{1.0, 2.0, 4.0}`(f_var 固定, ema=0.7 固定) | ① pool 勝率~0.5 / ② held-out 0→滑らか増加 | pool 勝率が最も 0.5 近傍に張り付き、かつ held-out が滑らかに伸びる p を採用 |
| **R5** | iterations / episodes_per_iter | R1 採用 p で ②③のトレードオフ最終調整 | ②十分な伸び + ③≤30分 | 3 条件同時達成で凍結 config 確定 |

- **R1 が最重要(f_var priority_p)**: f_var の重み `(x(1-x))^p` の集中度を決め、目標①②を直接左右する。
- **R2/R3/R4 (switch_iter / priority_ema / entropy_coef) は実施しない**(ユーザー指定で省略)。
- R1 で採用した priority_p を base に取り込み、R5 で iterations/episodes を調整し 3 条件同時達成で確定。
- 各 run は原則 **20 iter 固定**で回し(③の基準)、条件①②を観察。R5 のみ iterations を最適化。

## 実装ステップ

1. コード改修は不要(プール構成は full+lite+self 固定で既存 `set_entries(include_full=True, include_lite=true)`
   のまま。`priority_p` は既に config 駆動)。
2. `configs/ppo_base.yaml` を新設(iter1 の h0_ppo_short をベースに、held-out every=1, iterations=20,
   algo=ppo, include_lite=true)。これが R1 の base。
3. 各 run は `configs/phase1_r{N}_*.yaml` を 1 軸だけ変えて作成(R1 は priority_p の 3 値)。
4. RunPod registry に `reinforce_case8_phase1` (oneshot, config 差し替え式) を用意 or 既存
   `reinforce_case8_kaggle_jax_train` の config を run ごとに差し替え。

## 実行方法 (oneshot, auto-recover)

- **oneshot 学習を既定**: `dev/runpod train <sha> --case case8`(interactive pod の無言終了を回避、
  auto-recover が効く)。各 run 後 pod 自動 destroy。
- 1 run ≈ 20 iter ≈ 8-30 分 + pod 起動 ~7 分。コスト ~$0.3-0.5/run。
- smoke (1-ep, foreground 4-game gate) を各 config で実行してから GPU 投入。

## 検証方法

### スキップする検証 (from hypotheses.md skip list)
- 本物 case8 (python_v8) との対戦は Phase 1 では行わない(in-loop held-out = in-JAX baseline_jax_full)
- replay 詳細分析なし / Kaggle publicScore 不使用 / n<300 で結論出さない
- 最終確認 (本物 case8 offline paired 300戦) は Phase 3 の最良 config 確定後

### 実施する検証
- ローカル: smoke (foreground 4-game) → `dev/test-bot`
- リモート: `dev/runpod train --case case8` (oneshot), 各 run 20 iter
- 評価: in-JAX held-out (baseline_jax_full, every=1) 曲線 + pool per-iter 勝率 + wall-clock
- 採否: 上記 3 条件 ①②③。1 run = 1 軸の A/B、改善軸を累積

## リスク / 既知の不確実性
- プールに固定強度 full/lite を含むため、弱い初期 agent では f_var が full/lite を「勝率0」と学習し
  ほぼ self/lite ばかり選ぶ可能性 → pool 勝率①が見かけ上 0.5 でも held-out②が伸びるかで「成長を伴う
  0.5」かを判別する(①②セットで評価)。
- priority_p を上げ過ぎると勝率0.5ちょうどの相手に集中し過ぎて学習が偏る / 下げ過ぎると uniform 化
  して 0.5 から外れる → {1.0, 2.0, 4.0} の 3 点で曲線の形を比較。
- entropy collapse が priority_p だけで解けない場合があるが、R4 (entropy_coef) は省略指定のため
  R1 の範囲で許容するか、必要なら別途検討(計画外)とする。
- ③ 30分: iter1 実測 ~23s/iter + held-out every=1 で +10-15s/iter ≈ 33-38s/iter、20iter≈11-13分
  + pod 起動 ~7分 で達成見込み。未達なら R5 の episodes_per_iter / held-out episodes で調整。
