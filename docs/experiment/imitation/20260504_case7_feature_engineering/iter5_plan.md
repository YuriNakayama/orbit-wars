# imitation/case7 iter5 — ships F1 0.93 の適宜性診断

> 作成日: 2026-05-06
> 関連:
> - `./iter4_result.md` — iter4 で ships macro F1 が 0.6353 → 0.9349 へ突如劇的改善
> - `./iter3_result.md` (iter4 方針セクション) — K4 aux multi-horizon global 4 列追加
> - `bot/pipeline/imitation/case7/policy/featurizer.py` — iter4 で 61/12 schema
>
> スコープ: **訓練ラン無し**、ローカルのみで iter4 weights に対して 3 種類の診断を実行。新規モデル学習は無し。

## 仮説 (Hypothesis、ただし反証目的)

iter4 で ships head が 0.6353 → **0.9349** (+0.29) と劇的改善したが、これは **真の改善ではなく leak / class imbalance / 偶然のいずれか** ではないか?
diagnose_weights の数値だけでは判別できない。3 つの独立な診断 (K4 ablation / permutation importance / class 分布) で **真因を特定する**。

メカニズム (疑念の根拠):
- 前 3 iter は ships F1 0.62〜0.64 で安定 → iter4 で +0.29 は通常の特徴量追加で出る幅を超えている
- mae_bucket が 0.20 → 0.04 という 4 倍精度向上はモデルが「ほぼ完璧に」buckets 推定できていることを意味し、**情報リークの典型サイン**
- K4 (aux global multi-horizon: my_ships_h5/h15 + my_prod_log + my_ships_ratio) は **future predicted ships** を含む → policy が action_N の結果を input から逆算できているか?

成功指標 (診断完了基準):
- 3 つの診断結果が一貫して「真の改善」を支持する場合: **iter4 採用維持**、iter5 は更なる feature engineering へ
- いずれかが「leak / imbalance」を示す場合: **iter4 を再評価**、原因に応じた修正案を立てる

採否ゲート (本 plan は診断、metric ゲートではなく **真因解明** が成功条件):
- 全 3 診断が完了し、ships F1 0.93 が真の効果か leak かを **断定できる** こと

## 既存コードの現状 (from Step 1)

- iter4 featurizer (PLANET_FEAT_DIM=61, GLOBAL_FEAT_DIM=12)
- K4 列 (global 8-11):
  - `aux_my_ships_h5 = log1p(my_total_ships + 5 * my_total_prod)`
  - `aux_my_ships_h15 = log1p(my_total_ships + 15 * my_total_prod)`
  - `aux_my_prod_log = log1p(my_total_prod)`
  - `aux_ships_ratio = my_total_ships / (my_total_ships + enemy_total_ships)`
- iter4 weights は `bot/pipeline/imitation/case7/policy/weights.pt` で canonical
- val parquet: `data/mart/imitation/case7/val.parquet` (38,278 frames、iter4 schema)
- preprocess.py 内で ships バケットを `_ships_bucket()` で計算 (4 buckets: 25%/50%/75%/100%)

## スコープ (Scope)

### 変更ファイル

**コード変更なし**。診断スクリプトのみを `/tmp/` か `data/output/experiment/` に出力。

| Path | 内容 |
|------|------|
| `data/output/experiment/imitation_case7_iter4_grouped_importance.json` | iter3 と同じ機能別 group permutation importance を iter4 weights で実行 |
| `data/output/experiment/imitation_case7_iter4_k4_ablation.json` | K4 列だけを zero-out して val metrics を測る |
| `data/output/experiment/imitation_case7_iter4_ships_class_distribution.json` | val_ds で ships_per_src の class distribution + confusion matrix |
| `docs/experiment/imitation/20260504_case7_feature_engineering/iter5_result.md` | 3 診断の結論 + iter4 採用判定の最終確認 |

## 実装ステップ (Implementation outline)

### A: K4 ablation (1 列ずつ zero-out)

`/tmp/iter5_k4_ablation.py`:

1. iter4 weights をロード
2. val.parquet から features を読み込み
3. 4 種類の zero-out variants を作成:
   - `aux_my_ships_h5_zero`: global col 8 = 0
   - `aux_my_ships_h15_zero`: global col 9 = 0
   - `aux_my_prod_log_zero`: global col 10 = 0
   - `aux_ships_ratio_zero`: global col 11 = 0
   - `K4_all_zero`: global col 8-11 全て 0
4. 各 variant で diagnose_weights 相当の metrics を計算、baseline (iter4 元値) との Δ を出力
5. **Δ ships_F1** が大きく落ちる variant が ships head の真の貢献源

### B: Permutation importance を iter4 で再計算

`/tmp/iter5_grouped_importance.py` (iter3 で使った同じスクリプトを iter4 weights / parquet に向ける):

1. iter4 dim layout (planet 0-60, global 0-11) で機能別 group を再定義
2. iter3 と同じ 11 group + 5 global group で permutation importance
3. iter3 結果と並べて diff: K4 が global 8-11 で large negative Δ を示すか?

### C: ships class 分布 + confusion matrix

`/tmp/iter5_ships_distribution.py`:

1. val.parquet から `ships_per_src != -1` の フレームを抽出 (= 発射された source slot)
2. class 分布: bucket 0/1/2/3 のそれぞれの頻度
3. iter4 model で予測し confusion matrix
4. 「**majority class 予測でどこまで F1 が出るか**」を per-class accuracy + macro F1 のベースラインとして算出
5. もし bucket 3 (=100%発射) が val の 70%以上を占め、modelが全部 bucket 3 と予測しても F1 ~0.85 まで行くなら **iter4 の 0.93 はあまり驚異ではない** という解釈

### 検証ステップ

```bash
# A
uv --project bot run python /tmp/iter5_k4_ablation.py
# B
uv --project bot run python /tmp/iter5_grouped_importance.py  # iter4 weights/parquet バージョン
# C
uv --project bot run python /tmp/iter5_ships_distribution.py
```

各スクリプト ~3-5 分。総計 ~15 分でローカル完結。

## 想定される 3 つの結論パス

| パス | A 結果 | B 結果 | C 結果 | 結論 |
|------|--------|--------|--------|------|
| **真改善** | K4 各列 -0.05〜-0.10 ships_F1 落ちる | iter4 H4_aux_global の sum &#124;Δ&#124; が大きい (~0.3+) | bucket 3 の頻度 ~50%、majority predict で F1 0.5 程度 → 0.93 は本物 | **iter4 採用維持**、iter5 は新規 feature 追加へ |
| **K4 leak** | aux_ships_h5/h15 で ships_F1 が **iter1 レベル (0.64) 近くまで** 落ちる | K4 が H1 base 並みに巨大 importance | bucket 分布 ばらける | **K4 修正必要**: future predicted を input から外し、現在 ships sum のみにする |
| **Class imbalance** | K4 ablation で ships_F1 微減のみ | importance は均等 | bucket 3 が val の 80%+ を占める | **見かけ上の F1**: per-class metric や Cohen's kappa で再評価 |

## 検証方法 (Validation method)

### ローカル

```bash
dev/test-bot  # 既存テスト維持確認
uv --project bot run python /tmp/iter5_k4_ablation.py
uv --project bot run python /tmp/iter5_grouped_importance.py
uv --project bot run python /tmp/iter5_ships_distribution.py
```

### リモート

**RunPod 不要**。本 plan のスコープは診断のみ、再学習は含まない。

### 評価メトリクス

- 主要: ships macro F1 の Δ vs baseline (= iter4 元値 0.9349)
- 副次: ships acc, mae_bucket, per-class recall (bucket 0/1/2/3)

### 採否しきい値

- 「真の改善」と確定 → iter4 維持、iter5 を新 feature サイクルへ
- 「leak / imbalance」と判明 → 即修正の iter6 plan を立てる

## リスク / 想定失敗モード

1. **3 診断の結果が conflicting**: A/B が支持・C が反対するなど → 追加診断 (例: K4 を別 normalize で再学習) に進む
2. **K4 leak だが他 head には true 改善**: K4 のみ削除して再学習する partial fix が必要
3. **class imbalance による見かけの 0.93**: F1 に依存せず Cohen's kappa / balanced accuracy で再評価。Stage 2 self-play で結局問われる

## Stop conditions

以下を満たしたら本 plan のスコープは完了:

- [ ] A K4 ablation 結果が JSON で記録された
- [ ] B permutation importance (iter4) 結果が JSON で記録された
- [ ] C ships class 分布 + confusion matrix 結果が JSON で記録された
- [ ] iter5_result.md に 3 診断の結論 + iter4 採用判定の最終確認が書かれた

## 参考

- `iter3_result.md` permutation importance 章: 同じ手法を iter4 で再適用
- iter1〜3 の ships F1 推移 (0.6425 → 0.6292 → 0.6353) と iter4 の 0.9349 の不連続性
- BC literature: K4 のような "current state aggregate" は通常 leak しないが、`aux_my_ships_ratio` が target_per_src と相関する経路があれば leak 候補
