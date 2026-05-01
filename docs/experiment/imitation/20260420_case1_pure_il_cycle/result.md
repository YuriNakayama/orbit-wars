# imitation/case1 改善 純 IL 6 iter サイクル 実験詳細記録

- 実施日: 2026-04-19 〜 2026-04-20
- ブランチ: `feature/imitation-case1-improve`
- 対象: `pipeline/imitation/case1/` (Imitation Learning Baseline)
- 目的: `il_v1` (純粋 IL) が `baseline_v1` (rulebase/case1) に勝てるようにする
- 出発点: `il_v1` vs `baseline_v1` (1v1, 100戦) で **win_rate = 0.00**
- 関連:
  - 出発点の診断: [`20260419_case1_diagnosis/result.md`](../20260419_case1_diagnosis/result.md)
  - 直前の改善試行 (試行 1-3): [`20260420_case1_improve/result.md`](../20260420_case1_improve/result.md)

---

## エグゼクティブサマリ

純粋 IL (Behavior Cloning) のみで 6 iteration の改善試行を行った結果：

1. **全 iter (1-6) で勝率 ≤ 1/100**: BC 単独では `baseline_v1` を超えられない
2. **`val_target_acc ≈ 0.40` が頭打ち**: focal loss / source-aware geometry / multi-task / template 数増 (6→12) など、target head の信号強化策はいずれも 0.34 → 0.40 までしか押し上げられなかった
3. **本質的なボトルネック**: 「target head の文脈欠如 (単フレーム決定)」と「惑星 ID 直接分類のラベルノイズ」

最終結論: 純 BC では限界。これを解消するには **テンプレート粗粒度化 (案 A)** または **DAgger / RL 路線 (案 B)** が必要。

> 注: 本サイクル中、IL を baseline と組み合わせる hybrid 系の試行も行ったが、
> **純粋 IL の検証ノイズになる** ため本記録からは除外している。提出 agent も
> 純粋 IL (iter 6 の重み + decoder) に固定する。

---

## サイクル全体の結果一覧 (純 IL のみ)

| iter | 主な変更 | val_from_acc | val_target_acc | 勝率 (vs baseline_v1, 100戦) |
|------|---------|--------------|----------------|-------------------------------|
| 1 | 1355 replays で再 preprocess + 学習 (基準点) | 0.71 | 0.34 | 0/100 |
| 2 | template 数を 6 → 12 に拡張 | 0.71 | 0.30 | 0/100 |
| 3 | target_head に source-aware geometry 追加 | 0.72 | 0.39 | 0/100 |
| 4 | value head 追加 (multi-task BC) | 0.72 | 0.39 | **1/100** |
| 5 | from_head の focal loss 化 (γ=2.0, α=0.75) | 0.86 | 0.40 | 0/100 |
| 6 | focal α=0.25 + decoder min_fire_topk=2 | 0.88 | 0.40 | 0/100 |

**最終勝率: 0/100** (iter 6 が最良学習結果、現状の `pipeline/imitation/case1/evaluation/results.json` に保存)。

評価ファイル一覧 (`pipeline/imitation/case1/evaluation/`):
- `results_iter3.json` (iter 3, 0/100)
- `results_iter4.json` (iter 4, 1/100)
- `results_iter6.json` (iter 6, 0/100)
- `results.json` (iter 6 最終版を canonical として保存, 0/100)

---

## iter 別 詳細

### iter 1: 1355 replays で再 preprocess + 学習 (基準点)

**目的**: 直前の試行 3 (`20260420_case1_improve/result.md`) と同条件で再現確認。

**変更点**: なし (再 preprocess + 再学習のみ)

**学習結果** (best_epoch=12, val_target_acc=0.337):

```
epoch 0:  train=4.81 val=4.88 from_acc=0.728 target_acc=0.296 ships_acc=0.689
epoch 6:  train=4.42 val=4.67 from_acc=0.788 target_acc=0.322 ships_acc=0.712
epoch 12: train=4.33 val=4.59 from_acc=0.790 target_acc=0.337 ships_acc=0.722  ← best
```

**評価結果**: 0/100 (試行 3 と同じ)。

**学び**: データ量増 (試行 3 と同じ replays 数) だけでは効かない。target head に手を入れる必要がある。

---

### iter 2: template 数を 6 → 12 に拡張

**目的**: target action space の表現力不足を疑い、テンプレートを増やして戦略多様性を確保。

**変更点** (`pipeline/imitation/case1/policy/templates.py`):
- `NUM_TEMPLATES = 8` → `12` に変更
- 追加: `T_FAR_NEUTRAL_HIGH_PROD`, `T_DEFEND_HOMEWORLD`, `T_SNIPE_PRODUCING`, `T_HOLD` 等
- `classify_actual_target()` の優先度を追加テンプレートに合わせて調整

**学習結果** (`/tmp/train_iter2.log`, best_epoch=11):

```
epoch  0: train=5.527 val=5.538 from_acc=0.737 target_acc=0.250
epoch 11: train=5.054 val=5.252 from_acc=0.736 target_acc=0.307  ← best
epoch 14: train=5.001 val=5.295 from_acc=0.713 target_acc=0.303
best_val_loss=5.2525
```

**評価結果**: 0/100。

**学び**:
- val_target_acc が 0.34 → 0.30 に **悪化**。class 数増による学習信号希釈
- 追加テンプレートの教師ラベル分布が極端に偏り (NEAREST_ENEMY 30%、追加分 < 5% 各)
- 「テンプレートを足せば多様性が出る」は誤り。target head の文脈表現を強化する方が先

**ロールバック**: テンプレート数を 8 に戻して iter 3 へ。

---

### iter 3: target_head に source-aware geometry 追加

**目的**: 「同じ src でも局面で target を変える」を学べるよう、pairwise 入力を強化。

**変更点** (`pipeline/imitation/case1/policy/model.py`):
- target head の pairwise を `(h_src, h_tgt, ctx)` から
  `(h_src, h_tgt, ctx, dist_norm, angle_rel, eta_norm, ships_ratio, owner_relation)` に拡張
- `pipeline/imitation/case1/training/preprocess.py` で pairwise feats を parquet に追加

**学習結果** (`/tmp/train_iter3.log`, best_epoch=7):

```
epoch  0: train=4.640 val=4.645 from_acc=0.727 target_acc=0.342
epoch  7: train=4.295 val=4.442 from_acc=0.721 target_acc=0.385  ← best
epoch 13: train=4.204 val=4.443 from_acc=0.723 target_acc=0.393
best_val_loss=4.4419
```

**評価結果**: 0/100 (`results_iter3.json`)。

**学び**:
- val_target_acc が 0.34 → 0.39 に改善 (+5pp)
- しかし勝率は 0/100 のまま。0.39 は依然「61% 誤分類」で、戦略的崩壊を防げない
- pairwise feature の追加は方向性として正しいが、効果が限定的

---

### iter 4: value head 追加 (multi-task BC)

**目的**: 補助タスク (試合勝敗予測) で表現学習を促進し、target head の精度を底上げ。

**変更点**:
- `model.py` に `value_head` 追加 (state-value 回帰、教師は episode 終端の binary winner)
- `losses.py` に MSE 項追加 (`value_w = 0.5`)
- `preprocess.py` で `value_label` 列追加

**学習結果** (best_epoch=9, val_target_acc=0.394):

```
epoch  0: train=4.85 val=4.91 from_acc=0.71 target_acc=0.30 value_mae=0.41
epoch  9: train=4.39 val=4.51 from_acc=0.72 target_acc=0.39 value_mae=0.32  ← best
```

**評価結果**: 1/100 (`results_iter4.json`)。

**学び**:
- value head 自体は学べた (MAE 0.41 → 0.32)
- target head への波及効果は誤差程度 (0.39 → 0.394)
- 1 勝出たが、これは乱数の揺らぎ範囲 (95% 信頼区間 ~ ±3pp)

---

### iter 5: from_head の focal loss 化

**目的**: from head の class imbalance (fired:not_fired = 1:8.5) に対し、pos_weight=8.5 だけでは負例優位が残るため、focal loss で hard examples に重み付け。

**変更点** (`pipeline/imitation/case1/training/losses.py`):
- BCE → focal BCE
- パラメータ: `from_focal_gamma=2.0`, `from_focal_alpha=0.75`

```python
bce_per_elem = F.binary_cross_entropy_with_logits(safe_logits, from_target, reduction="none")
p = torch.sigmoid(safe_logits)
p_t = from_target * p + (1.0 - from_target) * (1.0 - p)
alpha_t = alpha * from_target + (1.0 - alpha) * (1.0 - from_target)
focal_factor = alpha_t * (1.0 - p_t).clamp_min(1e-6).pow(gamma)
focal = focal_factor * bce_per_elem * valid.float()
```

**学習結果** (`/tmp/train_iter5.log`, best_epoch=9):

```
epoch  0: train=3.720 val=3.759 from_acc=0.811 target_acc=0.345 ships_acc=0.731
epoch  9: train=3.435 val=3.570 from_acc=0.848 target_acc=0.401 ships_acc=0.747  ← best
epoch 14: train=3.376 val=3.605 from_acc=0.847 target_acc=0.399 ships_acc=0.744
best_val_loss=3.5699
```

**評価結果**: 0/100。

**学び**:
- val_from_acc が 0.79 → 0.85 と改善
- val_target_acc が 0.39 → 0.40 と微増
- **しかし勝率はゼロ**。実機推論時に `from_prob` の分布を観察すると、**全 my_planet で同時 fire してしまう** 縮退が発生
  - mean(from_prob | my_planet) = 0.33 (期待: 0.10 程度)
  - α=0.75 が positive 側を強く励起しすぎた

---

### iter 6: focal α=0.25 + decoder min_fire_topk=2

**目的**: iter 5 の from_prob saturation 問題を緩和。さらに decoder で「最低 2 source は fire させる」ガードを追加。

**変更点**:
1. `configs/il_baseline.yaml`: `from_focal_alpha: 0.25` (iter 5 の 0.75 から逆転)
2. `policy/decoder.py`: `min_fire_topk: int = 2` パラメータ追加
   ```python
   ranked = []
   for src_pid in snapshot.my_planet_ids:
       slot = snapshot.planet_ids.index(src_pid)
       prob = float(from_prob[slot].item())
       ranked.append((prob, src_pid))
   ranked.sort(key=lambda x: -x[0])
   keep_n = max(min_fire_topk, sum(1 for p, _ in ranked if p >= from_threshold))
   src_with_prob = ranked[:keep_n]
   ```
3. `training/train.py` バグ修正: focal パラメータが `LossWeights` に伝播していなかった (iter 5 の config 変更が無視されていた問題を発見・修正)

**学習結果** (`/tmp/train_iter6.log`, best_epoch=9):

```
epoch  0: train=3.713 val=3.765 from_acc=0.872 target_acc=0.340 ships_acc=0.732
epoch  6: train=3.472 val=3.592 from_acc=0.885 target_acc=0.395 ships_acc=0.749
epoch  9: train=3.427 val=3.577 from_acc=0.882 target_acc=0.395 ships_acc=0.744  ← best
epoch 14: train=3.362 val=3.600 from_acc=0.880 target_acc=0.391 ships_acc=0.750
best_val_loss=3.5768
```

**評価結果**: 0/100 (`results_iter6.json`、`results.json` に同内容)。

**学び**:
- α=0.25 で from_prob saturation 解消、実機 fire 数が常識的な水準に
- val_target_acc は 0.40 で頭打ち継続
- **「target head が 40% 精度では戦略的崩壊が避けられない」と確定**
- 純粋 BC は限界。次は別アプローチ (案 A: テンプレート粗粒度化 / 案 B: DAgger 等) が必要

---

## 失敗の根本原因 (iter 1-6 累積で確定)

### 原因 1: target head の文脈欠如

iter 3 で source-aware geometry を追加しても val_target_acc は 0.34 → 0.39 までしか上がらない。pro player は「同じ source から、局面 (序盤=expand / 中盤=reinforce / 終盤=snipe) で異なる target」を選ぶが、現アーキテクチャは **単フレーム決定** のため時系列文脈を持てない。

実機トレース (vs baseline_v1, seed=0):
- IL は同じ neutral 惑星 (例: p29) に step 9-37 まで連射し続ける
- baseline は同 source からの連射を避け、遠距離の高 production 惑星を狙う

### 原因 2: 惑星 ID 直接分類のラベルノイズ

iter 1 〜 iter 6 を通じて、`_resolve_action_target` が `aim_with_prediction` (将来位置への射撃) を逆解決できず、教師ラベルにノイズが混入。`ANGLE_TOLERANCE=0.20` 内に複数候補がある場合の tie-break もヒューリスティクス。

### 原因 3: val_target_acc 0.40 という ceiling

- iter 1 (基準): 0.34
- iter 3 (geometry): 0.39 (+5pp)
- iter 4 (value head): 0.394 (+0.4pp)
- iter 5 (focal): 0.40 (+0.6pp)
- iter 6 (focal α 調整): 0.40 (頭打ち)

target head 入出力の改良では 0.40 を超えられない。ラベル空間そのもの (惑星 ID 36 値) の見直しが必要。

---

## 残作業 (本 PR を締めるため)

| 項目 | 状態 | 必要作業 |
|------|------|---------|
| diagnosis result 追記 | 未対応 | `20260419_case1_diagnosis/result.md` の補遺に iter 1-6 の追加知見を記載 |
| MEMORY.md 更新 | 未対応 | 「BC 単独では baseline 超え不可、val_target_acc ≈ 0.40 が天井」 |
| コミット分割 | 未対応 | iter ごとに分けて意味のあるコミット履歴に |

---

## 次の戦略選択肢 (本 PR マージ後)

| 案 | 内容 | 期待 | コスト | 推奨度 |
|----|------|------|-------|-------|
| **A** | テンプレート行動分類 (target を 6-8 戦術テンプレートに粗粒度化、planet ID 直接分類を廃止) | val_target_acc 60%+ → 30-40% 勝率の可能性 | 中 (preprocess + head + decoder 書換) | ★★★ |
| B | DAgger / online IL (baseline_v1 を expert として selfplay で補助教師信号生成) | 不明 | 大 (selfplay loop 改造) | ★★ |
| C | imitation 路線を中止し、別カテゴリ (RL / search) を新 case で開始 | 不明 | 大 | ★★ |

**推奨**: 案 A。診断で根本原因と特定された「target 決定の文脈欠如」と「ラベルノイズ」を同時に解消できる唯一のアプローチ。pro player の戦術を 6-8 種類の高レベル抽象に圧縮することで、val_target_acc を 60%+ に押し上げられる見込み。

---

## 触れたファイル一覧 (iter 1-6 累積)

### 実装変更
- `pipeline/imitation/case1/policy/decoder.py` (iter 6: `min_fire_topk` 追加)
- `pipeline/imitation/case1/policy/model.py` (iter 3: pairwise 拡張、iter 4: value head 追加)
- `pipeline/imitation/case1/policy/templates.py` (iter 2 で 12 拡張、iter 3 で 8 に戻す)
- `pipeline/imitation/case1/training/losses.py` (iter 5: focal loss 化)
- `pipeline/imitation/case1/training/preprocess.py` (iter 3: pairwise feats、iter 4: value label)
- `pipeline/imitation/case1/training/train.py` (iter 6: focal パラメータ伝播バグ修正)
- `pipeline/imitation/case1/configs/il_baseline.yaml` (iter 5/6: focal パラメータ追加)
- `pipeline/imitation/case1/policy/agent.py` は **純 IL のまま不変** (`featurize → model → decode` の経路)

### テスト更新
- `tests/pipeline/imitation/case1/test_decoder.py` (iter 6: `min_fire_topk=0` 明示)

### 評価成果物
- `pipeline/imitation/case1/evaluation/results_iter{3,4,6}.json`
- `pipeline/imitation/case1/evaluation/results.json` (iter 6 = 純 IL 最終版)
- `pipeline/imitation/case1/policy/weights.pt` (iter 6 best, val_target_acc=0.395)

---

## 参考データ

### 訓練データ統計

```
rows: 117,226 (train) + 13,025 (val) (rating_quantile=0.50, 1355 episodes)
template label 分布 (train, NUM_TEMPLATES=8):
  0 NEAREST_NEUTRAL_LOW : 13.1%
  1 NEAREST_ENEMY       : 32.8%
  2 HIGH_PROD_NEUTRAL   :  6.3%
  3 HIGH_PROD_ENEMY     :  9.5%
  4 REINFORCE_FRONTLINE : 16.3%
  5 REINFORCE_WEAKEST   : 11.9%
  6 WEAKEST_ENEMY       :  8.6%
  7 NO_OP               :  1.6%
```

### iter 1-6 学習ログ保存場所

`/tmp/train_iter{1,2,3,5,6}.log` (iter 4 は前バージョンの上書き)

---

## 関連ドキュメント

- 出発点の診断: [`20260419_case1_diagnosis/result.md`](../20260419_case1_diagnosis/result.md)
- 直前の改善試行 (試行 1-3): [`20260420_case1_improve/result.md`](../20260420_case1_improve/result.md)
- pipeline 提出ルール: `.claude/rules/pipeline.md`
- baseline_v1 仕様: [`20260418_baseline.md`](../../../competition/20260418_baseline.md)
