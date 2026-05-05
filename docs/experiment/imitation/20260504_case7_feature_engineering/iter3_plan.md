# imitation/case7 iter3 — Pairwise Top-K + Defense surplus + Sparse mask

> 作成日: 2026-05-05
> 関連:
> - `./iter1_plan.md` / `./iter1_result.md` — iter1 (Stage 1 +0.136 PR-AUC、採用ゲート達成)
> - `./iter2_plan.md` / `./iter2_result.md` — iter2 (Stage 1 parity、ships head -0.013、ゲート未達)
> - `bot/pipeline/imitation/case7/policy/featurizer.py` — iter2 で 34 列 / 10 → 14 列 に拡張済
>
> スコープ: iter2 schema (34/14) を base に、**3 カテゴリ ~25-30 列**を追加。focus は Stage 1 ships head の改善 + target head の更なる前進 + 採否を「事前ゲートではなく feature importance 駆動」で判断。

## 仮説 (Hypothesis)

iter2 の ships head 後退 (-0.013) は **(a) sparse 列 (fleet trajectory が 0-fill 多) が信号を希釈**、**(b) decoder が必要とする「多 horizon defense surplus」が無く ships bucket 判定が曖昧**、**(c) target head が pair-wise 関係を直接観測できず src→tgt 距離 / ships比 を model が暗黙学習する非効率**、の 3 つが重なった結果。iter3 で次を追加し全 head 改善を狙う:

- **A. Pairwise Top-K**: src plane ごとに最近 K=5 個の other planet との {dx, dy, dist, ships比, owner_flag, prod_log} を前計算 → target head が直接 pair-wise 関係を見える
- **D. Defense surplus**: per-planet で「自軍 ships - 敵 incoming ships」を h=5/15/30 horizon × 2 (incl/excl prod) で渡す → ships head が defense margin を直接知る
- **J. Sparse mask**: iter2 で 0-fill 多かった列 (fleet trajectory / history / per-planet ship event) に has_xxx flag を併設 → policy がデフォルト値と「真の 0」を区別

## 既存コードの現状 (from Step 1)

- **iter2 featurizer (PLANET_FEAT_DIM=34, GLOBAL_FEAT_DIM=14)**:
  - planet 0-23: iter1 base + history + ship event
  - planet 24-27: fleet trajectory (inbound enemy 5-turn future delta) ← **sparse 主犯**
  - planet 28-31: multi-horizon (loss_5/15, min_owned_5/15) ← collinear
  - planet 32-33: home_dist + prod_centroid_dist
  - global 10-13: comet 2 + home_owner + prod_centroid_dist_norm
- **iter2 weights は best.pt (val_loss 3.5142, epoch 9) が canonical**、featurizer 34/14 schema と整合
- **iter2 result の 4 学び** (`iter2_result.md` より):
  1. fleet trajectory は sparse mask が必要
  2. multi-horizon は h=30 と相関高、独立信号にならず
  3. ships head の feature 設計が iter2 では不在
  4. 訓練 cost は許容範囲 (RTX 4090 で 6 分弱)

## スコープ (Scope)

### 変更ファイル

| Path | 変更内容 |
|------|----------|
| `bot/pipeline/imitation/case7/policy/featurizer.py` | PLANET_FEAT_DIM 34 → **約 60-65** (+25-30)、GLOBAL_FEAT_DIM 14 → **14 (変更なし)**。下記 catalogue の列を追加 |
| `bot/pipeline/imitation/case7/configs/il_case7.yaml` | `model.planet_in_dim: 34 → <new>`。他は iter2 と同じ (epochs=15, lr=1e-3, batch=256) |
| `bot/pipeline/imitation/case7/training/preprocess.py` / `dataset.py` | dim 変更のみ (constants 経由で自動追従) |
| `bot/tests/pipeline/imitation/case7/test_featurizer_iter3.py` (新規) | A / D / J 各カテゴリ 1-2 テスト + sparse mask の 0/1 値域 + Pairwise Top-K の K=5 default + Defense surplus の符号 (margin > 0 / < 0) |

### 変更なし

- AGENT_REGISTRY (`il_v7` のまま、weights.pt 上書き)
- model.py (input dim は constants 経由で受ける)
- decoder.py / main.py
- iter2 で追加した 14 列 (fleet trajectory / multi-horizon / production-centroid / comet) はすべて **保持**。iter3 で削らずに上乗せ

### Feature catalogue (追加予定)

#### A. Pairwise Top-K (planet 列 +20、K=5、4 feat each)

各 source planet について、**最近 K=5 個の other planet との関係**を前計算:

| idx (例) | 名前 | 定義 |
|----|------|------|
| 34, 38, 42, 46, 50 | `pair_k{0..4}_dist_log` | k 番目に近い planet との距離 / log1p(BOARD_SIZE) |
| 35, 39, 43, 47, 51 | `pair_k{0..4}_ships_ratio_log` | log1p(other.ships) - log1p(self.ships) |
| 36, 40, 44, 48, 52 | `pair_k{0..4}_owner_signed` | (1.0 if other.owner == player else (-1.0 if other.owner != -1 else 0.0)) |
| 37, 41, 45, 49, 53 | `pair_k{0..4}_prod_log` | log1p(other.production) |

K=5 全部で other planet が無い場合 (game start 直後) は dist=`-1` sentinel + 他 0 fill。

#### D. Defense surplus (planet 列 +4)

per-planet で **「(自軍 ships) - (敵 incoming ships)」** の予測値を h=5/15/30 で持ち、production 込/抜きを区別:

| idx | 名前 | 定義 |
|----|------|------|
| 54 | `defense_surplus_5turn` | (own_ships + 5*production - incoming_enemy_ships) / 100 (clamped to [-1, 1]) |
| 55 | `defense_surplus_15turn` | 同 h=15 |
| 56 | `defense_surplus_30turn` | 同 h=30 |
| 57 | `defense_margin_now` | 現時点での (own - incoming_enemy) / 100、production 抜き、即時 margin |

negative 値は「敵が打ち勝ちうる」、positive は「防御過剰」。ships head の bucket 0 (低送出) / 3 (全送出) 判定の重要シグナル。

#### J. Sparse mask flags (planet 列 +6)

iter1/iter2 で 0-fill 多かった列に has_xxx flag を併設:

| idx | 名前 | 定義 |
|----|------|------|
| 58 | `has_inbound_fleet_flag` | iter2 fleet trajectory (idx 26 dist != -1) なら 1.0 |
| 59 | `has_history_flag` | history snap_t1 が存在する (= match step >= 2) なら 1.0 |
| 60 | `has_enemy_targeted_flag` | iter1 enemy_targeted_count_last4 (idx 22) > 0 なら 1.0 |
| 61 | `is_home_planet_flag` | この planet が home (initial_planets[player]) なら 1.0 |
| 62 | `is_neighbor_to_enemy_flag` | この planet の半径 board_size/4 以内に敵 planet があるなら 1.0 |
| 63 | `comet_planet_flag_redundant_with_iter1` | (省略可、iter1 col 8 と完全重複) |

idx 63 は iter1 の `is_comet` と redundant なので **削除**。net mask flag = 5 列。

#### 合計

- planet: 34 (iter2) + 20 (A) + 4 (D) + 5 (J) = **63 列**
- global: 14 (iter2 そのまま) = **14 列**

### 設計原則

- **Pairwise Top-K の sort**: 各 source 視点で `dist` 昇順で top-5。同 source 自身は除外。tie-break は planet id 昇順 (deterministic)。
- **Top-K の boundary case**: `n - 1 < K` なら不足 slot は dist=`-1` (sentinel) + 他 0 で埋める。
- **Defense surplus の causality**: 現在 obs の ships / incoming のみ参照、history 不要。causal leak リスクなし。
- **mask flag は 0/1 binary**: float でも model は容易に分離学習できる。
- **計算コスト**: Pairwise Top-K は MAX_PLANETS=36 で 36×36 = 1296 距離計算/frame。numpy vectorize で <10ms。preprocess 全体に対して数% 増。

## 実装ステップ (Implementation outline)

1. **featurizer.py 拡張**:
   - PLANET_FEAT_DIM 34 → **63**
   - 既存 logic の後に Pairwise Top-K helper を追加 (`_compute_pairwise_topk(planet_x, planet_y, planet_owner, planet_ships, planet_prod, k=5) -> ndarray (P, K, 4)`)
   - per-planet defense_surplus helper (incoming は既存 `incoming` array を再利用、自軍 ships は raw_planets[slot])
   - has_inbound_fleet_flag は idx 26 (`inbound_fleet_dist`) > -1 で計算、has_history_flag は `snap_t1 is not None` 等
2. **`configs/il_case7.yaml` 更新**: `planet_in_dim: 63`
3. **テスト追加** (`test_featurizer_iter3.py`):
   - Pairwise Top-K: K=5 で planet が 3 個しかないとき余り 2 slot が sentinel
   - Pairwise Top-K: dist 昇順か (sort 検証)
   - Defense surplus: own_ships=10, incoming_enemy=5 で margin > 0
   - mask flag: has_inbound_fleet_flag は idx 26 と同期、has_history_flag は HistoryState なしで 0
   - causal safety: 全列が `obs_{N-1}` を参照しないことを deterministic test (iter2 と同様)
4. **dim sanity test 更新**: `test_featurizer_dim.py` の `test_planet_feat_dim_is_34` → `test_planet_feat_dim_is_63`
5. **ローカル smoke**: `dev/test-bot` (format/lint/mypy/pytest) green。iter3 unit test 追加 5-7 本。
6. **commit & push** (この plan は execution skill で実施)。
7. (execution side) **RunPod Step B**: trap #9 fix で deps hash 変更を検出 → preprocess 再実行 → train。約 30 分、~$0.30。
8. (execution side) **Stage 1 evaluation + permutation importance**: iter3 では採否ゲートを「事前 +0.01」ではなく **feature importance 計算 → ユーザーと議論**で決める (重要)。
9. (execution side) **iter3_result.md** で metrics + importance + 議論結果を整理。

## 検証方法 (Validation method)

### ローカル

```bash
dev/test-bot
uv run --directory bot pytest tests/pipeline/imitation/case7 -x

# featurizer dim sanity
uv --project bot run python -c "from pipeline.imitation.case7.policy.featurizer import PLANET_FEAT_DIM, GLOBAL_FEAT_DIM; print(PLANET_FEAT_DIM, GLOBAL_FEAT_DIM)"
# → 63 14
```

### リモート (execution skill 側)

```bash
git push origin feature/feature-engineering
dev/runpod train <iter3-sha> --case case7 \
  --gpu-name "NVIDIA GeForce RTX 4090" --gpu-name "NVIDIA GeForce RTX 3090" --gpu-name "NVIDIA RTX A6000"
# trap #9 fix で deps invalidate → preprocess 再実行
# 想定: preprocess ~7 分 + train ~7 分 + dvc/git push ~5 分 = ~25-30 分、~$0.30
```

### 評価 (採否は importance-driven)

iter3 では **事前固定のしきい値による採否ゲートを使わない**。代わりに次の手順:

1. **Stage 1 metrics** を `diagnose_weights.py` で取得 (iter1/iter2 同体系: from PR-AUC, target macro F1, ships macro F1 ほか)
2. **Permutation feature importance** を val.parquet 上で計算:
   - 各 column (or column group) を順番に shuffle して predicted distribution の KL divergence を測る
   - groupings: A_pairwise_topk (20 列を group)、D_defense_surplus (4 列)、J_mask_flags (5 列)、その他 iter1/iter2 既存 group
   - 出力: `data/output/experiment/imitation_case7_iter3_feature_importance.json`
3. **議論ベース判定**: 重要度 + Stage 1 metrics をユーザーに提示、採用 / 部分採用 / 破棄を **会話で決定**
4. (採用の場合) Stage 2 self-play は別途 follow-up plan で実施 (300 ep 級)

### リーク回帰防止

- iter3 の追加列 (A/D/J) はいずれも `obs_{N-1}` を参照しない設計だが、**`test_featurizer_iter3.py` で deterministic test を必須**:
  - 同 obs を 2 回 featurize して全列一致 (history 非依存)
  - HistoryState を渡しても渡さなくても A/D/J 列は同値

## リスク / 想定失敗モード

1. **Pairwise Top-K が target head に追加 noise**: target head は既に `template_ctx` (40 列) で per-source pair info を持つ。Top-K と redundant になり target macro F1 が iter2 と parity の可能性。緩和: `template_ctx` と異なる normalization (raw log 距離 vs prox-normalized) を使う。
2. **Defense surplus が ships head と redundant**: ships head が学んでいる "incoming - own" 関係を直接 input で渡すと、model がそれを使うようになり学習が容易になる一方、**他 feature への gradient が下がる** 可能性。緩和: importance で確認、ships F1 が伸びていれば OK。
3. **mask flag が情報無し**: iter2 で 0 fill 多かった列がそのまま 0 fill のままなら、mask flag も 0 ばかりで信号にならない。緩和: 統計を取って 0-fill 率が 50% 以下の列のみ mask 化。
4. **PLANET_FEAT_DIM=63 の memory cost**: parquet サイズ iter1 比 ~2.5 倍 (660 MB → 1.5 GB)、preprocess も比例して長い。trap #9 経由で再実行されるので問題は出ないが S3 push に -j 1 で時間かかる (~10-15 分)。
5. **trap 検証**: iter2 で trap #8 (data/output/models/imitation symlink unlink) と trap #9 (preprocess_skip 鮮度確認) は実証済み。新 trap が出る確率は低いが要観測。

## Stop conditions

以下を満たしたら本 plan のスコープは完了:

- [ ] case7 featurizer が iter3 仕様 (PLANET_FEAT_DIM=63) に拡張、unit test pass
- [ ] `dev/test-bot` green
- [ ] commit & push
- [ ] (execution side) RunPod Step B が `99_done` で終了、weights.pt がローカルに pull
- [ ] Stage 1 metrics + permutation importance 計算
- [ ] iter3_result.md に metrics + importance + ユーザーとの議論結果を記録

## 参考

- `iter2_result.md` の教訓セクション: sparse mask 必要 / multi-horizon collinear / ships head ゲートを絞れ
- iter1 case3 phase2 result: history 列の causal leak 事例 (iter3 では history を新規追加しないので低リスク)
- memory `project_imitation_case1_phase3`: n<300 self-play 信頼不可、Stage 1 で採否判断する妥当性
