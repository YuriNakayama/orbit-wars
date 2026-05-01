# imitation/case1 改善実験 (template化 + NO_OP fix + pos_weight)

- 実施日: 2026-04-19 〜 2026-04-20
- ブランチ: `feature/imitation-case1-improve`
- 対象: `pipeline/imitation/case1/` (Imitation Learning Baseline)
- 出発点: `il_v1` vs `baseline_v1` (1v1, 100戦) で **win_rate = 0.00**
  ([20260419_case1_diagnosis/result.md](../20260419_case1_diagnosis/result.md))
- 結果: 3 段階の修正を実施したが **win_rate = 0.00 のまま**。
  ただし発射数は 1 → 217 に回復し、ボトルネックは「fire しない問題」から
  「target 多様性の欠如」へシフトした。

---

## サマリ

診断結果に基づき、target action space を 1296-class (planet-id) から
**8 templates** に置換し、過剰発射を抑える decoder 後処理 (案4) を追加した。
さらに学習中に判明した 2 つの追加バグを修正したが、win rate は変わらなかった。

| 試行 | 主要変更 | val_target_acc | IL fires/episode | 勝率 (vs baseline_v1, 100戦) |
|---|---|---|---|---|
| 旧 (修正前) | DeepSets, planet-id 分類 (1296 class) | 0.34 | 不明 (no-op 極振り) | 0/100 |
| 試行1 | template 化 (NO_OP fallback あり) | 0.62 (見かけ) | 1 | 0/100 |
| 試行2 | NO_OP fallback fix (NO_OP 64% → 1.6%) | 0.34 | 263 | 0/100 |
| 試行3 | BCE `pos_weight = 8.5` を追加 | 0.34 | 217 | 0/100 |

**結論:** 発射するようにはなったが、全 source が template 0
(NEAREST_NEUTRAL_LOW) ばかり選択するため、戦略的多様性で baseline_v1 に
及ばない。target_acc が 0.34 で頭打ちとなっており、現アーキテクチャの
表現力が限界。次の改善には h 拡張または PPO/MARL 切替が必要。

---

## 実施した修正

### 1. Target action space を template に置換

**Before:** target head は 1296 (=36×36) class CE。実質的に "from planet" と
"target planet" の同時分類で、学習信号が極めて疎。

**After:** 8 strategic templates の class CE。各テンプレートは決定論的に
1 つの target planet に解決される。

```python
# pipeline/imitation/case1/policy/templates.py
NUM_TEMPLATES = 8
T_NEAREST_NEUTRAL_LOW = 0   # 最近接の中立 (garrison ≤ src.ships)
T_NEAREST_ENEMY = 1         # 最近接の敵
T_HIGH_PROD_NEUTRAL = 2     # 中立で最高 production
T_HIGH_PROD_ENEMY = 3       # 敵で最高 production
T_REINFORCE_FRONTLINE = 4   # 自軍のうち敵重心に最も近い惑星
T_REINFORCE_WEAKEST = 5     # 自軍のうち最少 ships
T_WEAKEST_ENEMY = 6         # 敵のうち最少 ships
T_NO_OP = 7                 # 発射しない
```

`resolve_template(template_id, src_row, planet_rows, player) -> int | None`
で template id → target planet id を解決。decoder はこれで argmax template から
具体的な target を求める。

**変更ファイル:**
- `pipeline/imitation/case1/policy/templates.py` (新規)
- `pipeline/imitation/case1/policy/model.py` (target head 出力次元を `MAX_PLANETS+1` → `NUM_TEMPLATES` に)
- `pipeline/imitation/case1/policy/decoder.py` (template → planet 解決)
- `pipeline/imitation/case1/training/preprocess.py` (label を template id に)

### 2. Overfire suppression (案4) を decoder に実装

複数 source が同じ target に集中して送る "全員ぶつける" 失敗モードを抑制。

```python
# pipeline/imitation/case1/policy/decoder.py
# 1) friendly fleet trajectory を解析して incoming_friendly[target_pid] を作る
# 2) source は from_prob 降順で処理 (greedy commitment)
# 3) 各 target に committed[target_pid] += ships を加算
# 4) enemy/neutral target: need = target.ships + 1 - committed
#    ships > need*2 なら ships = max(need, 1) にクリップ
# 5) 自軍 reinforcement: committed > target.ships*2 なら skip
```

### 3. NO_OP fallback の致命バグ修正 (試行1 → 試行2)

**症状:** `classify_actual_target` が「fired source の target にマッチする
template が存在しない場合 NO_OP を返す」実装になっていた。これにより
**fired source の 64.2% が NO_OP ラベルで学習** されていた
(モデルは「発射するけど何もしない」を学ぶ羽目に)。

**修正:** fired source は必ず非 NO_OP テンプレートに分類。マッチしない
場合は **resolved target が actual target に最も近いテンプレート** を選ぶ。

```python
# pipeline/imitation/case1/policy/templates.py
def classify_actual_target(src_row, target_row, planet_rows, player) -> int:
    actual_id = int(target_row[0])
    target = _to_p(target_row)
    resolved_per_tid = {}
    # 1) exact match in priority order
    for tid in range(NUM_TEMPLATES - 1):
        resolved = resolve_template(tid, src_row, planet_rows, player)
        resolved_per_tid[tid] = resolved
        if resolved == actual_id:
            return tid
    # 2) fallback: distance-closest template
    planets_by_id = {int(r[0]): _to_p(r) for r in planet_rows}
    best_tid, best_dist = T_NEAREST_ENEMY, float("inf")
    for tid, rid in resolved_per_tid.items():
        if rid is None: continue
        rp = planets_by_id.get(rid)
        if rp is None: continue
        d = math.sqrt((rp.x - target.x) ** 2 + (rp.y - target.y) ** 2)
        if d < best_dist:
            best_dist, best_tid = d, tid
    return best_tid
```

**ラベル分布の変化** (training set, 118,283 fired sources):

| template | 試行1 (旧) | 試行2 (修正後) |
|---|---|---|
| 0 NEAREST_NEUTRAL_LOW | 4.6% | 13.1% |
| 1 NEAREST_ENEMY | 13.4% | **32.8%** |
| 2 HIGH_PROD_NEUTRAL | 2.7% | 6.3% |
| 3 HIGH_PROD_ENEMY | 3.5% | 9.5% |
| 4 REINFORCE_FRONTLINE | 5.9% | 16.3% |
| 5 REINFORCE_WEAKEST | 2.7% | 11.9% |
| 6 WEAKEST_ENEMY | 3.1% | 8.6% |
| **7 NO_OP** | **64.2%** | **1.6%** |

### 4. BCE pos_weight=8.5 の追加 (試行2 → 試行3)

**症状:** 試行2 学習後、エージェントは 263 fires (vs baseline 521) と発射数が
回復したが、from_prob が常に 0.01-0.03 で `from_threshold=0.05` を切り、
home planet を持っていても発射しない局面が大半だった。

**原因:** `my_planet_mask` 内で fired:not_fired = 1 : 8.5 のクラス不均衡。
標準 BCE では「fire しない」が優勢解になっていた。

**修正:** `from head` の BCE に `pos_weight = neg/pos = 8.5` を導入。

```python
# pipeline/imitation/case1/training/losses.py
@dataclass(frozen=True)
class LossWeights:
    from_w: float = 1.0
    target_w: float = 1.0
    ships_w: float = 0.5
    from_pos_weight: float = 8.5  # neg/pos ratio in training data

# compute_loss():
pos_weight = torch.tensor(weights.from_pos_weight, device=device)
bce = nn.functional.binary_cross_entropy_with_logits(
    safe_logits, from_target, reduction="none", pos_weight=pos_weight
)
```

`from_pos_weight` は YAML config からも変更可能 (`train.loss_weights.from_pos_weight`)。

---

## 学習結果 (試行3, 最終版)

### config

```yaml
seed: 0
data:
  out_train: data/lake/imitation_case1/train.parquet  # 104,606 frames
  out_val: data/lake/imitation_case1/val.parquet      # 12,620 frames
  rating_quantile: 0.50
  modes: ["1v1"]
model:
  hidden: 64
  ships_buckets: 4
train:
  batch_size: 256
  epochs: 15
  lr: 1.0e-3
  weight_decay: 1.0e-4
  loss_weights:
    from: 1.0
    target: 2.0
    ships: 0.5
    from_pos_weight: 8.5
inference:
  from_threshold: 0.05
```

### 学習ログ抜粋 (background ID `b1znoo4d2`)

```
epoch 0:  train=4.81 val=4.88 from_acc=0.728 target_acc=0.296 ships_acc=0.689
epoch 6:  train=4.42 val=4.67 from_acc=0.788 target_acc=0.322 ships_acc=0.712
epoch 12: train=4.33 val=4.59 from_acc=0.790 target_acc=0.337 ships_acc=0.722  ← best
epoch 14: train=4.31 val=4.60 from_acc=0.756 target_acc=0.336 ships_acc=0.715
best_val_loss=4.5912 best_epoch=12
```

- `val_from_acc` は 0.79 前後 (試行2 の 0.90 から低下したが、これは
  positive クラスを多めに当てるよう学習した正常な変化)
- `val_target_acc` が 0.34 で頭打ち。試行2 (NO_OP fix 後) と同水準で、
  pos_weight 導入は target head の精度には影響しなかった

### 評価結果 (`il_v1` vs `baseline_v1`, 1v1, 100戦, seed=0)

```json
{
  "episodes": 100, "wins": 0, "losses": 100, "draws": 0,
  "win_rate": 0.0, "non_draw_win_rate": 0.0,
  "challenger": "il_v1", "baseline": "baseline_v1", "mode": "1v1"
}
```

`pipeline/imitation/case1/evaluation/results.json` に保存。

### サンプル対戦 (seed=0)

| 指標 | il_v1 | baseline_v1 |
|---|---|---|
| 総発射数 | 217 | 306 |
| 試合終了 step | 172 | 172 |
| reward | -1 | +1 |

---

## 診断: なぜ勝てないか

学習後のモデル出力を 1 試合追跡 (seed=0):

```
step= 1 my=1 ['p8:t0/0.49']
step= 2 my=1 ['p8:t0/0.96']
step= 5 my=1 ['p8:t0/0.97']
step=10 my=2 ['p8:t0/0.19', 'p28:t0/0.86']
step=13 my=2 ['p8:t0/0.98', 'p28:t0/0.97']
step=18 my=2 ['p8:t0/0.97', 'p28:t0/0.97']
```

**全ての source が `template 0` (NEAREST_NEUTRAL_LOW) を argmax として選んでいる。**
target ラベル分布 (試行2/3) では NEAREST_ENEMY が 32.8% と最大なので、
NEAREST_NEUTRAL_LOW (13.1%) ばかり出るのは "学習データの最頻クラスを当て続ける"
collapse とも違う。

考察:
- 序盤局面 (自軍 1 惑星、step ≤ 30) では「中立惑星を取る」が dominant
  だが、中盤以降も template 0 から動かない
- target_logits 8 値の差が小さく、序盤の bias が中盤以降の局面でも
  上書きされない (h=64 の表現力不足)
- pairwise (距離・production比) を target head に直接渡していないため、
  状況依存の template 選択ができない (試行は src embedding + global ctx
  だけで template を選んでいる)

---

## 残課題と次の改善案

優先度順:

1. **target head に pairwise 特徴を直接入力** — 距離・production比・
   ETA・garrison差を `(src, candidate)` ペアごとに計算して target head に
   渡す。template 選択を状況依存にする最短経路。
2. **hidden 拡張 (h=64 → 128)** + kNN k=8 → 16 — 表現力の純粋な拡張。
   重み 254KB → 1MB 程度になり Kaggle 提出 (1MB 制限) は要注意。
3. **NEAREST_NEUTRAL_LOW 偏りの mitigation** — class weighted CE
   (8 templates それぞれに inverse-frequency weight)。低頻度
   template (HIGH_PROD_NEUTRAL 6.3%) を学習させやすくする。
4. **PPO/MARL 切替** — `target_acc=0.34` が複数試行で頭打ちなので、
   BC では教師の戦略を表現しきれない可能性。報酬最大化に切り替えれば
   "勝つために必要な行動" を直接学習できる。

---

## 触れたファイル一覧

実装変更:
- `pipeline/imitation/case1/policy/templates.py` (新規, 142行)
- `pipeline/imitation/case1/policy/model.py` (target head 差し替え)
- `pipeline/imitation/case1/policy/decoder.py` (template 解決 + 案4)
- `pipeline/imitation/case1/policy/types.py` (PolicyOutput コメント更新)
- `pipeline/imitation/case1/training/preprocess.py` (template label 化)
- `pipeline/imitation/case1/training/losses.py` (pos_weight 追加)
- `pipeline/imitation/case1/training/train.py` (config から pos_weight 受け取り)

テスト更新:
- `tests/pipeline/imitation/case1/test_decoder.py` (template ベースに書き換え)
- `tests/pipeline/imitation/case1/test_model.py` (target_logits shape 更新)
- `tests/pipeline/imitation/case1/test_agent_integration.py` (重み不整合 skip)
- `tests/pipeline/imitation/case1/snapshots/action_001.json` (snapshot 更新)

成果物:
- `pipeline/imitation/case1/policy/weights.pt` (試行3, 254KB,
  best_epoch=12, val_target_acc=0.34)
- `data/lake/imitation_case1/train.parquet` (104,606 frames, NO_OP 1.6%)
- `data/lake/imitation_case1/val.parquet` (12,620 frames)
- `pipeline/imitation/case1/evaluation/results.json` (100戦, 0 wins)

---

## 関連ドキュメント

- 出発点の診断: [20260419_case1_diagnosis/result.md](../20260419_case1_diagnosis/result.md)
- ベースライン仕様: [pipeline/imitation/case1/README.md](../../../../backend/pipeline/imitation/case1/README.md)
- baseline_v1 (対戦相手) 詳細: [20260418_baseline.md](../../../competition/20260418_baseline.md)
