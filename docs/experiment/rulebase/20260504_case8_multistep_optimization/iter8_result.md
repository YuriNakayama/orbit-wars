# Rulebase/case10 — iter4 Result: Dynamic Threshold for Accumulate

> 作成日: 2026-05-05
> 対応 plan: [`iter4_plan.md`](./iter4_plan.md)
> 関連:
> - [`iter1_result.md`](./iter1_result.md) — step guard 単独で 53.0%
> - [`iter3_result.md`](./iter3_result.md) — thrash filter 機能するも勝率不変
> - 30戦敗北分析: `iter3_result.md` の所見 (Type A 8件 + Type B 6件) を踏まえた介入

## 結論

**仮説は否定。iter4 は採用却下、これまでで **最悪の悪化** (-13pp)。** 30 戦 vs `baseline_v4` で **40.0%** (12/30)、iter1/iter3 の 53.3% から **-13.3pp 大幅悪化**。

動的化は **Type A (早期消滅) を 8→6 に改善** するが、**Type B (長期消耗負け) を 6→12 に倍増** させた。net で大幅後退。撤退して `ACCUMULATE_KNEE_SHIPS_FLOOR=60` (= 旧 KNEE_SHIPS 等価) に復元。

## 数値

### iter4 30戦結果

| | wins (30戦, seed 84000+) | win_rate | turn_p95 |
|---|---|---|---|
| baseline_v10 (iter4) | 12 | **40.0%** | 0.347s |
| baseline_v4 | 18 | 60.0% | 0.690s |

### iter1-4 比較

| iter | win_rate | Type A (早期消滅) | Type B (長期消耗) |
|---|---|---|---|
| iter1 (step guard) | 53.3% | 推定 ~5 | 推定 ~9 |
| iter2 (KNEE=40) | 50.0% | (replay 取得無し) | — |
| iter3 (thrash filter) | 53.3% | 8 | 6 |
| **iter4 (動的 threshold)** | **40.0%** | **6 (改善)** | **12 (倍増)** |

### しきい値判定

| 項目 | しきい値 | 実測 | 判定 |
|---|---|---|---|
| 合算勝率 vs v4 | ≥55% | 40.0% | ❌ -15pp 大幅未達 |
| iter1 比改善 | +2pp | -13.3pp | ❌ これまでで最悪の悪化 |
| Type A 削減 | 8→4以下目標 | 8→6 | ⚠️ 部分達成 |
| Type B 不変 | 6 維持 | 6→12 | ❌ 倍増 |

## 診断 — なぜ動的化が逆効果だったか

### 動的化は意図通り発射量を絞った

iter4 の `_accumulate_target_threshold` は:
- need=10 → threshold 18 (旧 60 から -70%)
- need=20 → threshold 30 (旧 60 から -50%)

→ **small-need target には軽量発射、large-need target には維持**。

### 副作用: 敵 60-ships 反撃に押し負け

Type B 典型 seed 84002 (LOSS, 499T, 6 vs 26 planets) replay:
- self planet_gain: **36 件** (取りまくる)
- self planet_loss: **30 件** (取り返される)
- net planet 増加: 6 (production 比 6:70 で完敗)

iter1/iter3 では:
- 60 ships の標準発射 → 敵反撃を **耐えて planet を維持**
- net planet が +20-25 で勝てる

iter4 では:
- 18-30 ships の軽量発射で planet 取得 → 敵 60 ships 反撃で **即奪還**
- 取って奪われ取って奪われ…の wear-out が続き、production 拡大できず長期戦で完敗

つまり **「accumulate の発射量は敵の標準発射量 (60 ships) に対する競合戦力として 60 ships 必要」** が経験的に確認された。動的化で軽くしすぎると、capture 成功はするが守れない。

### Type A の改善 (8→6) はあくまで副次

軽量発射で home の ship 残存量が増えた → home 防衛が改善 → Type A (home 全滅) が減ったが、その代わりに中盤の競合力を失った = **トレードオフが負ける側に偏った**。

## 採用方針

- **iter4 は採用却下**、これまでで最悪の悪化
- `ACCUMULATE_KNEE_SHIPS_FLOOR = 60` に変更 (= 動的化を実質無効化、旧 `KNEE_SHIPS` 等価)
  - threshold 計算は `min(floor=60+SAFETY, overshoot_cap)` だが、`floor=60` 以上なので動的化分岐がほぼ発動しない
- case10 確定構成: **iter1 設定 (step guard=30, KNEE_SHIPS=60, no filter, no dynamic) で 53.0% (n=100)**

## 確定した知見

1. **accumulate の発射量 60 ships は敵反撃に対する最低限の戦力**: これより小さくすると競合に負ける wear-out が発生
2. **動的化は Type A と Type B のトレードオフ**: home 全滅 (Type A) を防ぐと中盤戦力 (Type B) が落ちる
3. **iter1-4 累計**: case10 の最良構成は iter1 (step guard 単独) で **53.0%、しきい値 ≥55% に -2pp 不足**
4. **case10 の 4 iter 累計でわかったこと**: 「heuristic 系の局所改修は完全に飽和」+ 「動的化のような mission logic 改修も負ける」 = **case7 base 上の改善はもう困難**

## 次の方向 (本ディレクトリスコープ外、最終結論)

| 案 | 期待 | コスト | 推奨度 |
|---|---|---|---|
| **case10 を 200戦で再評価** (53.0% 真値確定) | seed variance 縮小、≥55% に届くかは微妙 | 30分 | ★★ |
| 別ディレクトリで experiment-plan: case4 base 上の新 mission 追加 | LB745 production 突破狙い | 別 plan | ★★★ |
| 学習ベース value function (case4 base 上) | heuristic 飽和を脱出 | 数日 | ★ |

heuristic 系は **完全飽和 (8 連敗)** なので、次は構造的に新軸 (case4 base + 学習 / 新 mission / portfolio search) に進むべき。

## 関連ファイル

- `bot/pipeline/rulebase/case10/baseline/core/config.py:ACCUMULATE_KNEE_SHIPS_FLOOR` — iter4 で 30、本 result で 60 に復元
- `bot/pipeline/rulebase/case10/baseline/missions/stay.py:_accumulate_target_threshold` — 動的化式 (FLOOR=60 で実質無効化)
- `data/output/experiment/rulebase/case10/replay_analysis/20260505_iter4_all30/` — iter4 全 30 戦 replay

## 環境

- ハードウェア: M4 MacBook (local), parallel=4
- branch: `feature/rulebase-multistep-optimization`
- 実行日時: 2026-05-05
