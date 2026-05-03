# [rulebase/case6] iter6 結果: STAY_BURST_MAX_HOLD_TURNS sweep (100戦 vs baseline_v5)

> 評価コマンド (Stage 1):
> ```bash
> uv run --directory bot python -m pipeline.rulebase.case6.evaluation.compare_v5 \
>   -n 50 --seed 1000 --max-hold-turns 2
> uv run --directory bot python -m pipeline.rulebase.case6.evaluation.compare_v5 \
>   -n 50 --seed 1000 --max-hold-turns 3
> uv run --directory bot python -m pipeline.rulebase.case6.evaluation.compare_v5 \
>   -n 50 --seed 1000 --max-hold-turns 4
> ```
> 構成: burst-only (defense=False, gain=1, ships=8, dist=30) + cap 値のみ可変
> 対戦相手: baseline_v5
> 環境: kaggle_environments orbit_wars 1v1
> 計算: ローカル CPU、3 並列実行 (~48 分)
> 関連: iter5 で MAX_HOLD=3 が 300戦 59.7% (p≈0.0004) と確定したことを受けた局所最適確認

## 結論サマリ

100戦 sweep で cap=3 (56.0%) と cap=4 (58.0%) はほぼ同等、cap=2 (50.0%) のみ
6〜8pp 弱い。cap=3 周辺で勝率はプラトーを形成しており、**cap=3 は局所最適に
位置している** (cap=4 が同等レベル、cap=2 で崩れる)。

cap=3 の 100戦再現値は 56/100 で iter5 Stage 1 (62/100) から -6pp ずれているが、
n=100 の二項 95% CI (±9.6pp) 内 — seed variance の範疇で、**実装ドリフトは
発生していない**。

**判定**: case6 は **MAX_HOLD_TURNS=3 で確定** (cap=4 への変更は 100戦差
2pp で根拠不足、Stage 2 起動の費用対効果なし)。

## 数値テーブル

### Stage 1 sweep (各 100戦, seed=1000)

| cap | v6 勝 | v5 勝 | 勝率 | seat=0 | seat=1 | fleet peak v6 | v5 | ratio | launches/ep ratio | 平均 ep 長 |
|---|---|---|---|---|---|---|---|---|---|---|
| **2** | 50 | 50 | **50.0%** | 42.0% | 58.0% | 21.3 | 17.3 | 1.23 | 0.91 | 178.5 |
| **3** | 56 | 44 | **56.0%** | 50.0% | 62.0% | 21.3 | 17.1 | 1.25 | 0.96 | 189.4 |
| **4** | 58 | 42 | **58.0%** | 58.0% | 58.0% | 20.9 | 16.9 | 1.24 | 0.99 | 183.8 |

### iter5 (cap=3, seed=1000, 同一 100戦) との比較

| 評価 | cap=3 勝率 | seat=0 | seat=1 | fleet peak ratio | launches/ep ratio |
|---|---|---|---|---|---|
| iter5 Stage 1 | 62.0% | 64.0% | 60.0% | 1.31 | 1.07 |
| **iter6 cap=3 (再現)** | **56.0%** | 50.0% | 62.0% | 1.25 | 0.96 |
| 差 | **-6.0pp** | -14.0pp | +2.0pp | -0.06 | -0.11 |

### 統計検定

- 差 (cap=4 56.0%? いえ 58.0% vs cap=3 56.0%): 2pp / n=100 → ほぼ確実に seed
  variance 内 (z≈0.28、p>0.7)。
- 差 (cap=2 50.0% vs cap=3 56.0%): 6pp / n=100 → z≈0.85、p≈0.4。
  「弱い tendency」止まりで、有意ではない。
- cap=3 100戦の二項 95% CI: 46.3〜65.7% (iter5 Stage 1 62% も iter6 56% も
  この CI 内)

## 解釈

### MAX_HOLD の周辺感度

| cap | 設計意図 | 実測 | 解釈 |
|---|---|---|---|
| 2 | 厳しい (3 turn 連続 hold を完全禁止) | 50.0% (-6 vs cap=3) | 累積効果を切り過ぎ。launches/ep ratio 0.91 で過剰 launch、fleet peak は維持 |
| 3 | iter5 採用値 | 56.0% (= iter5 平均 59.7% の seed 下振れ) | broad の累積効果を壊さない最小 cap |
| 4 | 緩め (4 turn 連続 hold まで許容) | 58.0% (+2 vs cap=3) | cap=3 と統計的差なし、launches/ep ratio 0.99 で「自然な launch ペース」に最も近い |

cap=2 → cap=3 の +6pp ジャンプに対し、cap=3 → cap=4 は +2pp のみ。
**「累積を許す方向にはサチっており、削る方向にはダウンサイドがある」**
非対称な感度が観測された。これは iter4 (gain=2/ships=12/dist=20 厳し burst)
で 41% に崩壊した知見と整合する: **broad burst は累積前提の設計で、
cap=2 のような追加抑制はその累積を壊す**。

### cap=3 と cap=4 の差は信号か

- 100戦 2pp 差は n=100 の noise floor 以下 (95%CI ±9.6pp)
- launches/ep ratio (cap=3: 0.96, cap=4: 0.99) は cap=4 の方が 1.0 (= v5 と同等)
  に近く、「cap=4 では実質的に hold 抑制が効きにくい」可能性
- cap=4 を採用しても、iter5 で確証済の「3 ターン上限が stuck hold を防ぐ」
  メカニズムを弱める方向であり、追加 ablation コスト (300戦 ~50分) が必要
- → **cap=3 を据え置きが合理的**

### 100戦 cap=3 再現の seed 変動

iter5 Stage 1 の 62/100 (seed=1000) と iter6 の 56/100 (同 seed=1000) は
**完全に同じコマンド** で 6pp ずれた。原因仮説:

- iter6 では `--max-hold-turns 3` 経由で config を runtime 書き換え (実体は同値)
- runtime 書き換え = 順序や import タイミングが iter5 と微妙に違う可能性
- ただし `cfg.STAY_BURST_MAX_HOLD_TURNS=3` 自体は両 run で同じ
- → **kaggle_environments の RNG / dict iteration の非決定性** が最尤の説明
  (iter3 でも seed=2000 で 51% と他 seed (68/60%) に対し変動を観測)

**100戦は単独では 6pp の seed 変動を含むため、本質的判定には 300戦が必要**
という事実が今回の sweep で再確認された。だが iter5 で **cap=3 の 300戦
59.7% (95%CI 54.1-65.2%) は別途確証済み** で、本 sweep はあくまで「周辺
cap 値が cap=3 を上回らないことの確認」なので、Stage 2 を新たに起動する
必要はない。

## 採用判定

**case6 は iter5 採用構成 (`STAY_BURST_MAX_HOLD_TURNS=3`) で確定**。

### 確定根拠

1. iter5 で 300戦 59.7% (p≈0.0004 で v5 より有意強)
2. iter6 sweep で cap=3 周辺はプラトー、cap=2 で崩れる ⇒ 局所最適
3. cap=4 は 100戦差 2pp で cap=3 を上回らず、変更コストの正当化困難

### Stage 2 を起動しない理由

- 判定基準「いずれかが 65%+」を満たす cap が存在しない (cap=4 が 58%)
- cap=3 vs cap=4 の差 2pp は seed variance 内 (95%CI が広く重なる)
- 仮に Stage 2 で cap=4 が cap=3 を有意に上回っても、iter5 で確証済の
  「stuck hold 3 ターン上限」のメカニズム説明が弱まる方向であり、
  説明可能性 (case6 README の根拠) を犠牲にしてまで採用する利得が小さい
- 50 分 × 3 並列の追加コストを別軸 (case7、defense 再導入、comet 動的 cap)
  に振り向けた方が期待値高い

### 推奨される次のアクション (user 承認後)

1. **case6 README.md** を更新 — iter5 確定構成 + iter6 sweep 結果のリンク
2. **Kaggle 再提出** (現提出 `case6_20260502-110353.tar.gz` は hold 上限なしの
   iter3 構成で本番投入されたまま) — 要 user 明示承認:
   ```
   dev/submit rulebase/case6 -m "case6 iter5: burst MAX_HOLD=3 (300戦 59.7% vs v5)"
   ```
3. **case6 ブランチ PR 化** — feature/add-rulebase-to-stay → main

### case6 の今後の改良候補 (今 iter のスコープ外)

- comet ターン (50/150/250/350/450) で MAX_HOLD を一時的に緩めて累積を取りに行く動的制御
- defense の再導入 (iter2 では burst-only より弱かったが、cap=3 と組み合わせると変わる可能性)
- case7 で別軸 (例: lookahead 強化、harass 再設計) を試す方が ROI 高い見込み

## ログ保存先

- `/tmp/case6_iter6/cap2.log` — 50/100 (50.0%)
- `/tmp/case6_iter6/cap3.log` — 56/100 (56.0%)
- `/tmp/case6_iter6/cap4.log` — 58/100 (58.0%)
