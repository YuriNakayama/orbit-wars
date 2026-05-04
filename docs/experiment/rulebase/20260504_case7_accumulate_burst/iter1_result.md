# [rulebase/case7] iter1 結果: Stage 0 smoke (vs baseline_v6) — 棄却

> 評価コマンド (4 並列、各 26 ep / 計 104 ep, seed=1000/2000/3000/4000):
> ```bash
> nohup uv run --directory bot python -m pipeline.rulebase.case7.evaluation.compare_v6 -n 13 --seed 1000 > /tmp/case7_stage0/A.log 2>&1 &
> nohup uv run --directory bot python -m pipeline.rulebase.case7.evaluation.compare_v6 -n 13 --seed 2000 > /tmp/case7_stage0/B.log 2>&1 &
> nohup uv run --directory bot python -m pipeline.rulebase.case7.evaluation.compare_v6 -n 13 --seed 3000 > /tmp/case7_stage0/C.log 2>&1 &
> nohup uv run --directory bot python -m pipeline.rulebase.case7.evaluation.compare_v6 -n 13 --seed 4000 > /tmp/case7_stage0/D.log 2>&1 &
> ```
> 構成: 初期 config (`KNEE=60, SAFETY=4, MAX_HOLD=12, MIN_TGT=15, MAX_TGT=60, THREAT_RES=0`)
> 対戦相手: baseline_v6 (case6 cap=3 確定構成)
> 環境: kaggle_environments orbit_wars 1v1, **Python simulator** (Rust 未 build), seed 1000..4012
> 計算: ローカル CPU, 4 並列 (12 物理コア)
> 実行時間: 約 25 分 (104 ep)
> commit SHA: `0847e19` (working tree、case7 未 commit)

## 結論サマリ

**初期仮説 (target-aware multi-turn accumulate) は強く棄却**。104戦で **v7=19 / v6=85 / draws=0 → 18.3% 勝率** (棄却閾値 < 50% を大きく下回る)。Stage 1 sweep への移行は cost に見合わず、根本設計の見直しが必要。

### 4 並列の内訳

| seed base | 戦数 | v7 勝 | v6 勝 | 勝率 | seat=0 v7 | seat=1 v7 |
|---|---|---|---|---|---|---|
| A: 1000 | 26 | 6 | 20 | 23.1% | 23.1% | 23.1% |
| B: 2000 | 26 | 3 | 23 | 11.5% | 15.4% | 7.7% |
| C: 3000 | 26 | 5 | 21 | 19.2% | 23.1% | 15.4% |
| D: 4000 | 26 | 5 | 21 | 19.2% | 38.5% | 0.0% |
| **合計** | **104** | **19** | **85** | **18.3%** | 25.0% | 11.5% |

seat 非対称性大 (seat=0 25% / seat=1 11.5%) — accumulate 戦術は seat=1 で特に劣化。

## 数値テーブル

| メトリクス | 平均 | 解釈 |
|---|---|---|
| **launches/ep ratio (v7/v6)** | **0.38** | 致命的: v7 は v6 の 38% しか発射していない (累積し過ぎ) |
| avg fleet peak ratio | 1.08 | 単発は確かに大きいが (knee 達成効果) trade-off にならず |
| avg episode len | 168 turns | v6 同士の self-play 時より短く、v7 が早期に押される |

## 構造的解釈

### 失敗の主因 (推測): 「累積待ち」中の機会損失

initial config の `KNEE_SHIPS=60` は MAX_SPEED=6 の knee を狙ったが、Orbit Wars の典型ゲームでは:

1. **早期 (step 0~50)**: 初期 home planet ships=10 + production 2~3/turn。`60` ships に達するまで **~17 ターン** かかる。その間 v6 は capture / harass / snipe を 3-5 回打って中立惑星を確保し、production base を伸ばす。
2. **中期 (step 50~150)**: v7 はやっと 1 発目を撃てるが、その時点で v6 は既に 3-4 中立惑星を保有、production 差で逆転不可能。
3. **後期 (step 150+)**: v7 が累積した大艦隊で遠距離攻撃 → v6 は短距離 reinforce で受け切る (距離コストの差)。

`launches/ep ratio = 0.38` は「累積中に発射機会を逃している」直接的な証拠。fleet peak ratio が 1.08 とわずかに大きいのは knee が機能している証だが、**発射回数の 62% 削減を補えていない**。

### 副次仮説: `THREAT_RESERVE_MAX=0` が厳しすぎる

`reserve > 0` の source を全て accumulate 対象外にしたが、Orbit Wars では中盤以降ほぼ全 source に何らかの enemy fleet が向かい reserve > 0 になる。これも accumulate の発火頻度を下げる方向に作用。

### case6 (1ターン arbitrage) との根本的な違い

case6 STAY_BURST は「launch-now と launch-after-1-turn の ETA 比較」 = **常にどこかには発射する**選択。一方 case7 ACCUMULATE は「未来の理想単発」を狙って **発射そのものを抑制**。Orbit Wars は production-snowball 型ゲームのため、発射抑制は production 差を広げる方向に働く。

## 次の一手

### Option A (最優先): config 大幅緩和して Stage 0 再実行

initial config が「保守的すぎる累積」を作っている可能性が高い。以下の方向で再 smoke:

| 軸 | iter1 (初期) | iter2 候補 | 意図 |
|---|---|---|---|
| `KNEE_SHIPS` | 60 | **20-30** | 早期発射を許容、knee 効果は薄れるが launch 頻度を case6 並みに |
| `SAFETY_SHIPS` | 4 | **2** | しきい値を下げ、capture 必要量寄り |
| `MIN_TARGET_TURNS` | 15 | **20-25** | 近距離は通常 mission に完全に任せ、本当に遠距離 (orbit 半周相当) のみ accumulate |
| `MAX_HOLD_TURNS` | 12 | **5-8** | 累積上限を厳しく、case6 cap=3 と中間 |
| `THREAT_RESERVE_MAX` | 0 | **3** | reserve が小さいなら accumulate 許可 |

→ iter2 で 100戦 vs v6、勝率 ≥ 45% なら Stage 1 sweep へ進む。≥ 50% で初めて hypothesis が「使えるかもしれない」段階。

### Option B: 設計の根本見直し — accumulate を mission ではなく "augmentation" に

「accumulate fire を独立 mission にする」のではなく、capture/snipe mission の **send 量を knee 以上に底上げする** 形に変更。発射タイミングは既存 mission の判断に任せ、ship 数だけ knee を踏むよう augment。これなら launch 頻度を維持できる。

実装: `strategy.py` の `preferred_send` を knee-aware にする (capture 系のみ)。

### Option C: case7 を破棄、別軸を探索

case6 + comet-aware dynamic hold (iter7_result.md Option B) や rear_guard 改良などへ移行。

### 推奨: Option A → 効かなければ Option B

Option A は 30 分で結果が出るので試す価値あり。失敗時は Option B を iter3 で。

## 再現手順

```bash
mkdir -p /tmp/case7_stage0
nohup uv run --directory bot python -m pipeline.rulebase.case7.evaluation.compare_v6 -n 13 --seed 1000 > /tmp/case7_stage0/A.log 2>&1 &
nohup uv run --directory bot python -m pipeline.rulebase.case7.evaluation.compare_v6 -n 13 --seed 2000 > /tmp/case7_stage0/B.log 2>&1 &
nohup uv run --directory bot python -m pipeline.rulebase.case7.evaluation.compare_v6 -n 13 --seed 3000 > /tmp/case7_stage0/C.log 2>&1 &
nohup uv run --directory bot python -m pipeline.rulebase.case7.evaluation.compare_v6 -n 13 --seed 4000 > /tmp/case7_stage0/D.log 2>&1 &
wait
```

## 関連 docs

- `plan.md` (本実験の plan)
- `../20260502_case6_stay_mission/iter7_result.md` (case6 STAY_BURST cap=3 確定、`launches/ep ratio` 概念の出典)

## ログ保存先

`/tmp/case7_stage0/{A,B,C,D}.log`
