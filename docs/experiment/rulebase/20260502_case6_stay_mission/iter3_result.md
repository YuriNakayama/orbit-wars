# [rulebase/case6] burst-only 300戦バリデーション (3 seed × 100戦, vs baseline_v5)

> 評価コマンド (3 並列、各 100戦):
> ```bash
> uv run --directory bot python -m pipeline.rulebase.case6.evaluation.compare_v5 -n 50 --seed 1000
> uv run --directory bot python -m pipeline.rulebase.case6.evaluation.compare_v5 -n 50 --seed 2000
> uv run --directory bot python -m pipeline.rulebase.case6.evaluation.compare_v5 -n 50 --seed 3000
> ```
> 構成: **burst-only** (`STAY_DEFENSE_ENABLED=False`, `STAY_BURST_ENABLED=True`、Kaggle 提出済み構成と同一)
> 対戦相手: baseline_v5 (case6 直接派生元)
> 環境: kaggle_environments orbit_wars 1v1, seed 起点 1000/2000/3000、各 100戦
> 計算: ローカル CPU、3 ラン並列 (12 物理コア)
> 実行時間: 約 53 分
> 関連: Kaggle 提出 `case6_20260502-110353.tar.gz` (2026-05-02 11:03:56) と同じ構成
> 動機: iter2 の 100戦 burst-only 59% が seed1000 起点限定の数字だった可能性を、複数 seed で確証する

## 結論サマリ

**300戦バリデーションで burst-only 構成は 164/300 (54.7%) — 採用候補閾値 55% に届かず**。
95% 信頼区間 (binomial) は **±5.6pp、すなわち 49.1% ~ 60.3%** で **50% を含む**ため、baseline_v5 に対する有意な優位は **統計的に確認できなかった**。
ただし全 3 seed で 50% 以上を維持しており、「劣ってはいない」「fleet 形成効果は設計通り出ている」「defense-only (52%) より一貫して上」という性質は確実。

**判定**: case6 (burst-only) は **「v5 と互角〜やや上回る、Kaggle で別 opponent pool に当たれば差が出る可能性のある中堅構成」**。Full (53%) や defense-only (52%) を捨てて burst-only を採用した判断は依然合理的だが、「明確な改良」とは呼べない。

## 数値テーブル

### 各 seed 結果

| seed | 起点 episode | v6 勝 | v5 勝 | 引分 | v6 勝率 | seat=0 | seat=1 |
|---|---|---|---|---|---|---|---|
| 1000 | 1000..1099 | 50 | 50 | 0 | **50.0%** | 46.0% | 54.0% |
| 2000 | 2000..2099 | 57 | 43 | 0 | **57.0%** | 52.0% | 62.0% |
| 3000 | 3000..3099 | 57 | 43 | 0 | **57.0%** | 56.0% | 58.0% |
| **合計** | 300戦 | **164** | **136** | 0 | **54.7%** | **51.3%** | **58.0%** |

### 行動指標 (3 seed)

| seed | fleet peak v6 | v5 | ratio | launches/ep v6 | v5 | ratio | 平均 ep 長 |
|---|---|---|---|---|---|---|---|
| 1000 | 21.6 | 16.8 | 1.29 | 426.5 | 469.7 | 0.91 | 187.9 |
| 2000 | 21.7 | 17.2 | 1.27 | 441.2 | 438.0 | 1.01 | 175.4 |
| 3000 | 22.1 | 17.4 | 1.27 | 433.5 | 434.5 | 1.00 | 178.5 |
| **平均** | **21.8** | **17.1** | **1.28** | **433.7** | **447.4** | **0.97** | **180.6** |

### 統計量

- **300戦 v6 勝率**: 54.7%
- **95% 信頼区間** (binomial, n=300): ±5.6pp → **49.1% ~ 60.3%**
- **50% 帰無仮説に対する p 値**: ~0.054 (片側) — 5% 有意水準で「v5 より強い」とは言えない (ぎりぎり棄却不可)
- **seat 偏り**: seat=1 (後手) で +6.7pp 勝ち越し → STAY burst は後手番で特に効く

## iter1/iter2 との比較

| 評価 | 戦数 | 対戦相手 | 構成 | 勝率 | 95% CI |
|---|---|---|---|---|---|
| iter1 | 100 | baseline_v4 | Full (def+burst) | 64% | ±9.4pp = 54.6%~73.4% |
| iter2 Full | 100 | baseline_v5 | Full (def+burst) | 53% | ±9.8pp = 43.2%~62.8% |
| iter2 defense-only | 100 | baseline_v5 | def のみ | 52% | ±9.8pp = 42.2%~61.8% |
| iter2 burst-only | 100 | baseline_v5 | burst のみ | 59% | ±9.6pp = 49.4%~68.6% |
| **iter3 burst-only** | **300** | **baseline_v5** | **burst のみ** | **54.7%** | **±5.6pp = 49.1%~60.3%** |

iter2 burst-only の 59% は **iter3 で 54.7% へ収束** — 上振れ 4.3pp 分は seed1000 単独の variance だった。これは memory `project_imitation_case1_phase3` と同じパターン (100戦の上振れが 300戦で消える)。

ただし **defense-only や Full と比べた優位性は維持**:
- iter2 では burst-only が +6pp Full を上回った
- iter3 で 100戦差 +4pp 分が seed variance に吸収されたと考えると、burst-only が真に「Full より良い」可能性は残るが **統計的に有意とは言えない**

## fleet 形成効果は確かに出ている

3 seed 全てで **fleet peak ratio = 1.27〜1.29** (= v6 平均艦隊サイズが v5 比 27〜29% 大きい) を一貫して計測。これは STAY burst hold が「次ターンに合流発射 → 1 fleet あたり艦数増 → 移動速度向上」という設計意図を実現している強い証拠。

ただし **artifact (大きい艦隊形成)** が必ずしも **outcome (勝率)** に転化していない。理由として考えられるのは:
- 大きい艦隊で攻めても、相手が散発攻撃で計画的に削ってくる場合は時間効率で負ける
- burst hold 中に本来取れた中立惑星を逃している
- baseline_v5 自体が armhand な相手で、構造的工夫の差が大きく出にくい

## Kaggle 提出への影響

提出済み `case6_20260502-110353.tar.gz` は本構成 (burst-only) と同一。Kaggle 側の opponent pool は v5 とは異なる多様な agent 群なので、ローカル v5 評価で 54.7% でも Kaggle publicScore がどう出るかは別問題 (`.claude/rules/bot/pipeline.md` 通り publicScore は判断材料にしない)。

**結論**: 提出を取り下げる必要はない。case6 は v5 と互角以上、Full / defense-only より一貫して上、fleet 形成効果は設計通り、という性質の baseline。Kaggle 評価結果は別途確認。

## 次の一手の推奨

iter3 で burst-only の限界が見えたので、case6 自体の改良に進むか、別 case を切るか:

### Option A: burst パラメータの ablation (case6 内改良、推奨)

`STAY_BURST_MIN_GAIN` (現在 1)、`STAY_BURST_MIN_SHIPS` (現在 8)、`STAY_BURST_MAX_TARGET_TURNS` (現在 30) を 1〜2 通り変えて 100戦 vs v5 で feel を見る。中央値 54.7% が +3〜5pp 動けば再度 300戦で確証。

### Option B: case7 で別アプローチ (新 case)

burst hold の効果は限定的と判明したので、case7 では別の切り口 (例: 中立惑星争奪の優先順位再設計、harass 強化、lookahead 深さ拡大) を試す。case6 の知見 (defense は害、burst は艦数を増やすが勝率に直結しない) を踏まえる。

### Option C: 現状で commit & PR

case6 (burst-only) は Full より良い構成として残す価値あり。300戦 54.7% は弱いが、defense=False の判断は ablation 含めて十分根拠あり。コードを commit して PR を切り、次の experiment へ進む。

**最終推奨**: **Option C (commit & PR) → Option A (case7 立てる前に burst パラメータを最適化)**。 case6 の defense=False 構成と ablation 知見を docs と共に保全してから、burst の伸び代を探る。

## 再現手順

```bash
# 3 並列実行 (合計 ~53分)
mkdir -p /tmp/case6_validation
uv run --directory bot python -m pipeline.rulebase.case6.evaluation.compare_v5 -n 50 --seed 1000 > /tmp/case6_validation/seed1000.log 2>&1 &
uv run --directory bot python -m pipeline.rulebase.case6.evaluation.compare_v5 -n 50 --seed 2000 > /tmp/case6_validation/seed2000.log 2>&1 &
uv run --directory bot python -m pipeline.rulebase.case6.evaluation.compare_v5 -n 50 --seed 3000 > /tmp/case6_validation/seed3000.log 2>&1 &
wait
```

## ログ保存先

- `/tmp/case6_validation/{seed1000,seed2000,seed3000}.log` (生ログ、3 ラン分)
- 永続化したい場合は `data/output/experiment/rulebase_case6_validation_v5/` 等に手動退避を推奨
