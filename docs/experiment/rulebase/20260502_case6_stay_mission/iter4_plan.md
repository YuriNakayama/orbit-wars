# [rulebase/case6] iter4: burst パラメータ厳しめ ablation

## 仮説

iter3 で 300戦 vs v5 = 54.7% (95% CI: 49.1%~60.3%) の結果、artifact (fleet peak ratio 1.28) は出るが outcome (勝率) に十分転化していない。
原因は **burst hold が広すぎて、価値の低い hold (= 短い ETA gain、少ない合流量、遠い目標) で発射機会を逃している** 可能性。

→ **hold 判定を「価値の高いものだけ」に絞れば、launches/ep が増えて outcome に近づく** という仮説。

## 変更点

`backend/pipeline/rulebase/case6/baseline/core/config.py`:

| パラメータ | iter3 (現状) | iter4 (厳しめ) | 意図 |
|---|---|---|---|
| `STAY_BURST_MIN_GAIN` | 1 | **2** | hold 価値を「2 ターン以上の ETA 改善」に限定。1 ターンの微改善は無視 |
| `STAY_BURST_MIN_SHIPS` | 8 | **12** | 速度曲線の効果が顕著な大艦隊候補のみ hold |
| `STAY_BURST_MAX_TARGET_TURNS` | 30 | **20** | 遠い目標 (ETA 31〜30 ターン) は艦数効果が dilute するので除外 |

`STAY_DEFENSE_ENABLED` は `False` 維持 (iter2 で有害確定済み)。
`STAY_BURST_ENABLED` は `True` 維持。

## 期待効果

- hold 発生回数が ~30〜50% 減る (ETA 1 改善 / 8〜11 隻 / 距離 21〜30 ターンの hold が消える)
- 浮いた turn で発射機会が増え launches/ep ratio が **0.97 → 1.05+** に上昇
- fleet peak ratio は **1.28 → 1.15〜1.20** にやや低下 (許容)
- 勝率が **54.7% → 58〜62%** に上昇する想定

下振れシナリオ:
- launches は増えるが個々の fleet が弱体化して勝率が下がる (hold すべきだったケースを潰す)
- 効果なし (gain=1 の hold は実は重要だった)

## 評価方針

### Stage 1: 100戦 vs v5 で feel を見る (~50分)

```bash
uv run --directory backend python -m pipeline.rulebase.case6.evaluation.compare_v5 -n 50 --seed 1000
```

- 判定基準:
  - **57%+** → iter3 (54.7%) より明確に改善、Stage 2 (300戦) に進む
  - **52%~56%** → seed variance 内、Stage 2 で要確証
  - **52% 未満** → 厳しめは逆効果、別パラメータか別アプローチへ

### Stage 2: 300戦バリデーション (Stage 1 で改善が見えたら、~53分)

iter3 と同じく seed 1000/2000/3000 で 100戦ずつ並列。

```bash
uv run --directory backend python -m pipeline.rulebase.case6.evaluation.compare_v5 -n 50 --seed 1000 &
uv run --directory backend python -m pipeline.rulebase.case6.evaluation.compare_v5 -n 50 --seed 2000 &
uv run --directory backend python -m pipeline.rulebase.case6.evaluation.compare_v5 -n 50 --seed 3000 &
```

判定: 95% CI 下限が **52% を超えれば採用候補**、Kaggle 提出を更新検討。

## 非ゴール

- defense の再有効化はしない (iter2 で有害確定)
- burst の機構そのものは変えない (パラメータのみ)
- 別 case (case7) は立てない (case6 内のチューニング)
- Vast.ai は使わない (rulebase は学習なし)
- Kaggle 再提出は本 plan の判定後にユーザー承認を取って別途
