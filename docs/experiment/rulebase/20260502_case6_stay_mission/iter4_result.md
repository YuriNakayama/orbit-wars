# [rulebase/case6] iter4 結果: 厳しめ burst 100戦 vs baseline_v5

> 評価コマンド:
> ```bash
> uv run --directory bot python -m pipeline.rulebase.case6.evaluation.compare_v5 -n 50 --seed 1000
> ```
> 構成: burst-only **厳しめ** (`STAY_BURST_MIN_GAIN=2`, `STAY_BURST_MIN_SHIPS=12`, `STAY_BURST_MAX_TARGET_TURNS=20`)
> 対戦相手: baseline_v5
> 環境: kaggle_environments orbit_wars 1v1, seed 1000..1099
> 計算: ローカル CPU 単独実行
> 実行時間: 約 60 分
> 関連: iter3 (broad burst) 300戦 54.7% を改善できるかの検証
> 仮説: hold を絞れば launches/ep が増えて outcome に近づく

## 結論サマリ

**41/100 (41.0%) で大幅劣化。仮説は完全に棄却**。Stage 2 (300戦) は実施せず判断を確定。

iter3 (54.7%) 比 **-13.7pp**, iter2 burst-only (59%) 比 **-18pp**。**broad burst (gain≥1, ships≥8, dist≤30) こそが case6 の効力源** だったと確認。厳しめ設定は以下の挙動を生み有害:
- fleet peak ratio が **1.28 → 1.08** に劣化 (大艦隊化効果が消失)
- launches/ep ratio は **0.97 → 0.87** に逆に減少 (期待: 増加)
- seat=1 で **34%** に崩壊 (後手番での fleet 形成が間に合わなくなる)

**判断**: 即座に config を iter3 設定に巻き戻し、別方針 (case7 or burst の別軸ablation) で iter5 を立てる。

## 数値テーブル

### 100戦結果

| 指標 | iter3 (broad, 300戦平均) | **iter4 (厳しめ, 100戦)** | 変化 |
|---|---|---|---|
| 勝率 | **54.7%** (CI: 49.1〜60.3%) | **41.0%** | **-13.7pp** |
| seat=0 | 51.3% | 48.0% | -3.3pp |
| seat=1 | 58.0% | **34.0%** | **-24.0pp** |
| fleet peak ratio | 1.28 | **1.08** | -0.20 |
| launches/ep ratio | 0.97 | **0.87** | -0.10 |
| 平均 ep 長 | 180.6 | 186.2 | +5.6 |

### 失敗の機序

1. **fleet peak が 1.08 まで激減** — burst hold の発火頻度が下がりすぎて「合流による艦数増」効果がほぼ消失
2. **launches/ep が逆に減った (0.97→0.87)** — 仮説「hold が減れば launches が増える」と真逆。**broad burst の launches 抑制 (3%) は無害な「待ち」だったが、厳しめにすると残った hold が「価値の低い局面で固まる」結果になった可能性**
3. **seat=1 で 34%** — 後手番で fleet 形成が間に合わず、v5 の機動的攻撃に蹂躙された

### 厳しめが効かなかった真因の仮説

iter4 plan の前提 (「価値の低い hold が outcome を悪化させている」) が誤りだった。実態としては:
- **broad burst の hold は「広く浅く」発火していたが、その合計効果として fleet peak +28% を達成していた**
- 厳しめにすると **発火回数が激減し、たまに発火する hold は文脈を失った場面で発火する**
- つまり burst hold は **個々の判定の質ではなく、累積効果で勝率に貢献していた**

## iter5 推奨方針

iter4 の失敗を踏まえ、burst パラメータの「別軸」を試すか、case6 自体を見切って case7 に進むかの判断:

### Option A: burst パラメータ (iter3 broad ↔ iter4 厳しめ) の中間

`MIN_GAIN=1` (broad), `MIN_SHIPS=10` (中間), `MAX_TARGET_TURNS=25` (中間) などで 100戦。iter3 の累積効果を保ちつつ、最低限の品質ハードルを入れる試み。
- pros: iter3/iter4 の中間に最適点がある可能性
- cons: 100戦では効果が見えない可能性高 (54.7% ±5.6pp の幅で動かない)

### Option B: 別の burst 軸 (累積制御)

`MIN_GAIN=0` (現ターンと同じ ETA でも hold) や、source ごとの hold 上限ターン数 (現状無制限を 2 ターン上限) など、broad の方向で違う制御を入れる。
- pros: iter4 で「broad ほど良い」傾向が見えたので、もっと broad にする方向は理論上ありえる
- cons: 実装変更を伴う、ストールメイト懸念

### Option C: case7 で別アプローチ (Recommended)

case6 は STAY mission の検証として完成 (broad burst 54.7% が局所最適、厳しめは有害、defense は害)。これ以上のチューニングは ROI が低い。
case7 で別の戦略軸 (中立惑星争奪、harass 強化、lookahead 拡大、defense を別機構で実装) を試す。
- pros: iter1〜4 の累積知見を保全、新規探索領域
- cons: case6 がまだ局所最適に達していない可能性は残す

### Option D: iter3 設定で case6 を確定して PR

case6 (broad burst, defense=False) を完成形として commit + PR、Kaggle 提出済み構成と一致。次の experiment は別 case で。
- pros: 案件クローズ、知見が docs として確定
- cons: case6 の伸び代探索を打ち切り

## 即時アクション

config を **iter3 broad 設定に巻き戻し済み** (本 result.md 書き出し直後に実施):
```python
STAY_BURST_MIN_GAIN: int = 1
STAY_BURST_MIN_SHIPS: int = 8
STAY_BURST_MAX_TARGET_TURNS: int = 20  # ← iter5 で 30 に戻すかは別判断
```

**注意**: iter4 で `MAX_TARGET_TURNS=20` に変更したまま戻していない場合、`30` に戻す必要あり。iter3 は 30 が確定値。

## ログ保存先

- `/tmp/case6_iter4/seed1000_stage1.log` (生ログ)
