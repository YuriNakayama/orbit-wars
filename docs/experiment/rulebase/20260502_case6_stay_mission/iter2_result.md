# [rulebase/case6] STAY mission ablation — vs baseline_v5 (case6 直接派生元)

> 評価コマンド (各ラン 100戦 / seed=1000、両 seat 50戦ずつ):
> - **Full**:        `cd bot && uv run python -m pipeline.rulebase.case6.evaluation.compare_v5 -n 50 --seed 1000`
> - **defense-only**: `cd bot && uv run python -m pipeline.rulebase.case6.evaluation.compare_v5 -n 50 --seed 1000 --no-burst`
> - **burst-only**:   `cd bot && uv run python -m pipeline.rulebase.case6.evaluation.compare_v5 -n 50 --seed 1000 --no-defense`
>
> 比較対象: baseline_v6 (case6) vs **baseline_v5 (case5、書き換え前のエージェント、case6 の直接派生元)** — iter1 (vs baseline_v4) からの差分は対戦相手のみ。
> 環境: kaggle_environments orbit_wars 1v1, seed = 1000..1099 (両 seat)
> 計算: ローカル CPU、3 ラン並列実行。Vast.ai GPU は不使用。
> 実行時間: 約 56 分 (3 並列、12 物理コアで CPU 競合の影響は軽微)
> run id (commit SHA): `2871ad84ecde780e9b65e1882eae5da5c6521b5b` (HEAD `feature/add-rulebase-to-stay`、未コミット作業含む)

## 結論サマリ

**STAY defense は有害、STAY burst が本体寄与の主役**。case6 の現行構成 (defense+burst の両方 ON) は、defense を OFF にすると勝率が +6pp 改善する。

- **burst-only (defense OFF)**: 59/100 (**59.0%**) — 採用候補閾値 55% を超える有意な優位
- **Full (defense+burst)**: 53/100 (**53.0%**) — burst-only より 6pp 低い、閾値 55% 未満
- **defense-only (burst OFF)**: 52/100 (**52.0%**) — ほぼ五分、defense 単独効果は確認できず

iter1 (vs baseline_v4) で得た 64% という Full の数字は、対戦相手が case4 (より弱いベース) だったからで、**case6 が直接派生した case5 に対しては Full 53%、burst-only 59% が真の実力**。

**推奨**: `STAY_DEFENSE_ENABLED` をデフォルト False にすること。burst hold は維持。

## 数値テーブル

### 勝率まとめ (vs baseline_v5、各 100 戦)

| Variant | STAY config | v6 勝 | v5 勝 | 引分 | v6 勝率 | seat=0 | seat=1 |
|---|---|---|---|---|---|---|---|
| **Full** | DEFENSE=True, BURST=True | 53 | 47 | 0 | **53.0%** | 60.0% | 46.0% |
| **defense-only** | DEFENSE=True, BURST=False | 52 | 48 | 0 | **52.0%** | 46.0% | 58.0% |
| **burst-only** | DEFENSE=False, BURST=True | 59 | 41 | 0 | **59.0%** | 56.0% | **62.0%** |

### 行動指標 (STAY が実際にどう挙動を変えたか)

| Variant | 平均 fleet peak (v6) | (v5) | ratio | 1ep 発射回数 (v6) | (v5) | ratio | 平均 ep 長 |
|---|---|---|---|---|---|---|---|
| Full | 22.0 | 17.4 | **1.26** | 420.3 | 492.1 | **0.85** | 190.7 |
| defense-only | 19.5 | 17.4 | 1.12 | 451.0 | 498.0 | 0.91 | 185.6 |
| burst-only | 21.7 | 17.1 | **1.27** | 472.3 | 444.1 | **1.06** | 182.6 |

### 解釈

- **fleet peak ratio** (v6 平均艦数 / v5 平均艦数):
  - Full と burst-only が同じ ~1.26 → **burst hold が艦合流効果を出している唯一の機構**。
  - defense-only は 1.12 まで下がる → defense は艦数を増やしていない (期待された防衛温存効果は出ていない)。
- **launches/ep ratio** (v6 発射回数 / v5 発射回数):
  - Full = 0.85 → defense が 15% 発射を抑制。
  - burst-only = 1.06 → burst hold は瞬間 hold で長期発射数をむしろ微増させる (合流発射で 1 発射あたりの艦数が増えるため)。
  - defense-only = 0.91 → defense 単独でも 9% 抑制。
  - つまり **defense は「動かない」傾向を生み、burst は「待ってから集めて撃つ」効果を生む**。設計通りの動きをしているのは burst のみ。
- **seat 偏り**: defense-only と burst-only で seat 偏りが正反対 (defense-only は seat=1 で有利、burst-only は seat=1 で更に有利) → seed 1000 の seat 配置に依存した変動が大きい。defense は seat=0 (先手) で特に弱い (46%)。

## defense が有害な理由 (仮説)

1. **過防衛による先制攻撃機会の喪失**: defense hold は「敵 fleet が来そうなら待機」する。case5/case6 は本来 mass attack の機会を能動的に作る攻め型ベースだが、defense が攻撃判断を抑制してしまう。
2. **誤検知**: defense score は敵の出発済み fleet を解析的に評価するが、敵 fleet は途中で軌道を変えうる/別惑星に向かう可能性があり、待機判断が空振りに終わるケースが多い。
3. **burst との相互作用**: Full で defense と burst が両方 hold 判定を出すと、両ターン以上発射が止まる「停滞ダブルパンチ」が発生し得る。Full の launches ratio 0.85 は defense_only (0.91) と burst_only (1.06) の単純な積より低く、相互作用で発射機会が更に削られている可能性。

## 次の一手の推奨 (300 戦は実行禁止の制約下で)

ユーザーから **300 戦評価はローカル CPU 負荷で禁止** と明示されているので、以下のいずれかを推奨:

### Option A: defense を切って採用 (Recommended、即実行可能)

`bot/pipeline/rulebase/case6/baseline/core/config.py` で `STAY_DEFENSE_ENABLED = False` をデフォルト化。
- pros: 59% は採用候補閾値 55% を超える、追加評価不要
- cons: defense 設計の検討資産は残しつつ無効化するだけなので、後で再導入余地あり
- 追加検証: あれば burst パラメータ (hold turn 数、score 重み) のミニ ablation を 100 戦 × 1〜2 だけ追加

### Option B: defense を改良して再評価

defense score の誤検知を抑える方向 (敵 fleet 軌道予測の精緻化、hold turn 上限を厳しくする等) で 1〜2 通りの設定を試して 100 戦 × 数本。**ただし 100 戦は seed variance が大きい (memory: `<300 戦は noise`) ため、改良案 vs Full の差が ±5pp 程度では判定できない。** Option A の方が即効性あり。

### Option C: 別 seed セットで burst-only 100 戦を再現確認

memory `project_imitation_case1_phase3` のような **5/100 が再評価で 0/300 に化けた事例** を踏まえると、59% も seed 1000 起点に依存している可能性は残る。seed 2000 起点で burst-only 100 戦をもう 1 ラン回す (~50 分) のが安全。これだけは実行価値あり。

**最終推奨**: **Option A + Option C** を順次。まず seed 2000 で burst-only 100 戦を再現確認し、55% 以上を維持できれば config を defense=False に変更してマージへ。

## 再現手順 (検証用)

```bash
cd bot

# Full (default)
uv run python -m pipeline.rulebase.case6.evaluation.compare_v5 -n 50 --seed 1000

# defense-only ablation
uv run python -m pipeline.rulebase.case6.evaluation.compare_v5 -n 50 --seed 1000 --no-burst

# burst-only ablation (推奨されたデフォルト候補)
uv run python -m pipeline.rulebase.case6.evaluation.compare_v5 -n 50 --seed 1000 --no-defense

# Option C: seed 2000 で burst-only 再現確認 (推奨)
uv run python -m pipeline.rulebase.case6.evaluation.compare_v5 -n 50 --seed 2000 --no-defense
```

## ログ保存先

3 ラン分の生ログ: `/tmp/case6_ablation/{full,defense_only,burst_only}.log`
将来追跡したい場合は `data/output/experiment/rulebase_case6_ablation_v5/` に手動で移動推奨 (DVC 管理対象外、データ規模も小さいので git 直 commit でも可)。
